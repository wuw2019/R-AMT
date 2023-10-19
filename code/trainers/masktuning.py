import os.path as osp

from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import VisionTransformer

from modules.resnet import MaskModifiedResNet
from modules.visiontransformer import MaskVisionTransformer
from loss.reg_loss import RegLoss

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

CUSTOM_TEMPLATES = {
    "OxfordPets": ["a photo of a {}, a type of pet."],
    "OxfordFlowers": ["a photo of a {}, a type of flower."],
    "FGVCAircraft": ["a photo of a {}, a type of aircraft."],
    "DescribableTextures": ["{} texture."],
    "EuroSAT": ["a centered satellite photo of {}."],
    "StanfordCars": ["a photo of a {}."],
    "Food101": ["a photo of {}, a type of food."],
    "SUN397":["a photo of a {}."],
    "Caltech101":["a photo of a {}."],
    "UCF101": ["a photo of a person doing {}."],
    "ImageNet": ["a photo of a {}."],
    "ImageNetSketch": ["a photo of a {}."],
    "ImageNetV2": ["a photo of a {}."],
    "ImageNetA": ["a photo of a {}."],
    "ImageNetR": ["a photo of a {}."],
}

class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        clip_model = load_clip_to_cpu(cfg)
        if cfg.MASK.PREC == "fp32":
            clip_model.float()
        self.clip_model = clip_model.to('cuda')
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        text_features = []
        with torch.no_grad():
            for classname in self.classnames:
                classname = classname.replace('_', ' ')
                classname = classname.lower()
                texts = [t.format(classname) for t in temp]
                prompts = torch.cat([clip.tokenize(p) for p in texts])
                prompts = prompts.to('cuda')
                class_embeddings = self.clip_model.encode_text(prompts)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=0).cuda()
        return text_features

class MASKCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        origin_image_encoder = clip_model.visual

        clip_state_dict = clip_model.state_dict()
        vit = "visual.proj" in clip_state_dict
        self.vit = vit
        vision_layers, embed_dim, vision_heads, vision_width, image_resolution, vision_patch_size = self.get_params(vit,clip_state_dict)
        if vit:
            self.image_encoder = MaskVisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size,width=vision_width,layers=vision_layers,heads=vision_heads,output_dim=embed_dim,
                mask_init = cfg.MASK.INIT, mask_scale = cfg.MASK.SCALE, threshold_fn = cfg.MASK.THRESHOLD_FN, threshold=cfg.MASK.THRESHOLD, mask_mlp=cfg.MASK.MASK_MLP)
        else:
            self.image_encoder = MaskModifiedResNet(vision_layers, embed_dim, vision_heads, input_resolution=image_resolution, width=vision_width,
                mask_init = cfg.MASK.INIT, mask_scale = cfg.MASK.SCALE, threshold_fn = cfg.MASK.THRESHOLD_FN, threshold=cfg.MASK.THRESHOLD)
        
        self.make_model(origin_image_encoder)
        self.origin_image_encoder = origin_image_encoder

        text_encoder = TextEncoder(cfg, classnames)
        self.text_features = text_encoder()

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.threshold = cfg.MASK.THRESHOLD

    def get_params(self, vit, state_dict):

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
            vision_heads = vision_width // 64
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
            vision_heads = vision_width * 32 // 64

        embed_dim = state_dict["text_projection"].shape[1]
        return vision_layers, embed_dim, vision_heads, vision_width, image_resolution, vision_patch_size

    def make_model(self, origin_image_encoder):
        """Creates the model."""

        if self.vit:
            self.image_encoder.class_embedding.data.copy_(origin_image_encoder.class_embedding.data)
            self.image_encoder.positional_embedding.data.copy_(origin_image_encoder.positional_embedding.data)
            self.image_encoder.proj.data.copy_(origin_image_encoder.proj.data)

        # Copy weights from the pretrained to the modified model.
        for module, module_pretrained in zip(self.image_encoder.modules(), origin_image_encoder.modules()):
            if 'MultiheadAttention' in str(type(module)):
                module.in_proj_weight.data.copy_(module_pretrained.in_proj_weight.data)
                if module.in_proj_bias is not None:
                    module.in_proj_bias.data.copy_(module_pretrained.in_proj_bias.data)

                module.out_proj.weight.data.copy_(module_pretrained.out_proj.weight.data)
                if module.out_proj.bias is not None:
                    module.out_proj.bias.data.copy_(module_pretrained.out_proj.bias.data)
            elif 'ElementWise' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                if module.bias is not None:
                    module.bias.data.copy_(module_pretrained.bias.data)
            elif 'Linear' in str(type(module)) or 'Conv2d' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                if module.bias is not None:
                    module.bias.data.copy_(module_pretrained.bias.data)
            elif 'BatchNorm' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                module.bias.data.copy_(module_pretrained.bias.data)
                module.running_mean.copy_(module_pretrained.running_mean)
                module.running_var.copy_(module_pretrained.running_var)
            elif 'LayerNorm' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                module.bias.data.copy_(module_pretrained.bias.data)
            elif 'MaskAttentionPool' in str(type(module)):
                module.positional_embedding.data.copy_(module_pretrained.positional_embedding.data)
                for attnpool_module, attnpool_module_pretrained in zip(self.image_encoder.attnpool.modules(), origin_image_encoder.attnpool.modules()):
                    if 'ElementWise' in str(type(attnpool_module)):
                        attnpool_module.weight.data.copy_(attnpool_module_pretrained.weight.data)
                        if attnpool_module.bias is not None:
                            attnpool_module.bias.data.copy_(attnpool_module_pretrained.bias.data)
            

        print('Creating model: Mask layers created.')

        self.shared = nn.Sequential()
        for name, module in self.image_encoder.named_children():
            self.shared.add_module(name, module)

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        if self.shared.training:
            return logits, image_features
        else: return logits
        

    def original_forward(self, image):
        image_features = self.origin_image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    
    def compute_sparsity(self, threshold_fn):
        total_zeros = 0.0
        total_param = 0.0
        
        weights = self.shared.state_dict()
        for k in list(weights.keys()):
            if 'mask_real' not in k:
                continue
            if threshold_fn == 'binarizer':
                num_zero = weights[k].lt(self.threshold).sum()
                num_param = weights[k].data.numel()
            
            total_param += num_param
            total_zeros += num_zero
            
        return (total_zeros)/total_param*100.

class KLLoss(_Loss):
    def __init__(self, T, alpha=1.):
        super(KLLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, stu_logits, tea_logits, label):
        tea_logits = self.alpha * tea_logits+ (1 - self.alpha) * stu_logits
        
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return kl_loss


@TRAINER_REGISTRY.register()
class MaskTuning(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        clip_model.float()

        print("Building custom CLIP")
        self.model = MASKCLIP(cfg, classnames, clip_model)
    
        for name, param in self.model.shared.named_parameters():
            param.requires_grad_(False)
            if 'mask_real' in name:
                param.requires_grad_(True)
                
        # # Double check
        enabled = set()
        for name, param in self.model.shared.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")


        param_groups = [
            {
                "params" : self.model.shared.parameters(),
            },
            ]

        trainable_param = sum(p.numel() for p in self.model.shared.parameters() if p.requires_grad)

        self.model.to(self.device)
        # NOTE: only give mask to the optimizer
        self.optim = build_optimizer(None, cfg.OPTIM, param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model("shared", self.model.shared, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.kl_loss = KLLoss(T=self.cfg.MASK.GDR_T, alpha=self.cfg.MASK.FUSE_ALPHA)
        self.count = []

    def forward_backward(self, batch):
        self.set_model_mode("train")
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        n_iter = self.epoch * self.num_batches + self.batch_idx

        # t0 = time.time()
        logits, image_feat = model(image)
        loss = F.cross_entropy(logits, label)
        loss_summary = {"main_loss": loss.item()}
        if self.cfg.MASK.MASK_LOSS:
            reg_loss = 0
            for name, param in model.shared.named_parameters():
                if 'mask_real' in name:
                    reg_loss += RegLoss(param, 2)
            mask_loss = self.cfg.MASK.LOSS_WEIGHT*reg_loss
            loss += mask_loss
            loss_summary["mask loss"] = mask_loss.item()

        tea_logits = model.original_forward(image)

        if self.cfg.MASK.GDR:
            kl_loss = self.kl_loss(logits, tea_logits, label)
            self.graddrop_backward_and_update(loss, kl_loss, self.cfg.MASK.GDR_LAMBDA)
            loss_summary["kl loss"] = kl_loss.item()

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_feat @ model.text_features.t()
        loss_summary["acc"] = compute_accuracy(logits, label)[0].item()
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def graddrop_backward_and_update(
        self, loss_a, loss_b, lambda_=1, names=None, epsilon=1e-8
    ):
        # print('=================use grad drop===============')

        # loss_b not increase is okay
        # loss_a has to decline
        self.model_zero_grad(names)
        # get name of the model parameters
        names = self.get_model_names(names)
        # backward loss_a
        self.detect_anomaly(loss_b)
        loss_b.backward(retain_graph=True)
        # normalize gradient
        b_grads = []
        for name in names:
            for p in self._models[name].parameters():
                if p.grad is not None:
                    b_grads.append(p.grad.clone())
                else: b_grads.append(None)

        # optimizer don't step
        for name in names:
            self._optims[name].zero_grad()

        # backward loss_a
        self.detect_anomaly(loss_a)
        loss_a.backward()
        for name in names:
            for p, b_grad in zip(self._models[name].parameters(), b_grads):
                if b_grad is not None:
                    a_grad = p.grad.clone()
                    sgn_a_grad = torch.sign(a_grad)
                    
                    # calculate the possibility to keep grad_a, ce loss
                    # for element i
                    # if a_grad_i < 0, b_grad_i < 0 => sgn_a_grad<0 P=1
                    # if a_grad_i > 0, b_grad_i > 0 => sgn_a_grad>0 P=1
                    P = (sgn_a_grad*(a_grad+b_grad)/(torch.abs(a_grad)+torch.abs(b_grad)+epsilon)+1)/2

                    # U is the threshold to keep a_grad
                    U = torch.rand(a_grad.shape, device=a_grad.device)
                    # lambda_<1. means the drop the grad to a small value 
                    # lambda_=1. means the drop the grad to a 0
                    p.grad = ((1.-lambda_)+ lambda_*(P > U))*a_grad


        # optimizer
        for name in names:
            self._optims[name].step()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def before_train(self):
        super().before_train()

        # calculate init sparsity
        sparsity = self.model.compute_sparsity(self.cfg.MASK.THRESHOLD_FN)
        print("++++++++++++ Init Sparsity: ",sparsity,"++++++++++++")


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the last model is loaded
        model_file = "model.pth.tar-" + str(self.cfg.OPTIM.MAX_EPOCH)

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            if 'shared' in name:
                def decode(mask, int_value):
                    # decode binary values from bytes
                    shape = mask.view(-1).shape[0]

                    bin_str = ''.join('{:08b}'.format(c) for c in int_value)
                    bin_str = bin_str[:shape]
                    decoded_mask = torch.FloatTensor([int(bin_str[i]) for i in range(len(bin_str))])
                    decoded_mask = decoded_mask.reshape_as(mask)
                    decoded_mask = decoded_mask.to(mask.device)
                    return decoded_mask
                # decode mask from bytes
                model_state_dict = self._models[name].state_dict()
                for key in model_state_dict.keys():
                    # pass
                    if 'mask_real' in key:
                        mask = decode(model_state_dict[key], state_dict[key])
                        model_state_dict[key] = mask.data #.cpu()
                state_dict = model_state_dict

            self._models[name].load_state_dict(state_dict, strict=True)
            sparsity = self.model.compute_sparsity(self.cfg.MASK.THRESHOLD_FN)
    

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
                for module in self._models[name].modules():
                    if 'BatchNorm' in str(type(module)):
                        module.eval()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError
            
    @torch.no_grad()
    def test(self, split=None, during_train=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        if during_train: return list(results.values())[0]

        sparsity = self.model.compute_sparsity(self.cfg.MASK.THRESHOLD_FN)
        print("++++++++++++ Sparsity: ",sparsity,"++++++++++++")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
        
    def check(self):

        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)
        
        clip_model.float()
        pretrained = MASKCLIP(cfg, classnames, clip_model)
        for module, module_pretrained in zip(self.model.shared.modules(), pretrained.shared.modules()):
            if 'ElementWise' in str(type(module)) or 'BatchNorm' in str(type(module)) or 'LayerNorm' in str(type(module)):
                weight = module.weight.data.cpu()
                weight_pretrained = module_pretrained.weight.data.cpu()
                # Using small threshold of 1e-8 for any floating point inconsistencies.
                # Note that threshold per element is even smaller as the 1e-8 threshold
                # is for sum of absolute differences.
                assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                    'module %s failed check' % (module)
                if module.bias is not None:
                    bias = module.bias.data.cpu()
                    bias_pretrained = module_pretrained.bias.data.cpu()
                    assert (bias - bias_pretrained).abs().sum() < 1e-8
                if 'BatchNorm' in str(type(module)):
                    rm = module.running_mean.cpu()
                    rm_pretrained = module_pretrained.running_mean.cpu()
                    assert (rm - rm_pretrained).abs().sum() < 1e-8
                    rv = module.running_var.cpu()
                    rv_pretrained = module_pretrained.running_var.cpu()
                    assert (rv - rv_pretrained).abs().sum() < 1e-8
        
        assert (self.model.image_encoder.attnpool.positional_embedding.data.cpu() - clip_model.visual.attnpool.positional_embedding.data.cpu()).abs().sum() < 1e-8
        
        print('Passed checks...')

        
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if last_epoch:
            last_result = self.test(split="test")
            self.save_model(self.epoch, self.output_dir)

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:

            model_dict = self._models[name].state_dict()
            if 'shared' in name:
                # save binary values into  bytes
                binarized_model_dict = {}
                def binarized_mask(mask):
                    s = ''.join('%s' % int(m) for m in mask)
                    l = len(s)
                    if r := l%8:
                        s += '0'*(8-r)
                    value = bytes([int(s[i:i+8],2) for i in range(0,len(s),8)])
                    return value
                    
                for key in model_dict.keys():
                    if 'mask_real' in key:
                        mask = model_dict[key].clone()
                        mask[model_dict[key].le(self.cfg.MASK.THRESHOLD)] = 0
                        mask[model_dict[key].gt(self.cfg.MASK.THRESHOLD)] = 1
                        mask = mask.view(-1).data.cpu()
                        binarized_model_dict[key] = binarized_mask(mask)
                model_dict = binarized_model_dict

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )