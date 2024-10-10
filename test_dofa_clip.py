import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import pdb
import timm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from open_clip.push_to_hf_hub import save_for_hf, save_config_for_hf
from open_clip import get_model_config
from einops import rearrange

# Load and modify the model
#model, preprocess = create_model_from_pretrained('hf-hub:XShadow/GeoLB-ViT-B-16-SigLIP')
#tokenizer = get_tokenizer('hf-hub:XShadow/GeoLB-ViT-B-16-SigLIP')

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')

def encode_image(model, image, wvs, normalize: bool = False):
    features = model.visual.trunk(image, wvs)
    return F.normalize(features, dim=-1) if normalize else features

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))


intermediate_outputs = {}

# Define the hook function
def hook_fn(module, input, output):
    intermediate_outputs['image_features'] = output

model.visual.trunk.norm.register_forward_hook(hook_fn)


#image = Image.open("download.png")

image = preprocess(image).unsqueeze(0).cuda()

#labels_list = ["A busy airport with many aeroplanes.", "Satellite view of Hohai university.", "Satellite view of sydney", "Many people in a stadium"]
labels_list = ["a dog", "a cat", "a donut", "a beignet"]

text = tokenizer(labels_list, context_length=model.context_length)
text = text.cuda()
model = model.cuda()

pdb.set_trace()

with torch.no_grad(), torch.cuda.amp.autocast():
    wvs = torch.tensor([0.665, 0.560, 0.490]).cuda()
    #image_features,_ = encode_image(model, image, wvs)
    image_features = model.encode_image(image)
    feats = intermediate_outputs['image_features']
    s_feats = rearrange(feats, "b (w h) c -> b c w h", w=27)
    s_feats = torch.nn.functional.adaptive_avg_pool2d(s_feats, (4,4))
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)


zipped_list = list(zip(labels_list, [round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)


