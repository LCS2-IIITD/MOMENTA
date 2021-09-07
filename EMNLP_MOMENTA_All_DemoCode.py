#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import all the dependencies
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchnlp import encoders
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


torch.cuda.empty_cache()


# COVID DATA

# Either load the pre-saved ROI/Entity features, or compute them on demand.
# Load the training, validation and test sets for the corresponding experimental setup as per the requirement

# In[ ]:


# # Load the ROI features (Covid)
# train_ROI = torch.load("path_to_features/harmeme_cov_train_ROI.pt")
# val_ROI = torch.load("path_to_features/harmeme_cov_val_ROI.pt")
# # test_ROI = torch.load("path_to_features/harmeme_cov_test_ROI.pt")
# # Load the ENT features
# train_ENT = torch.load("path_to_features/harmeme_cov_train_ent.pt")
# val_ENT = torch.load("path_to_features/harmeme_cov_val_ent.pt")
# # test_ENT = torch.load("path_to_features/harmeme_cov_test_ent.pt")


# In[ ]:


# Harmful Meme dataset (Covid-Ternary data)
data_dir_cov = "path_to_images/images"
train_path_cov = "path_to_jsonl/train.jsonl"
dev_path_cov   = "path_to_jsonl/val.jsonl"
# test_path_cov  = "path_to_jsonl/test.jsonl"


# In[ ]:


# train_samples_frame = pd.read_json(train_path_cov, lines=True)
# train_samples_frame.head()


# In[ ]:


# test_samples_frame = pd.read_json(test_path_cov, lines=True)
# test_samples_frame.head()


# POLITICAL DATA

# In[ ]:


# # # Load the ROI features (Political)
# train_ROI = torch.load("path_to_features/harmeme_pol_train_ROI.pt")
# val_ROI = torch.load("path_to_features/harmeme_pol_val_ROI.pt")
# test_ROI = torch.load("path_to_features/harmeme_pol_test_ROI.pt")
# # # Load the ENT features
# train_ENT = torch.load("path_to_features/harmeme_pol_train_ent.pt")
# val_ENT = torch.load("path_to_features/harmeme_pol_val_ent.pt")
# test_ENT = torch.load("path_to_features/harmeme_pol_test_ent.pt")


# In[ ]:


# Harmful Meme dataset (Political-Binary)
data_dir_pol = 'path_to_images/images'
train_path_pol = 'path_to_jsonl/train.jsonl'
dev_path_pol = "path_to_jsonl/val.jsonl"
test_path_pol = "path_to_jsonl/test.jsonl"


# In[ ]:


# train_samples_frame = pd.read_json(train_path_pol, lines=True)
# train_samples_frame.head()


# In[ ]:


test_samples_frame = pd.read_json(test_path_pol, lines=True)
test_samples_frame.head()


# Text pre-processing block for CLIP input

# In[ ]:


#@title

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


# In[ ]:


# Downloadig the clip model
# MODELS = {
#     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
#     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
#     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
#     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",    
# }
# ! wget {MODELS["ViT-B/32"]} -O clip_model.pt


# In[ ]:


clip_model = torch.jit.load("clip_model.pt").cuda().eval()
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# In[ ]:


preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
    ])
tokenizer = SimpleTokenizer()


# In[ ]:


# Get the image features for a single image input
def process_image_clip(in_img):
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    
    image = preprocess(Image.open(in_img).convert("RGB"))
    
    image_input = torch.tensor(np.stack(image)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    return image_input


# In[ ]:


# Get the text features for a single text input
def process_text_clip(in_text):    
    text_token = tokenizer.encode(in_text)
    text_input = torch.zeros(clip_model.context_length, dtype=torch.long)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokens = [sot_token] + text_token[:75] + [eot_token]
    text_input[:len(tokens)] = torch.tensor(tokens)
    text_input = text_input.cuda()
    return text_input


# ### vgg16

# VGG16 for extracting image encodings.

# In[ ]:


import torch
from torch import optim, nn
from torchvision import models, transforms
import cv2


# In[ ]:


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
  
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 

# Initialize the model
model_vgg_pretrained = models.vgg16(pretrained=True)
model_vgg = FeatureExtractor(model_vgg_pretrained)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model_vgg = model_vgg.to(device)


# For cropped ROI proposals

# In[ ]:


from tqdm import tqdm
import numpy as np

# Transform the image, so it becomes readable with the model
transform_vgg_BB = transforms.Compose([
  transforms.ToPILImage(),
#   transforms.CenterCrop(512),
  transforms.Resize((448,448)),
  transforms.ToTensor()                              
])


# Iterate each image
def get_image_vgg_BB(l, t, r, b, in_im): 
#     left, top, right, bottom and input image
    img = cv2.imread(in_im)
    h, w, _ = img.shape
    # crop
    x1 = int(np.floor(l*w))
    x2 = int(np.floor(r*w))
    y1 = int(np.floor(b*h))
    y2 = int(np.floor(t*h))
    crop_img = img[y1:y2, x1:x2]    
    
    # Transform the cropped image
    img = transform_vgg_BB(crop_img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = model_vgg(img).squeeze()
    # Convert to NumPy Array, Reshape it, and save it to features variable
    return feature


# In[ ]:


from tqdm import tqdm
import numpy as np

# Transform the image, so it becomes readable with the model
transform_vgg_center = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

# Iterate each image
def get_image_vgg_center(in_im):
    # Set the image path
    path = in_im
    # Read the file
    img = cv2.imread(path)
    # Transform the image
    img = transform_vgg_center(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = model_vgg(img).squeeze()
    
    return feature


# ### Sentence embedding
# https://github.com/UKPLab/sentence-transformers

# In[ ]:


from sentence_transformers import SentenceTransformer
model_sent_trans = SentenceTransformer('paraphrase-distilroberta-base-v1')


# ## Harmeme dataset ROI+ENT augmentation all return (both one-hot and numeric)

# In[ ]:


class HarmemeMemesDatasetAug2(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        split_flag=None,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )
        if split_flag=='train':
            self.ROI_samples = train_ROI
            self.ENT_samples = train_ENT
        elif split_flag=='val':
            self.ROI_samples = val_ROI
            self.ENT_samples = val_ENT
        else:
            self.ROI_samples = test_ROI
            self.ENT_samples = test_ENT
        
    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]
        img_file_name = self.samples_frame.loc[idx, "image"]
        
        image_clip_input = process_image_clip(self.samples_frame.loc[idx, "image"])
# --------------------------------------------------------------------------------------        
#         Pre-extracted features
        image_vgg_feature = self.ROI_samples[idx]        
# --------------------------------------------------------------------------------------
# On-demand computation
#         BB_info = self.samples_frame.loc[idx, "bbdict"]
#         roi_vgg_feat_list = []
#         if BB_info:
#             total_BB = len(BB_info)
#             if total_BB>4:
#                 BB_info_final = BB_info[:4]
#             else:
#                 BB_info_final = BB_info
# #             Have to get VGG reps for each cropped BB and get the mean             
#             for item in BB_info_final:
# #                 Get the top left (left,top) and bottom right (right,bottom) values of the coordinates
# #                 top left and bottom right value extraction                
#                 left   = item['Vertices'][3][0]
#                 top    = item['Vertices'][3][1]
#                 right  = item['Vertices'][1][0]
#                 bottom = item['Vertices'][1][1]
#                 get_image_vgg_center(img_file_name)
#                 roi_vgg_feat = get_image_vgg_BB(left, top, right, bottom, img_file_name)
#                 roi_vgg_feat_list.append(roi_vgg_feat)
# #             print(np.shape(roi_vgg_feat_list))
# #             print(torch.cat(roi_vgg_feat_list, dim=0))
# #             print(np.mean(np.array(roi_vgg_feat_list), axis=0))
#             image_vgg_feature = torch.mean(torch.vstack(roi_vgg_feat_list), axis=0)
# #             print(image_vgg_feature.shape)
#         else:
#             image_vgg_feature = torch.tensor(get_image_vgg_center(img_file_name))
# --------------------------------------------------------------------------------------
        text_clip_input = process_text_clip(self.samples_frame.loc[idx, "text"])
#         -------------------------------------------------------------------------------
#         Process entities
        #         Use them directly from the saved files
        text_drob_feature = self.ENT_samples[idx]
#         -------------------------------------------------------------------------------
#         Get the mean representation for the set of entities ""on-demand
#         cur_ent_rep_list = []
#         cur_ent_list = self.samples_frame.loc[idx, "ent"]
        
#         if len(cur_ent_list):
#             for item in cur_ent_list:
#                 cur_ent_rep = torch.tensor(model_sent_trans.encode(item)).to(device)
#                 cur_ent_rep_list.append(cur_ent_rep)
#             text_drob_feature = torch.mean(torch.vstack(cur_ent_rep_list), axis=0)
#         else:
#             text_drob_feature = torch.tensor(model_sent_trans.encode(self.samples_frame.loc[idx, "text"])).to(device)
#         -------------------------------------------------------------------------------

        if "labels" in self.samples_frame.columns:
#             label = torch.Tensor(
#                 [self.samples_frame.loc[idx, "label"]]
#             ).long().squeeze()

#             Uncoment below for binary index creation
#             if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
#                 lab=0
#             else:
#                 lab=1            
#             label = torch.tensor(lab).to(device)  

#             Uncomment below for one hot encoding
#             y = torch.tensor(lab).to(device)
#             label = F.one_hot(y, num_classes=2)  

# #             Multiclass setting - harmfulness
            if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
                lab=0
            elif self.samples_frame.loc[idx, "labels"][0]=="somewhat harmful":
                lab=1  
            else:
                lab=2
            label = torch.tensor(lab).to(device)  

            
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "image_vgg_feature": image_vgg_feature,
                "text_clip_input": text_clip_input,
                "text_drob_embedding": text_drob_feature,
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image_clip_input": image_clip_input,
                "image_vgg_feature": image_vgg_feature,
                "text_clip_input": text_clip_input,
                "text_drob_embedding": text_drob_feature
            }

        return sample


# In[ ]:


# hm_dataset_train = HarmemeMemesDatasetAug2(train_path_cov, data_dir_cov, 'train')
# dataloader_train = DataLoader(hm_dataset_train, batch_size=64,
#                         shuffle=True, num_workers=0)
# hm_dataset_val = HarmemeMemesDatasetAug2(dev_path_cov, data_dir_cov, 'val')
# dataloader_val = DataLoader(hm_dataset_val, batch_size=64,
#                         shuffle=True, num_workers=0)
# hm_dataset_test = HarmemeMemesDatasetAug2(test_path_pol, data_dir_pol, 'test')
# dataloader_test = DataLoader(hm_dataset_test, batch_size=64,
#                         shuffle=False, num_workers=0)


# In[ ]:


hm_dataset_train = HarmemeMemesDatasetAug2(train_path_pol, data_dir_pol, split_flag='train')
dataloader_train = DataLoader(hm_dataset_train, batch_size=64,
                        shuffle=True, num_workers=0)
hm_dataset_val = HarmemeMemesDatasetAug2(dev_path_pol, data_dir_pol, split_flag='val')
dataloader_val = DataLoader(hm_dataset_val, batch_size=64,
                        shuffle=True, num_workers=0)
hm_dataset_test = HarmemeMemesDatasetAug2(test_path_pol, data_dir_pol, split_flag='test')
dataloader_test = DataLoader(hm_dataset_test, batch_size=64,
                        shuffle=False, num_workers=0)


# ### MODEL

# Provide the specific model definition/module here

# In[ ]:


# Get the cross attention value features 
# Vanilla model

class MM(nn.Module):
    def __init__(self, n_out):
        super(MM, self).__init__()  
        
        self.dense_vgg_1024 = nn.Linear(4096, 1024)
        self.dense_vgg_512 = nn.Linear(1024, 512)
        self.drop20 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.05) 
        
        self.dense_drob_512 = nn.Linear(768, 512)
        
        self.gen_key_L1 = nn.Linear(512, 256) # 512X256
        self.gen_query_L1 = nn.Linear(512, 256) # 512X256
        self.gen_key_L2 = nn.Linear(512, 256) # 512X256
        self.gen_query_L2 = nn.Linear(512, 256) # 512X256
        self.gen_key_L3 = nn.Linear(512, 256) # 512X256
        self.gen_query_L3 = nn.Linear(512, 256) # 512X256
#         self.gen_value = nn.Linear(512, 256) # 512X256
        self.soft = nn.Softmax(dim=1)
        self.soft_final = nn.Softmax(dim=1)
        self.project_dense_512a = nn.Linear(1024, 512) # 512X256
        self.project_dense_512b = nn.Linear(1024, 512) # 512X256
        self.project_dense_512c = nn.Linear(1024, 512) # 512X256 
        
        
        self.fc_out = nn.Linear(512, 256) # 512X256
        self.out = nn.Linear(256, n_out) # 512X256
        

    def selfattNFuse_L1a(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L1(vec1))
            k1 = F.relu(self.gen_key_L1(vec1))
            q2 = F.relu(self.gen_query_L1(vec2))
            k2 = F.relu(self.gen_key_L1(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512a(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    def selfattNFuse_L1b(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L2(vec1))
            k1 = F.relu(self.gen_key_L2(vec1))
            q2 = F.relu(self.gen_query_L2(vec2))
            k2 = F.relu(self.gen_key_L2(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512b(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    
    def selfattNFuse_L2(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L3(vec1))
            k1 = F.relu(self.gen_key_L3(vec1))
            q2 = F.relu(self.gen_query_L3(vec2))
            k2 = F.relu(self.gen_key_L3(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.project_dense_512c(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep


    def forward(self, in_CI, in_VGG, in_CT, in_Drob):        
        VGG_feat = self.drop20(F.relu(self.dense_vgg_512(self.drop20(F.relu(self.dense_vgg_1024(in_VGG))))))
        Drob_feat = self.drop5(F.relu(self.dense_drob_512(in_Drob)))
        out_img = self.selfattNFuse_L1a(VGG_feat, in_CI)
        out_txt = self.selfattNFuse_L1b(Drob_feat, in_CT)        
        out_img_txt = self.selfattNFuse_L2(out_img, out_txt)
        final_out = F.relu(self.fc_out(out_img_txt))
#         out = torch.sigmoid(self.out(final_out)) #For binary case
        out = self.out(final_out)
        return out


# In[ ]:


# output_size = 1 #Binary case
output_size = 3
exp_name = "EMNLP_MCHarm_GLAREAll_COVTrain_POLEval"
# pre_trn_ckp = "EMNLP_MCHarm_GLAREAll_COVTrain" # Uncomment for using pre-trained
exp_path = "path_to_saved_files/EMNLP_ModelCkpt/"+exp_name
lr=0.001
# criterion = nn.BCELoss() #Binary case
criterion = nn.CrossEntropyLoss()
# # ------------Fresh training------------
model = MM(output_size)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


# In[ ]:


# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total trainable parameters are: {total_params}")


# ## Training

# https://github.com/Bjarten/early-stopping-pytorch

# In[ ]:


# import EarlyStopping
import os, sys
sys.path.append('path_to_the_module/early-stopping-pytorch')
from pytorchtools import EarlyStopping


# In[ ]:


# # For BCE loss all features
# def train_model(model, patience, n_epochs):
#     epochs = n_epochs
# #     clip = 5

#     train_acc_list=[]
#     val_acc_list=[]
#     train_loss_list=[]
#     val_loss_list=[]
    
#     # initialize the experiment path
#     Path(exp_path).mkdir(parents=True, exist_ok=True)
#     # initialize early_stopping object
#     chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
#     early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)

#     model.train()
#     for i in range(epochs):
#         total_acc_train = 0
#         total_loss_train = 0

#         for data in dataloader_train:
            
# #             Clip features...
#             img_inp_clip = data['image_clip_input']
#             txt_inp_clip = data['text_clip_input']
#             with torch.no_grad():
#                 img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
#                 txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

#             img_feat_vgg = data['image_vgg_feature']
# #             txt_feat_trans = data['text_drob_embedding']

#             label = data['label'].to(device)

#             model.zero_grad(), 
#             output = model(img_feat_clip, img_feat_vgg, txt_feat_clip)
# #             output = model(img_feat_vgg, txt_feat_trans)

#             loss = criterion(output.squeeze(), label.float())
            
# #             print(loss)
#             loss.backward()
# #             nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()

#             with torch.no_grad():
# #                 print(output.squeeze().shape)
# #                 print(label.float().shape)
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
#                 total_loss_train += loss.item()

#         train_acc = total_acc_train/len(dataloader_train)
#         train_loss = total_loss_train/len(dataloader_train)
#         model.eval()
#         total_acc_val = 0
#         total_loss_val = 0

#         with torch.no_grad():
#             for data in dataloader_val:                
# #                 Clip features...                
#                 img_inp_clip = data['image_clip_input']
#                 txt_inp_clip = data['text_clip_input']
#                 with torch.no_grad():
#                     img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
#                     txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)
                
                
#                 img_feat_vgg = data['image_vgg_feature']                
# #                 txt_feat_trans = data['text_drob_embedding']

                

#                 label = data['label'].to(device)

#                 model.zero_grad()
                
#                 output = model(img_feat_clip, img_feat_vgg, txt_feat_clip)
# #                 output = model(img_feat_vgg, txt_feat_trans)
                

#                 val_loss = criterion(output.squeeze(), label.float())
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_val += acc
#                 total_loss_val += val_loss.item()
#         print("Saving model...")         
        
#         torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))

#         val_acc = total_acc_val/len(dataloader_val)
#         val_loss = total_loss_val/len(dataloader_val)

#         train_acc_list.append(train_acc)
#         val_acc_list.append(val_acc)
#         train_loss_list.append(train_loss)
#         val_loss_list.append(val_loss)
        
#         early_stopping(val_loss, model)
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break
            
#         print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
#         model.train()
#         torch.cuda.empty_cache()
        
#     # load the last checkpoint with the best model
# #     model.load_state_dict(torch.load(chk_file))
    
#     return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        


# In[ ]:


# For cross entropy loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):
#         total_acc_train = 0
        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:
            
#             Clip features...
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']

            label_train = data['label'].to(device)

            model.zero_grad()
            output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#             print(output.shape)
#             output = model(img_feat_vgg, txt_feat_trans)

            loss = criterion(output.squeeze(), label_train)
            
#             print(loss)
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
                _, predicted_train = torch.max(output.data, 1)
                total_train += label_train.size(0)
                correct_train += (predicted_train == label_train).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
                total_loss_train += loss.item()

        
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        model.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
#                 Clip features...                
                img_inp_clip = data['image_clip_input']
                txt_inp_clip = data['text_clip_input']
                with torch.no_grad():
                    img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                    txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)
                
                
                img_feat_vgg = data['image_vgg_feature']                
                txt_feat_trans = data['text_drob_embedding']

                

                label_val = data['label'].to(device)

                model.zero_grad()
                
                output = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)
#                 output = model(img_feat_vgg, txt_feat_trans)
                
                
                val_loss = criterion(output.squeeze(), label_val)
                _, predicted_val = torch.max(output.data, 1)
                total_val += label_val.size(0)
                correct_val += (predicted_val == label_val).sum().item()                
                total_loss_val += val_loss.item()
        print("Saving model...") 
        torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))

        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
#     model.load_state_dict(torch.load('checkpoint_1.pt'))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        


# ## Testing

# In[ ]:


# # For BCE loss
# def test_model(model):
#     model.eval()
#     total_acc_test = 0
#     total_loss_test = 0
#     outputs = []
#     test_labels=[]
#     with torch.no_grad():
#         for data in dataloader_test:
#             img_inp_clip = data['image_clip_input']
#             txt_inp_clip = data['text_clip_input']
#             with torch.no_grad():
#                 img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
#                 txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

#             img_feat_vgg = data['image_vgg_feature']
# #             txt_feat_trans = data['text_drob_embedding']            

#             label = data['label'].to(device)
            
# #             out = model(img_feat_vgg, txt_feat_trans)        

#             out = model(img_feat_clip, img_feat_vgg, txt_feat_clip)        

#             outputs += list(out.cpu().data.numpy())
#             loss = criterion(out.squeeze(), label.float())
# #             print(out.squeeze())
# #             print(label.float())
#             acc = torch.abs(out.squeeze() - label.float()).view(-1)
#     #         print((acc.sum() / acc.size()[0]))
#             acc = (1. - acc.sum() / acc.size()[0])
#     #         print(acc)
#             total_acc_test += acc
#             total_loss_test += loss.item()

#     acc_test = total_acc_test/len(dataloader_test)
#     loss_test = total_loss_test/len(dataloader_test)
#     print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
#     return outputs


# In[ ]:


# For CE loss
def test_model(model):
    model.eval()
    total_test = 0
    correct_test =0
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:
            img_inp_clip = data['image_clip_input']
            txt_inp_clip = data['text_clip_input']
            with torch.no_grad():
                img_feat_clip = clip_model.encode_image(img_inp_clip).float().to(device)
                txt_feat_clip = clip_model.encode_text(txt_inp_clip).float().to(device)

            img_feat_vgg = data['image_vgg_feature']
            txt_feat_trans = data['text_drob_embedding']            

            label_test = data['label'].to(device)
            
#             out = model(img_feat_vgg, txt_feat_trans)        

            out = model(img_feat_clip, img_feat_vgg, txt_feat_clip, txt_feat_trans)        

            outputs += list(out.cpu().data.numpy())
            loss = criterion(out.squeeze(), label_test)
            
            _, predicted_test = torch.max(out.data, 1)
            total_test += label_test.size(0)
            correct_test += (predicted_test == label_test).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
            total_loss_test += loss.item()
            
            
#     #         print(label.float())
#             acc = torch.abs(out.squeeze() - label.float()).view(-1)
#     #         print((acc.sum() / acc.size()[0]))
#             acc = (1. - acc.sum() / acc.size()[0])
#     #         print(acc)
#             total_acc_test += acc
#             total_loss_test += loss.item()

    
    acc_test = 100 * correct_test / total_test
    loss_test = total_loss_test/total_test   
    
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs


# In[ ]:


# del model
# path = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
# path = os.path.join(exp_path, "final_covpretrain.pt")
# model = MM(output_size)
# model.load_state_dict(torch.load(path))
# model.to(device)
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


# Start training

# In[ ]:


n_epochs = 25
# early stopping patience; how long to wait after last time validation loss improved.
patience = 25
model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)


# Plot the training and validation curves

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
epochs = range(epoc_num+1)
train_acc_list
val_acc_list
train_loss_list
val_loss_list
# plt.plot(epochs, train_acc_list)
# plt.plot(epochs, val_acc_list)
fig1, ax1 = plt.subplots()
ax1.plot(epochs, train_acc_list, label="train acc")
ax1.plot(epochs, val_acc_list, label="val acc")
ax1.set_title("accuracy plot")
ax1.set_xlabel("epochs")
ax1.legend(loc="upper left")
fig2, ax2 = plt.subplots()
ax2.plot(epochs, train_loss_list, label="train loss")
ax2.plot(epochs, val_loss_list, label="val loss")
ax2.set_title("loss plot")
ax2.set_xlabel("epochs")
ax2.legend(loc="upper left")


# Evaluate on test-set

# In[ ]:


outputs = test_model(model)


# In[ ]:


# # # Binary setting
# np_out = np.array(outputs)
# y_pred = np.zeros(np_out.shape)
# y_pred[np_out>0.5]=1
# y_pred = np.array(y_pred)

# # # Binary setting
# test_labels=[]
# # for index, row in test_samples_frame.iterrows():
# for index, row in test_samples_frame.iterrows():
#     lab = row['labels'][0]
#     if lab=="not harmful":
#         test_labels.append(0)    
#     else:
#         test_labels.append(1)


# In[ ]:


# Multiclass setting - Harmful
y_pred=[]
for i in outputs:
#     print(np.argmax(i))
    y_pred.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting
test_labels=[]
for index, row in test_samples_frame.iterrows():
    lab = row['labels'][0]
    if lab=="not harmful":
        test_labels.append(0)
    elif lab=="somewhat harmful":
        test_labels.append(1)
    else:
        test_labels.append(2)


# In[ ]:


def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


# In[ ]:


rec = np.round(recall_score(test_labels, y_pred, average="macro"),4)
prec = np.round(precision_score(test_labels, y_pred, average="macro"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="macro"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1,2]),4)
mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)
print(classification_report(test_labels, y_pred))


# In[ ]:


print("Acc, F1, Rec, Prec, MAE, MMAE")
print(acc, f1, rec, prec, mae, mmae)

