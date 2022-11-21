import cv2
import matplotlib.pyplot as plt

import numpy as np
from torch.nn import functional as F

# from pe import *
from transformer import *

import torch
import os

# from torch.utils.data import DataLoader, ConcatDataset
# import torch.optim as optim
# import torch.nn as nn

import argparse
# from pycocotools import coco
from coco_s import *
from tqdm import tqdm
from myconfig import Config
from tokenizers import Tokenizer
import glob
from PIL import Image
from torchvision import transforms
import json
write_log = False
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    # RandomRotation(),
    # tv.transforms.Lambda(under_max),
    # tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
    #                           0.8, 1.5], saturation=[0.2, 1.5]),
    # tv.transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def fixed_seed(myseed):
    np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)  # , map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")

def get_visual_single(attn, original_image, h=224, w=224): # attn: my_attn[i]
    mask = attn.permute(1, 2, 0).numpy()
    mask = cv2.resize(mask, (h, w))
    origin = cv2.cvtColor(cv2.resize(original_image, (h, w)), cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * mask / mask.max()), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    final = (heatmap.astype(np.float32) * .7 + origin.astype(np.float32) * .3)
    final = final.astype(np.uint8)
    # fig, ax = plt.subplots(num="MRI_demo")
    # ax.imshow(mask, cmap="jet") # 使用imshow以灰度顯示 use imshow to show in greyscale.
    # ax.axis('off')
    # plt.show()
    return final



fixed_seed(1)
config = Config()
# main(config)
if (write_log):
    log_path = './part3_log.txt'
    open(log_path, 'w').close()
img_dir = './hw3_data/p3_data/images/'

if(img_dir[-1] != '/'):
    img_dir += '/'
images_filename = glob.glob(img_dir + '*.jpg')
images_filename.sort()
# print(images_filename[:3])
# images_filename = ['./hw3_data/p3_data/images/bike.jpg', './hw3_data/p3_data/images/girl.jpg', './hw3_data/p3_data/images/sheep.jpg', './hw3_data/p3_data/images/ski.jpg', './hw3_data/p3_data/images/umbrella.jpg']
# images_filename = ['./hw3_data/p2_data/images/val/6320721815.jpg', './hw3_data/p2_data/images/val/000000179758.jpg']#, './hw3_data/p3_data/images/sheep.jpg', './hw3_data/p3_data/images/ski.jpg', './hw3_data/p3_data/images/umbrella.jpg']
# images_filename = ['./hw3_data/p3_data/images/bike.jpg']
tokenizer = Tokenizer.from_file("./hw3_data/caption_tokenizer.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

# captions = [B, word_embedding 128]

model = Transformer(vocab_size=18022, d_model=1024, dec_ff_dim=512,
                    dec_n_layers=6, dec_n_heads=8, max_len=128, dropout=0.1, pad_id=0)

load_parameters(model, './hw3_2_image_capt.pt')

model = model.to(device)
model.eval()

pad_id = 0
bos_id = 2
eos_id = 3

max_len = 64

pb = tqdm(images_filename, leave=False, total=len(images_filename))
pb.unit = "step"

pred_dict = {}

for imgpath in pb:
    my_attn = []
    abnormal_count = 0
    img_name = imgpath.split('/')[-1].split('.')[0]
    # print(img_name, sep='')
    if (write_log):
        with open(log_path, 'a') as logf:
            logf.write(img_name + ', ')
    imgs = tfm(Image.open(imgpath).convert('RGB')).unsqueeze(0)
    # imgs: Tensor  # images [1, 3, 256, 256]
    # cptns_all: Tensor  # all 5 captions [1, lm, cn=5]
    # lens: Tensor  # lengthes of all captions [1, cn=5]
    h, w = 16, 16
    k = 1  # 1= greedy
    # start: [1, 1]
    imgs = imgs.to(device)
    start = torch.full(size=(1, 1),
                        fill_value=bos_id,
                        dtype=torch.long,
                        device=device)
    with torch.no_grad():
        # imgs_enc = image_enc(imgs)  # [1, is, ie]
        logits, attns = model(imgs, start)
        logits: Tensor  # [k=1, 1, vsc]
        attns: Tensor  # [ln, k=1, hn, S=1, is]
        log_prob = F.log_softmax(logits, dim=2)
        log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)
        # log_prob_topk [1, 1, k]
        # indices_topk [1, 1, k]
        current_preds = torch.cat(
            [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)
        # print(current_preds)
        # current_preds: [k, S]

        # get last layer, mean across transformer heads
        # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
        attns = attns[-1, :, :, :, 1:].mean(dim=1).view(k, -1, h, w)
        
        # print(attns[0,0,:].shape)
        my_attn.append(attns.detach().cpu().squeeze(0))
        # print('attns.shape:', attns.shape)
        current_attns = attns.repeat_interleave(repeats=k, dim=0)
        # [k, s=1, h, w]


    seq_preds = []
    seq_log_probs = []
    seq_attns = []
    while current_preds.size(1) <= (
            max_len - 2) and k > 0 and current_preds.nelement():
        with torch.no_grad():
            imgs_expand = imgs.expand(k, *imgs.size()[1:])
            # [k, is, ie]
            # print('imgs_expand.shape:', imgs_expand.shape)
            # print('current_preds:', current_preds)
            logits, attns = model(imgs_expand, current_preds)
            # print('attns.shape:', attns.shape)
            # logits: [k, S, vsc]
            # attns: # [ln, k, hn, S, is]
            # get last layer, mean across transformer heads
            # print('attns[-1].shape', attns[-1].shape)
            attns = attns[-1, :, :, :, 1:].mean(dim=1).view(k, -1, h, w)
            # print('cccccccc attns.shape', attns.shape)
            # # [k, S, h, w]
            attns = attns[:, -1].view(k, 1, h, w)  # current word
            
            # print(attns[0,0,:].shape)
            my_attn.append(attns.detach().cpu().squeeze(0))
            # print('dddddd attns.shape', attns.shape)



            # next word prediction
            log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
            # log_prob: [k, vsc]
            log_prob = log_prob + log_prob_topk.view(k, 1)
            # top k probs in log_prob[k, vsc]
            log_prob_topk, indxs_topk = log_prob.view(-1).topk(k, sorted=True)
            # indxs_topk are a flat indecies, convert them to 2d indecies:
            # i.e the top k in all log_prob: get indecies: K, next_word_id
            prev_seq_k, next_word_id = np.unravel_index(
                indxs_topk.cpu(), log_prob.size())
            next_word_id = torch.as_tensor(next_word_id).to(device).view(
                k, 1)
            # prev_seq_k [k], next_word_id [k]

            current_preds = torch.cat(
                (current_preds[prev_seq_k], next_word_id), dim=1)
            # print(current_preds)
            current_attns = torch.cat(
                (current_attns[prev_seq_k], attns[prev_seq_k]), dim=1)
        if current_preds.size(1) == max_len - 1:
            next_word_id = torch.tensor([[eos_id]]).to(device)
            current_preds[:, -1] = 13
            abnormal_count += 1
            if write_log:
                with open(log_path, 'a') as logf:
                    # logf.write(seq_preds + '\n')
                    logf.write(' [abnormal] ')
        # find predicted sequences that ends
        seqs_end = (next_word_id == eos_id).view(-1)
        if torch.any(seqs_end):
            seq_preds.extend(seq.tolist()
                                for seq in current_preds[seqs_end])
            seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
            # get last layer, mean across transformer heads
            # print('attns.shape:', attns.shape)
            # attns = attns[-1, :, :, :, 1:].mean(dim=1).view(k, -1, h, w)
            # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
            # [k, S, h, w]
            seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

            k -= torch.sum(seqs_end)
            current_preds = current_preds[~seqs_end]
            log_prob_topk = log_prob_topk[~seqs_end]
            current_attns = current_attns[~seqs_end]

    # Sort predicted captions according to seq_log_probs
    specials = [pad_id, bos_id, eos_id]
    # seq_preds, seq_log_probs = zip(*sorted(
    #     zip(seq_preds, seq_log_probs), key=lambda tup: -tup[1]))
    # print('seq_attns.shape = ', seq_attns.shape)
    seq_preds, seq_attns, seq_log_probs = zip(*sorted(
        zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))
    # print(', seq_preds:', seq_preds)
    pred_capt = tokenizer.decode(seq_preds[0])
    if write_log:
        with open(log_path, 'a') as logf:
            # logf.write(seq_preds + '\n')
            logf.write(pred_capt[:-2] + pred_capt[-1] + '\n')
    # print(img_name, ':', pred_capt[:-2] + pred_capt[-1])
    pred_dict[img_name] = pred_capt[:-2] + pred_capt[-1]
    # print('abnormal_count:', abnormal_count)

    # Serializing json
    # json_object = json.dumps(pred_dict, indent=4)
    # if (write_log):
    # with open("part3_pred.json", "a") as outfile:
    #     outfile.write(json_object)

    origin = cv2.imread(imgpath)
    # attn = my_attn[0]

    toks = pred_dict[img_name][:-1].split()
    toks.append('.')
    toks.append('<EOS>')


    plt.figure(figsize=(16, 16))
    plt.subplot(4, 4, 1)
    # plt.imshow(origin)
    plt.imshow(cv2.cvtColor(origin, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title('<BOS>')
    for i, image in enumerate(my_attn):
        # print(i)
        plt.subplot(4, 4, i+2)
        mask = get_visual_single(my_attn[i], origin)
        plt.imshow(mask)
        plt.axis("off")
        plt.title(toks[i])
        # plt.subplot(4, 4, 2 * i + 2)
        # y = np.arange(top_probs.shape[-1])
        # plt.grid()
        # plt.barh(y, top_probs[i])
        # plt.gca().invert_yaxis()
        # plt.gca().set_axisbelow(True)
        # plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
        # plt.xlabel("probability")

    # plt.subplots_adjust(wspace=0.5)
    plt.savefig('./part3_' + img_name + '.png', dpi=150)