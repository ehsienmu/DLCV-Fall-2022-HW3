# from pathlib import Path
# from collections import defaultdict
# from statistics import mean, pstdev
# from tqdm import tqdm
# import pdb
import numpy as np
# import pandas as pd
# import torch
# from torch import Tensor
from torch.nn import functional as F
# from torchtext.vocab import Vocab
# from torchvision.transforms import Normalize, Compose
# from torch.utils.data import DataLoader

# from dataset.dataloader import HDF5Dataset, collate_padd
# from models.cnn_encoder import ImageEncoder
# from models.IC_encoder_decoder.transformer import Transformer
# from nlg_metrics import Metrics
# from utils.train_utils import seed_everything, load_json
# from utils.test_utils import parse_arguments
# from utils.gpu_cuda_helper import select_device

# from pe import *
from transformer import *

import torch
import os

# from torch.utils.data import DataLoader, ConcatDataset
# import torch.optim as optim
# import torch.nn as nn
import os
import argparse
# from pycocotools import coco
# from coco_s import *
from tqdm import tqdm
# from myconfig import Config
from tokenizers import Tokenizer
# import glob
from PIL import Image
from torchvision import transforms
import json
# args.write_log = True
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


if __name__ == '__main__':
    fixed_seed(1)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--model_file', default='', type=str)
    parser.add_argument('--write_log', default=False, type=bool)
    parser.add_argument('--log_path', default='./inferece_log.txt', type=str)
    parser.add_argument('--k_value', default=1, type=int)

    
    args = parser.parse_args()
    # config = Config()
    # main(config)
    if (args.write_log):
        # args.log_path = #'./inferece_log_epoch_1_k_3.txt'
        open(args.log_path, 'w').close()
    img_dir = args.input_dir#'./hw3_data/p2_data/images/val/'

    if(img_dir[-1] != '/'):
        img_dir += '/'
    # images_filename = glob.glob(img_dir + '*.jpg')
    # images_filename.sort()
    images_filename = os.listdir(img_dir)
    # images_filename = images_filename[765:]
    # print(images_filename[765:])
    # images_filename = ['./hw3_data/p2_data/images/val/000000333704.jpg']
    tokenizer = Tokenizer.from_file("./caption_tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # captions = [B, word_embedding 128]


    model = Transformer(vocab_size=18022, d_model=1024, dec_ff_dim=512,
                        dec_n_layers=6, dec_n_heads=8, max_len=128, dropout=0.1, pad_id=0)
    
    load_parameters(model, args.model_file)

    model = model.to(device)
    model.eval()

    pad_id = 0
    bos_id = 2
    eos_id = 3
    
    max_len = 64

    pb = tqdm(images_filename, leave=False, total=len(images_filename))
    pb.unit = "step"

    pred_dict = {}
    abnormal_count = 0
    for img_name in pb:
        
        if (args.write_log):
            with open(args.log_path, 'a') as logf:
                logf.write(img_name + ', ')
        imgs = tfm(Image.open(os.path.join(img_dir, img_name)).convert('RGB')).unsqueeze(0)
        

        k = args.k_value  # 1= greedy
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
            # current_preds: [k, S]

            # get last layer, mean across transformer heads
            # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
            # current_attns = attns.repeat_interleave(repeats=k, dim=0)
            # [k, s=1, h, w]

        seq_preds = []
        seq_log_probs = []
        seq_attns = []
        while current_preds.size(1) <= (
                max_len - 2) and k > 0 and current_preds.nelement():
            with torch.no_grad():
                imgs_expand = imgs.expand(k, *imgs.size()[1:])
                # [k, is, ie]
                logits, _ = model(imgs_expand, current_preds)
                # logits: [k, S, vsc]
                # attns: # [ln, k, hn, S, is]
                # get last layer, mean across transformer heads
                # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # # [k, S, h, w]
                # attns = attns[:, -1].view(k, 1, h, w)  # current word

                # next word prediction
                log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                # log_prob: [k, vsc]
                log_prob = log_prob + log_prob_topk.view(k, 1)
                # top k probs in log_prob[k, vsc]
                log_prob_topk, indxs_topk = log_prob.view(-1).topk(k,
                                                                   sorted=True)
                # indxs_topk are a flat indecies, convert them to 2d indecies:
                # i.e the top k in all log_prob: get indecies: K, next_word_id
                prev_seq_k, next_word_id = np.unravel_index(
                    indxs_topk.cpu(), log_prob.size())
                next_word_id = torch.as_tensor(next_word_id).to(device).view(
                    k, 1)
                # prev_seq_k [k], next_word_id [k]

                current_preds = torch.cat(
                    (current_preds[prev_seq_k], next_word_id), dim=1)
                # current_attns = torch.cat(
                #     (current_attns[prev_seq_k], attns[prev_seq_k]), dim=1)
                # print(next_word_id.shape)
                # print('current_preds.shape:', current_preds.shape)
            if current_preds.size(1) == max_len - 1:
                # next_word_id = torch.tensor([[eos_id]]).to(device)
                next_word_id = torch.full(seqs_end.size(), eos_id).to(device)
                current_preds[:, -1] = 13
                abnormal_count += 1
                if args.write_log:
                    with open(args.log_path, 'a') as logf:
                        # logf.write(seq_preds + '\n')
                        logf.write(' [abnormal] ')
            # find predicted sequences that ends
            seqs_end = (next_word_id == eos_id).view(-1)
            # print(seqs_end)
            if torch.any(seqs_end):
                # pdb.set_trace()
                # print('in if else', seqs_end)
                # pdb.set_trace()
                seq_preds.extend(seq.tolist() for seq in current_preds[seqs_end])
                # pdb.set_trace()
                seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                # get last layer, mean across transformer heads
                # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # [k, S, h, w]
                # seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

                k -= torch.sum(seqs_end)
                current_preds = current_preds[~seqs_end]
                log_prob_topk = log_prob_topk[~seqs_end]
                # current_attns = current_attns[~seqs_end]

        # Sort predicted captions according to seq_log_probs
        specials = [pad_id, bos_id, eos_id]
        seq_preds, seq_log_probs = zip(*sorted(
            zip(seq_preds, seq_log_probs), key=lambda tup: -tup[1]))
        # print(', seq_preds:', seq_preds)
        pred_capt = tokenizer.decode(seq_preds[0])
        if args.write_log:
            with open(args.log_path, 'a') as logf:
                # logf.write(seq_preds + '\n')
                logf.write(pred_capt[:-2] + pred_capt[-1] + '\n')
        # print(img_name, ':', pred_capt[:-2] + pred_capt[-1])
        pred_dict[img_name] = pred_capt[:-2] + pred_capt[-1]

    # Serializing json
    json_object = json.dumps(pred_dict, indent=4)
    print('abnormal_count:', abnormal_count)
    # if (args.write_log):
    with open(args.output_file, "w") as outfile:
        outfile.write(json_object)