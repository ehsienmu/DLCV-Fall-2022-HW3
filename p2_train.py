from pe import *
from transformer import *

import torch
import os

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torch.nn as nn

import argparse
# from pycocotools import coco
from coco_s import *
from tqdm import tqdm
from myconfig import Config
# from torch.cuda.amp import GradScaler, autocast

# import pdb


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

# class MyCOCODataset(Dataset):
#     def __init__(self, dataset, labels, tfm=unlabel_tfm):
#         self.data = dataset
#         self.labels = labels
#         self.tfm = tfm

#     def __getitem__(self, idx):
#         img = self.data[idx][0]
#         img = transforms.ToPILImage()(img).convert("RGB")
#         return self.tfm(img), self.labels[idx]

#     def __len__(self):
#         return len(self.labels)

# return


# def train_one_epoch(model, criterion, data_loader,
#                     optimizer, device, epoch, max_norm):
#     model.train()
#     criterion.train()

#     epoch_loss = 0.0
#     total = len(data_loader)

#     with tqdm.tqdm(total=total) as pbar:
#         for images, masks, caps, cap_masks in data_loader:
#             samples = utils.NestedTensor(images, masks).to(device)
#             caps = caps.to(device)
#             cap_masks = cap_masks.to(device)

#             outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
#             loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
#             loss_value = loss.item()
#             epoch_loss += loss_value

#             if not math.isfinite(loss_value):
#                 print(f'Loss is {loss_value}, stopping training')
#                 sys.exit(1)

#             optimizer.zero_grad()
#             loss.backward()
#             if max_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#             optimizer.step()

#             pbar.update(1)

#     return epoch_loss / total
# save_path = os.path.join('./)
if __name__ == '__main__':
    fixed_seed(1)
    save_path = './'
    config = Config()
    # main(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # captions = [B, word_embedding 128]

    # model = Transformer(52, 512, 196, 512, 2048, 2, 8, 8, 8, 30, 0.1, 0)
    model = Transformer(vocab_size=18022, d_model=1024, dec_ff_dim=512,
                        dec_n_layers=6, dec_n_heads=8, max_len=128, dropout=0.1, pad_id=0)
    # load_parameters(model, './1118_0_epoch_2.pt')

    for param in model.encoder.parameters():
        param.requires_grad = False
    model = model.to(device)
    # print(model)
    # # CocoCaption()
    dataset_train = build_dataset(config, mode='training')
    dataset_val = build_dataset(config, mode='validation')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    pad_id = 0
    criterion = nn.CrossEntropyLoss(
        reduction='mean', ignore_index=pad_id).to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.lr)
    # scaler = GradScaler()
    for epoch in range(config.start_epoch, config.epochs):

        epoch_loss = 0.0
        total_len = 0
        # train
        with tqdm(data_loader_train, unit="batch") as tepoch:
            for images, _, caps, _ in tepoch:
                # with torch.no_grad():
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                model.train()

                # print(images)
                # print(caps)
                images = images.to(device)
                caps = caps.to(device)
                # with autocast():  # ("cuda", dtype=torch.float16):
                    # output = model(input)
                    # loss = loss_fn(output, target)
                logits, _ = model(images, caps[:, :-1])
                # loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
                loss = criterion(logits.reshape(-1, logits.size()[-1]), caps[:, 1:].reshape(-1))
                # if (torch.isnan(loss)):
                #     print('found nan!')
                #     pdb.set_trace()
                #     continue
                    # loss = criterion(outputs[0].view(-1, outputs[0].shape[-1]), caps[:,1:].contiguous().view(-1))
                epoch_loss += loss.item()
                total_len += 1
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                tepoch.set_postfix(loss=loss.item())
                # break
        torch.save(model.state_dict(), os.path.join(
            save_path, f'epoch_{epoch}.pt'))
        # epoch_loss = 0.0
        # total_len = len(data_loader_train)
        # ### train
        # for batch_idx, (images, _, caps, _,) in enumerate(tqdm(data_loader_train)):
        #     # with torch.no_grad():

        #     optimizer.zero_grad()

        #     model.train()

        #     # print(images)
        #     # print(caps)
        #     images = images.to(device)
        #     caps = caps.to(device)
        #     with autocast():#("cuda", dtype=torch.float16):
        #     # output = model(input)
        #     # loss = loss_fn(output, target)
        #         outputs = model(images, caps[:,:-1])
        #     # loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
        #         loss = criterion(outputs[0].view(-1, outputs[0].shape[-1]), caps[:,1:].contiguous().view(-1))
        #     epoch_loss += loss.item()
        #     # loss.backward()image.png
        #     # optimizer.step()
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()

        print(epoch_loss/total_len)
    # print("Start Training..")
    # for epoch in range(config.start_epoch, config.epochs):
    #     print(f"Epoch: {epoch}")
    #     epoch_loss = train_one_epoch(
    #         model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
    #     lr_scheduler.step()
    #     print(f"Training Loss: {epoch_loss}")

    #     torch.save({
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr_scheduler': lr_scheduler.state_dict(),
    #         'epoch': epoch,
    #     }, config.checkpoint)

    #     validation_loss = evaluate(model, criterion, data_loader_val, device)
    #     print(f"Validation Loss: {validation_loss}")

    #     print()
