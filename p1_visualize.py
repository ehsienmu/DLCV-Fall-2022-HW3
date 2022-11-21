import os
import clip
import torch
# from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import torchvision.transforms as T
import glob
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

def get_all_image_and_labels(data_folder_path, have_label=True):

    if(data_folder_path[-1] != '/'):
        data_folder_path += '/'
    images_filename = glob.glob(data_folder_path+'*.png')
    # images_filename.sort()

    # print(images_filename[:5])
    if have_label:
        labels = []
        for full_path in images_filename:
            lb_str = full_path.split('/')[-1].split('_')[0]
            # labels.append(label2text[lb_str])
            labels.append(int(lb_str))
    return images_filename[:3], labels[:3]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)
# print(preprocess)
# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=False, train=False)

# transform_toPIL = T.ToPILImage()
# # Prepare the inputs
# image, class_id = cifar100[3637]
# image_input = preprocess(transform_toPIL(image))
# image_input = image_input.unsqueeze(0).to(device)

images_data_path = "./hw3_data/p1_data/val/"
json_path = "./hw3_data/p1_data/id2label.json"

with open(json_path) as f:
    all_classes = json.load(f)
# text_descriptions = [f"This is a photo of a {label}" for label in all_classes.values()]
# text_tokens = clip.tokenize(text_descriptions).cuda()


# train_set = CustomImageDataset(data_folder_path=images_data_path, idlabel_path = json_path, have_label=True, transform=preprocess)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)




text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_classes.values()]).to(device)
original_images = []
top_probs = []
top_labels = []
correct_num = 0
imgfnames, lbs = get_all_image_and_labels(images_data_path)
for ifnames, gt in tqdm(zip(imgfnames, lbs), total=len(imgfnames)):
    image = Image.open(ifnames)
    original_images.append(image)
    image_input = preprocess(image).unsqueeze(0).to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    top_probs.append(values.cpu().detach().numpy())
    top_labels.append(indices.cpu().detach().numpy())
    if (gt == indices[0].item()):
        correct_num += 1

    # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{all_classes[str(index.item())]:>16s}: {100 * value.item():.2f}%")

print('acc:', correct_num / len(imgfnames))
top_probs = np.vstack(top_probs)
top_labels = np.vstack(top_labels)

plt.figure(figsize=(32, 20))

for i, image in enumerate(original_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("correct label: " + all_classes[str(lbs[i])])

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, ['a photo of ' + all_classes[str(index)] for index in top_labels[i]])
    plt.xlabel("probability")
    plt.title("correct probability: " + str(top_probs[i, 0]))

plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
plt.savefig('part1.png', dpi = 300)