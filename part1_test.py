import clip
import torch
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
import glob
import json
from PIL import Image
from tqdm import tqdm
import argparse

def get_all_image_and_labels(data_folder_path, have_label=True):
    if(data_folder_path[-1] != '/'):
        data_folder_path += '/'
    images_filename = glob.glob(data_folder_path+'*.png')
    images_filename.sort()

    # print(images_filename[:5])
    if have_label:
        labels = []
        for full_path in images_filename:
            lb_str = full_path.split('/')[-1].split('_')[0]
            labels.append(int(lb_str))
        return images_filename, labels
    else:
        return images_filename

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_images_dir', default='', type=str)
    parser.add_argument('--input_json_file', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)

    args = parser.parse_args()

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

    images_data_path = args.input_images_dir # "./hw3_data/p1_data/val/"
    json_path = args.input_json_file # "./hw3_data/p1_data/id2label.json"

    with open(json_path) as f:
        all_classes = json.load(f)
    # text_descriptions = [f"This is a photo of a {label}" for label in all_classes.values()]
    # text_tokens = clip.tokenize(text_descriptions).cuda()


    # train_set = CustomImageDataset(data_folder_path=images_data_path, idlabel_path = json_path, have_label=True, transform=preprocess)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)




    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_classes.values()]).to(device)

    results = []

    imgfnames = get_all_image_and_labels(images_data_path, have_label=False)
    for ifnames in tqdm(imgfnames):
        image = Image.open(ifnames)
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
        
        results.append((ifnames.split('/')[-1], str(indices[0].item())))
        # Print the result
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{all_classes[str(index.item())]:>16s}: {100 * value.item():.2f}%")

    # print('acc:', correct_num / len(imgfnames))

    with open(args.output_file, 'w') as f:
        f.write('filename,label\n')
        for fname, predl in results:
            f.write(fname)
            f.write(',')
            f.write(predl)
            f.write('\n')
        
    