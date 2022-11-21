# wget https://www.dropbox.com/s/1iugnjof4fubd9z/clip_ViT_L_14.pt?dl=0 -O hw3_1_clip.pt
python -c "import clip; clip.load('ViT-L/14')"
wget https://www.dropbox.com/s/k58cm37hmdsx3ry/epoch_2.pt?dl=0 -O hw3_2_image_capt.pt
