# Robotics Transformer-1

The RT-1 relies on a data-efficient and compact tokenization of images and language instructions. 

pretrained EFFICIENT-NET

RT-1 tokenizes a history of 6 images by passing images through an ImageNet (300 * 300 * 3 > 9 * 9 * 512)

flatten the output feature map from the EfficienNet into 81(9*9) visual tokens instead of patchifying the images into visual tokens.

**pretrained language embedding** -> natural language instruction to constrain the image tokenizer

(extract task-relevant image features early on and improving performance) UNIVERSAL SENTENCE ENCODER

TokenLeaner is designed to speed up inference by extract more important tokens.

**ACTION TOKENIZATION ** each action dimension is discretized into 256 bins.

