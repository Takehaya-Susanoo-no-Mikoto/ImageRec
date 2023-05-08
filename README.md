# ImageRec
Image recomendation model for artist's blog website

## Search similar images

This code allows you to search for images that are similar to a given image using ResNet50, a popular deep learning model for image recognition. The script uses PCA (Principal Component Analysis) and Nearest Neighbors algorithm to create a compressed index that enables a fast search for nearest neighbors in the space of image features.

## Prerequisites:

  1) Python 3.x >
  
  2) Lib:
    
    - glob
    - pickle
    - os
    - matplotlib
    - numpy
    - torch
    - tqdm
    - PIL
    - scikit-learn
    - torchvision
  
## Usage

1) Clone the repository or copy the code into your local environment.
2) Put your images into the "Data" folder or modify the code accordingly.
3) Run the code in your local environment.
4) Call the "search" function with an image path or index and the number of factors you want to use for PCA decomposition. The output will be a set of similar images to the input image.
  search("ris1.jpg", 500, concl=True)

## Options

The script has several options that you can change:

  - "trained": A boolean value that determines whether to load the pre-trained weights or train the model from scratch. By default, it is set to "True" 
  
  - "emb_filename": The name of the pickle file to save the image embeddings. By default, it is set to "images_weights.pickle"
  
  - "num_images": The number of images to display in the output. By default, it is set to "6"


## Examples  

![image](https://user-images.githubusercontent.com/124432421/236913737-f310dbe5-b5cf-4285-aa8e-fc05d86b274f.png)

![image](https://user-images.githubusercontent.com/124432421/236913815-7e4cc88c-ea20-4190-8ea3-6377616d5841.png)

![image](https://user-images.githubusercontent.com/124432421/236913884-f3f84fd8-98cb-4539-a717-a7ee2d585660.png)

![image](https://user-images.githubusercontent.com/124432421/236914029-5589af18-141f-4e80-9023-4788ecc14fb7.png)
