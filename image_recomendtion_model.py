import glob
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from torchvision.models import resnet50, ResNet50_Weights

trained = True
emb_filename = 'images_weights.pickle'
num_images = 6

# Загружаем модель
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()


def pil_loader(path):
    '''Uploading Images'''

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if trained:
    with open(emb_filename, 'rb') as fIn:
        img_names, img_emb_tensors = pickle.load(fIn)
    print("Images:", len(img_names))
else:
    img_names = list(glob.glob('Data/*.jpg'))
    img_emb = []

    for image in tqdm(img_names):
        img_emb.append(
            model(preprocess(pil_loader(image)).unsqueeze(0)).squeeze(0).detach().numpy()
        )
    img_emb_tensors = torch.tensor(img_emb)

    with open(emb_filename, 'wb') as handle:
        pickle.dump([img_names, img_emb_tensors], handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_compressed_index(n_features):
    '''We build a compressed index for a quick search of the nearest neighbors
                        in the space of image features'''

    pca = PCA(n_components=min(n_features, 50))
    pca.fit(img_emb_tensors)
    compressed_features = pca.transform(img_emb_tensors)
    dataset = np.float32(compressed_features)

    index_compressed = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
    index_compressed.fit(dataset)
    return [pca, index_compressed]


def main_image(img_path, desc):
    '''Displays the image located on the specified path
           and adds a name and description to it'''

    plt.imshow(mpimg.imread(img_path))
    plt.xlabel(img_path.split('.')[0], fontsize=12)
    plt.title(desc, fontsize=20)
    plt.show()


def similar_images(indices, suptitle, num_images=6):
    '''Displays several images that are most similar to the specified image
         using the nearest neighbor indexes returned by the index'''

    plt.figure(figsize=(15, 10), facecolor='white')

    plotnumber = 1
    for index in indices[0:num_images]:
        if plotnumber <= num_images:
            ax = plt.subplot(2, 3, plotnumber)
            plt.imshow(mpimg.imread(img_names[index]))
            plt.xlabel(img_names[index], fontsize=12)
            plotnumber += 1
    plt.tight_layout()


def search(query, factors, concl=False):
    '''Searches for the most similar images for a given image or image path'''

    number = []

    if concl:
        if isinstance(query, str):
            img_path = query
        else:
            img_path = img_names[query]

        one_img_emb = torch.tensor(model(preprocess(pil_loader(img_path)).unsqueeze(0)).squeeze(0).detach().numpy())
        main_image(img_path, '')

        compressor, index_compressed = build_compressed_index(factors)
        D, I = index_compressed.kneighbors(
            np.float32(compressor.transform([one_img_emb.detach().numpy()])), n_neighbors=10)
        similar_images(I[0][1:], str(factors))
    else:
        if isinstance(query, str):
            img_path = query
        else:
            img_path = img_names[query]

        one_img_emb = torch.tensor(model(preprocess(pil_loader(img_path)).unsqueeze(0)).squeeze(0).detach().numpy())

        compressor, index_compressed = build_compressed_index(factors)
        D, I = index_compressed.kneighbors(
            np.float32(compressor.transform([one_img_emb.detach().numpy()])), n_neighbors=10)

        for i, index in enumerate(I[0][1:]):
            if i < factors:
                path = img_names[index]
                filename = os.path.splitext(os.path.split(path)[1])[0]
                number.append(filename.split('.')[0])

            else:
                break
        return number


# Calling the search function with a model
search("1.jpg", 500, concl=False)
