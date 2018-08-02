from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def process(images,new_images, signs):
    # TODO: fix means and variances of images
    new_images = signs * new_images
    new_images = new_images.reshape((3, 800, 600, 4))
    return (new_images*np.std(images)/np.std(new_images)+np.mean(images))
    

if __name__ == '__main__':

    images = np.array([mpimg.imread(f'gen_img{i}.png').flatten() for i in range(5)])
    # TODO: Fit Fast ICA and get new images
    new_images = None
    ica = FastICA(3)
    new_images = ica.fit_transform(images.T).T
    
    signs = np.array([-1,1,1])[None].T
    new_images = process(images, new_images, signs)
    
    for i, image in enumerate(new_images):
        mpimg.imsave(f"a_{i}.png", image)
        plt.imshow(image.clip(min=0, max=1))
        plt.show()
    