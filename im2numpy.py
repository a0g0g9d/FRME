import numpy as np
import os
import pandas as pd
import cv2
from cv2 import imread
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

lst1=os.listdir('data/archive/Faces/Faces')
data=pd.read_csv('data/archive/Dataset.csv')
X=list()
Y=list()
def contrast_stretching(img: Image.Image) -> Image.Image:
    """
    Perform contrast stretching on an image.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1.5)  # Increase contrast by a factor of 1.5

def mirror_flip(img: Image.Image) -> Image.Image:
    """
    Perform mirror flip on an image.
    """
    return ImageOps.mirror(img)  # Flip the image horizontally

def blur(img: Image) -> Image:
    """
    Perform blur on an image.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=2))  # Apply Gaussian blur

def sharpening(img: Image.Image) -> Image.Image:
    """
    Perform sharpening on an image.
    """
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(1.5)  # Sharpen by a factor of 1.5

def shearing(img: Image.Image) -> Image.Image:
    """
    Perform shearing on an image.
    """
    # Convert PIL Image to numpy array
    img_array = np.array(img)
    rows, cols = img_array.shape[:2]

    # Define shear matrix
    M = np.float32([[1, 0.5, 0], [0, 1, 0]])  # Shearing transformation matrix

    # Apply shearing transformation
    sheared_img = cv2.warpAffine(img_array, M, (cols, rows))

    # Convert back to PIL Image
    return Image.fromarray(sheared_img)

def Augument(photo: Image.Image):
    """
    Perform multiple augmentations on the input image.
    Returns five augmented images.
    """
    imga = shearing(photo)               # Apply shearing
    imgb = contrast_stretching(photo)    # Apply contrast stretching
    imgc = blur(photo)                   # Apply blur
    imgd = sharpening(photo)             # Apply sharpening
    imge = mirror_flip(photo)            # Apply mirror flip
    return imga,imgb,  imgc, imgd, imge
    
for i in range(len(lst1)):
    fname_img=lst1[i]
    file_to_read='data/archive/Faces/Faces/'+str(fname_img)
    # print(file_to_read)
    img=imread(file_to_read)
    dimension=(160,160)
    resized=cv2.resize(img,dimension)
    X.append(resized)
    resized_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    # resized=Image(resized)
    fname_img_no_ext = fname_img.replace('.jpg', '')
    # Create a boolean mask for matching IDs
    mask = data['id'] == fname_img_no_ext

# Get corresponding labels using the boolean mask
    corresponding_labels = data.loc[mask, 'label'].values  # This will return the matching labels as a NumPy array

# If you want to convert it to a NumPy array
    output_labels = np.array(corresponding_labels)

# Print the output labels
   
    Y.append(output_labels[0])
    new_img1,new_img2,new_img3,new_img4,new_img5=Augument(resized_pil)
    # X.append(np.array(new_img1))
    # X.append(np.array(new_img2))
    # X.append(np.array(new_img3))
    # X.append(np.array(new_img4))
    # X.append(np.array(new_img5))
    # for i in range(1):
    #     Y.append(output_labels[0])
print("x ke phle pookie jay")   
x_n=np.array(X)
print("dayan jay")
y_n=np.array(Y)
print(x_n.shape)
print(y_n.shape)
np.save('initial_x4',x_n)
np.save('initial_y4',y_n)
