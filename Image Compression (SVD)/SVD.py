#########################################################
# Name: Chirath Abeysinghe
# Topic: Image Compression for RGB and Grayscale Images
# Date: 04/09/2017
#########################################################

###############
# RGB Version #
###############

# Importing necessary packages
import numpy as np 
import matplotlib.pyplot  as plt 
import matplotlib.image as img 
from PIL import Image
from scipy.misc import imsave

# Image Path
path = "einstein.jpg"

# Load the image
x = img.imread(path)

# Applying SVD on each frames (R, G, B)
U1, s1, V1 = np.linalg.svd(x[:,:,0], full_matrices=False)
U2, s2, V2 = np.linalg.svd(x[:,:,1], full_matrices=False)
U3, s3, V3 = np.linalg.svd(x[:,:,2], full_matrices=False)

# Creating new zero matrics for each sigma matrices
S1 = np.zeros((U1.shape[1], s1.shape[0]))
S2 = np.zeros((U2.shape[1], s2.shape[0]))
S3 = np.zeros((U3.shape[1], s3.shape[0]))

# Number of singular values to be removed
t = 50

# Removing singular values from each sigma vectors
s1 = s1[:-t]
s2 = s2[:-t]
s3 = s3[:-t]

# Converting each sigma vectors to Diagonal matrices
s1 = np.diag(s1)
s2 = np.diag(s2)
s3 = np.diag(s3)

# Setting the diagonal values for zero matrices and keep rest as zero
S1[:s1.shape[0], :s1.shape[1]] = s1
S2[:s2.shape[0], :s2.shape[1]] = s2
S3[:s3.shape[0], :s3.shape[1]] = s3

# Reconstructing each frames where number of rows will be removed when taking dot product with Sigma matrix
Y1 = np.dot(U1, np.dot(S1,V1))
Y2 = np.dot(U2, np.dot(S2,V2))
Y3 = np.dot(U3, np.dot(S3,V3))

# Re-stack all the frames to a 3-Dimentional Array
Y = np.dstack((Y1,Y2,Y3))

# Printing the reconstructed image
plt.imshow(Y)
plt.show()

# Printing the dimensions of each components
print("Original Shape: ", x.shape)
print("U : ", U1.shape, "s : ", s1.shape, "V : ", V1.shape)

#####################
# Grayscale Version #
#####################

# Converto a Grayscale image
gray = np.dot(x,[0.299, 0.587, 0.114])

# Apply SVD 
Ug, sg, Vg = np.linalg.svd(gray, full_matrices=False)

# Create a zero matric for sigma with Ug_columns x Vg_rows
Sg = np.zeros((Ug.shape[1], Vg.shape[0]))

# Number of singular values to be removed
p = 520

# Printing new dimensions
print("Sg: ", sg.shape, Sg.shape)

# Removing singular values
sg = sg[:-p]

# Make it a Disgonal matrix
sg = np.diag(sg)

# Setting the values only available in "sg" and rest will be remaining as zeros
Sg[:sg.shape[0], :sg.shape[1]] = sg
print("Original Shape: ", gray.shape)
print("U : ", Ug.shape, "s : ", Sg.shape, "V : ", Vg.shape)

# Reconstructing the image where remaining zeros will remove unwanted rows by multiplication
Z = np.dot(Ug, np.dot(Sg,Vg))

# Compressed image dimension will be same as the original image but REDUCED IN SIZE
print("Compress Image: ", Z.shape)
plt.imshow(Z, cmap=plt.get_cmap('gray'))
plt.show()

# Savign the Compressed version and Grayscale version
imsave("Compressed.jpeg", Z)
imsave("gray.jpeg", gray)
