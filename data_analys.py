

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training dataset shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing dataset shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

#visualize
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i in np.arange(0, 10):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].axis('off')
    axes[i].set_title("Label: %s" % y_train[i])

plt.subplots_adjust(hspace=0.5)
plt.show()

#We can also calculate some basic statistics for the dataset, like the distribution of the labels:
unique_labels, counts = np.unique(y_train, return_counts=True)
label_stats = pd.DataFrame({"Label": unique_labels, "Count": counts})
print(label_stats)

#calculate the mean and standard deviation of the pixel values for each class
mean_pixel_values = []
std_pixel_values = []

for label in unique_labels:
    images_of_label = x_train[y_train == label]
    mean_pixel_values.append(np.mean(images_of_label))
    std_pixel_values.append(np.std(images_of_label))

label_stats["Mean Pixel Value"] = mean_pixel_values
label_stats["Std. Pixel Value"] = std_pixel_values
print(label_stats)

#Distribution of pixel values for each class:

import seaborn as sns

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for label, ax in zip(unique_labels, axes):
    images_of_label = x_train[y_train == label]
    pixel_values = images_of_label.ravel()
    sns.histplot(pixel_values, ax=ax, kde=True, bins=50)
    ax.set_title(f"Label: {label}")

plt.tight_layout()
plt.show()

#Plot the average image for each class:
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for label, ax in zip(unique_labels, axes):
    images_of_label = x_train[y_train == label]
    mean_image = np.mean(images_of_label, axis=0)
    ax.imshow(mean_image, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Average Image: {label}")

plt.tight_layout()
plt.show()

#Compute the pairwise correlation of the average images:
avg_images = []

for label in unique_labels:
    images_of_label = x_train[y_train == label]
    avg_images.append(np.mean(images_of_label, axis=0).ravel())

correlation_matrix = np.corrcoef(avg_images)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, xticklabels=unique_labels, yticklabels=unique_labels)
ax.set_title("Pairwise Correlation of Average Images")
plt.show()

#These additional analyses provide a deeper understanding of the MNIST dataset. The distribution of pixel values helps us understand the intensity patterns within each class. The average images provide a sense of the common features within each class. Lastly, the pairwise correlation of the average images can reveal similarities or differences between the classes.
