import matplotlib.pyplot as plt


from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import imread

img=imread('C:/Users/omid/OneDrive/Desktop/binaee/my program/watershed/omid.jpg')

img = img_as_float(img)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
ax[0].imshow(img)
ax[0].set_title("Original image")
ax[1].imshow(mark_boundaries(img, segments_watershed))
ax[1].set_title('Watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()