import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import time

rho=-0.5
sigma_xx=100
sigma_yy=100
sigma_xy=int(np.floor(rho*sigma_xx*sigma_yy))

levels=5
img = cv2.imread('images/image_r_100.jpg',1)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print img
height, width, channels = img.shape
#print channels

cv2.imshow('image',img)
cv2.waitKey(20)
#cv2.destroyAllWindows()

#plt.imshow(img)

# Create the Laplacian blending object
my_lap_obj=fv(width,height,levels,sigma_xx,sigma_yy,sigma_xy)
#print width
#print height
try:
    while True:
        start = time.time()

        # RANDOM FIXATION POINTS
        center=[int(randint(1, width)), int(randint(1, height))]

        # RANDOM FOVEA SIZE
        sigma_x=randint(1, sigma_xx)
        sigma_y=randint(1, sigma_yy)
	sigma_xy=int(np.floor(rho*sigma_x*sigma_y))
        my_lap_obj.update_fovea(width,height,sigma_x,sigma_y,sigma_xy)

        #print npcv.test_np_mat(np.array(center))
        # Foveate the image
        #print npcv.test_np_mat(np.array(center))
        foveated_img=my_lap_obj.foveate(img,npcv.test_np_mat(np.array(center)))

	#print foveated_img
	#height, width, channels = foveated_img.shape
	#print channels
	#foveated_img = cv2.cvtColor(foveated_img, cv2.COLOR_BGR2RGB)
	#print foveated_img
        end = time.time()
        #print(end - start)

        # Display the foveated image
	cv2.imshow('image',foveated_img.astype(np.uint8))
	cv2.waitKey(100)
        #plt.imshow(foveated_img)
        #img.set_data(im)

        #circle=plt.Circle((center[0],center[1]),1.0,color='blue')
        #ax = plt.gca()
        #ax.add_artist(circle)

        #plt.draw()
        #plt.pause(.001)
        #plt.cla()

except KeyboardInterrupt:
    print('interrupted!')



