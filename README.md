# SelfDrivingObjectDetection

This repository contains the code for self-driving car object detection based off video stream or static images. 

### Data 
The algorithm is trained on the CIFAR10 dataset which contains 60000 32x32 colour images in 10 classes and is able to classify humans, cars, trucks, dogs, bikes, and traffic lights. We use ```load_vehicle_dataset()``` to load the images in both the training set and the test set.

### Model
The first model is a CNN with 128 neurons in the Hidden Layer with activation `relu` and 3 neurons in the Output Layer with activation `softmax`. To compile the model we can run 
`perceptron.compile(loss='categorical_crossentropy', optimizer = optimizers.SGD(lr=1e-3, momentum=0.9), metrics = ['accuracy'])`. The first model achieves around 72% accuracy.

The second model was given an additional image preprocessing step. We implemented a sliding window to crop specific portions of the image to feed into a new 3 layer CNN.
```
cnn = Sequential()
cnn.add(Conv2D(64, (3, 3), input_shape=(32,32,3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation = 'relu'))
cnn.add(Dense(units = 3, activation = 'softmax'))
# compile the network
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=1e-3, momentum=0.95), metrics = ['accuracy'])
```
This achieves an accuracy of 85%.

Our final model is built on the YOLOv3 architecture developed by Joseph Redmon in 2015. This approach first splits the input into a grid of cells using the DarkNet CNN. With DarkNet's bounding box predictions, we apply a threshold to filter the results. 
```
obj_thresh = 0.1
nms_thresh = 0.3
```
Changing this threshold will alter the sensitivity of the CNN. A threshold greater than 0.5 will start to lose objects. A threshold smaller than 0.5 will start drawing too many bounding boxes. We use `boxes = do_nms(boxes, nms_thresh, obj_thresh)` to remove overlapping boxes.

### Output
The final prediction on the image can be called with the `detect_image` function
```
image_pil = Image.open("/content/img1.jpg")
image_w, image_h = image_pil.size
plt.imshow(image_pil)
plt.show()
detect_image(image_pil)
```
The final prediction on a video can be called with the `detect_video` function
```
video_path = '/content/data/video1.mp4'
output_path = '/content/data/video1_detected.mp4'
detect_video(video_path, output_path)
```
