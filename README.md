# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Project Description

### 1. Submission Files

My project includes the following files:

| Files         		|     Description	        							|
|:---------------------:|:-----------------------------------------------------:|
|model.py 				|containing the script to create and train the modules 	|
|drive.py 				|for driving the car in autonomous mode 				|
|model.h5 				|containing a trained convolution neural network		|
|README.md 		        |summarizing the results								|
|carnd-p3-env.yml 		|conda environment 										|
|run1_track1_cw.mp4 	|video on track1 clockwise, 30mph 						|
|run2_track1_ccw.mp4 	|video on track1 counterclockwise, 30mph 				|
|run3_track2_cw.mp4 	|video on track2 clockwise, 25mph 						|
|run4_track2_ccw.mp4 	|video on track2 counterclockwise, 25mph 				|

### 2. How To Run

#### 2.1. Install Environment

Here is my environment, it's a little different with Udacity Official Environment, I updated many packages. 

```sh
conda env create -f carnd-p3-env.yml
```

#### 2.2. Run Model

**1**. Open Simulator, select Track and Autonomous Mode.
**2**. Activate Environment on the command line.
```sh
source activate carnd-term1
```
**3**. Execute following script, waiting the model to load, it will take half a minute.
```sh
python drive.py model.h5
```

### 3. Data

#### 3.1. Data Collection

I run a lap clockwise and counterclockwise respectively on both two tracks. Then I trained the model, everywhere I fell off, I capture 2 or 3 times by recovering back to center on that section.

#### 3.2. Data Selection and Augment

I didn't make more image processing except normalization and cropping, because the track is very clean, no shadow. 
I used Python generator and following steps and technique to generate images.

* Select left, center and right camera randomly.
* For left camera images, steer angle +0.18. [1]
* For right camera images, steer angle -0.18.
* Flip all images.
* Normalization (x / 255.0) - 0.5.
* Cropping.

Note [1]: Here is a little confused according to course [description](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/2cd424ad-a661-4754-8421-aec8cb018005). As my understanding, turn left, the left camera angle absolute should be less than the center one and the right camera angle absolute should be large than the center one, and vice versa. We can notice the steer angle in csv file, the angle is negative during turning left, so for left camera angle, abs(center_angle + 0.18) < abs(center_angle). The steer angle is positive when turning right, so for left camera angle, abs(center_angle + 0.18) > abs(center_angle).

```python
def augment(x, y):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(x, y):
        augmented_images.append(np.fliplr(image))
        augmented_measurements.append(-measurement)

    return augmented_images, augmented_measurements

def generator(samples, file_path ='./data/IMG/', is_augment = True, batch=32):
    n_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        batch_size = int(batch / AUG_MULTIPLY) if is_augment else batch
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if is_augment:
                    camera = np.random.randint(3)
                    try:
                        image = plt.imread(file_path + batch_sample[camera].split(delimiter)[-1])
                    except PermissionError:
                        print(batch_sample[camera].split(delimiter))
                        continue
                    images.append(image)
                    center_angle = float(batch_sample[3])
                    correction = 0.18
                    if camera == 0:
                        angle = center_angle
                    elif camera == 1:
                        angle = center_angle + correction
                    else:
                        angle = center_angle - correction
                    angles.append(angle)
                else:
                    name = file_path + batch_sample[0].split(delimiter)[-1]
                    center_image = plt.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

            if is_augment:
                aug_images, aug_angles = augment(images, angles)
                X_train = np.concatenate((np.array(images), np.array(aug_images)), axis=0)
                y_train = np.concatenate((np.array(angles), np.array(aug_angles)), axis=0)
            else:
                X_train = np.array(images)
                y_train = np.array(angles)

            yield shuffle(X_train, y_train)
```

#### 3.3. Training and Validation Data

I split the samples to two parts, 20% for validation. I only use center camera images for validation.

### 4. Model Architecture and Training Strategy

#### 4.1. Model Architecture

At first, I use [NVIDA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model, it's very good, but I thought it's too heavy for this project, so I change two convolutional layers to two maxpool layers. At the same time, I add three dropout layers to reduce overfitting.

| Layer         		|     Description	        					| 		 Output Shape	|
|:---------------------:|:---------------------------------------------:|:---------------------:| 
|Lambda					|	 (None, 160, 320, 3) 						|		0				|
|Cropping2D				|	 (None, 65, 320, 3) 						|		0				|
|Conv2D					|	 (None, 31, 158, 24)						|		1824      		|
|MaxPooling2D			|	 (None, 15, 79, 24)							|		0				|
|Dropout				|	 (None, 15, 79, 24)							|		0				|
|Conv2D					|	 (None, 6, 38, 48)							|		28848     		|
|MaxPooling2D			|	 (None, 3, 19, 48)							|		0				|
|Conv2D					|	 (None, 1, 17, 64)							|		27712			|
|Dropout				|	 (None, 1, 17, 64)							|		0				|
|Flatten				|	 (None, 1088) 								|		0				|
|Dense					|	 (None, 100)								|		108900    		|
|Dropout				|	 (None, 100)								|		0				|
|Dense					|	 (None, 50) 								|		5050      		|
|Dense					|	 (None, 10)									|		510       		|
|Dense					|	 (None, 1)									|		11				|
	
```python
def net():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, strides=(2, 2), padding='valid', activation='elu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, 5, strides=(2, 2), padding='valid', activation='elu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, 3, strides=(1, 1), padding='valid', activation='elu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    return model
```


#### 4.2. Model Training

* The model used an adam optimizer with learning rate 1e-4, because default adam learning rate is high for this project.
* I use early stopping with patience 2
* Every training, I load last saved weights, so that I only need 1 or 2 epoches for new data and save time.

### 5. Result

I believe this project is mainly to practice how to collect a good data, this is very important for deeplearning study. I also understand why use 3 cameras on the self driving car. At last, I get a result which 30MPH on Track1 and 25MPH on Track2. Full video both 25MPH is [here](https://youtu.be/73obO5EgSkk).