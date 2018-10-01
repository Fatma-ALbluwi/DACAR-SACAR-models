# DACAR-SACAR
Artifacts reduction in JPEG-Compressed Images using CNNs

# DACAR Network/ SACAR Network

In this work, we consider both types of CNN architectures to reduce the artifacts (e.g.; blurring, blocking artifacts and ringing artifacts) in JPEG compressed images. We have proposed two CNNs which we refer to as Direct Architecture Compression Artifacts Removal (DA-CAR) network and Skip Architecture Compression Artifacts Removal (SA-CAR).

* For DA-CAR version, we experiment with 3, 4, and 5 layer architectures (DACAR3, DA-CAR4 and DA-CAR5, respectively).
* For the skip-based architecture we consider 6 layers (SACAR6), the third layer is concatenating activation (feature maps) between the first layer and the second layer (2+1).

# Direct and Skip Architectures.

![cnn](https://user-images.githubusercontent.com/16929158/46287702-4db29a80-c57b-11e8-862b-30543f45b507.jpg)


# Network-Keras

These codes are to reduce the different artifacts from JPEG compressed images.

If these codes are helpful for you, please cite this paper: Artifacts reduction in JPEG-Compressed Images using CNNs,
F. Albluwi, V. Krylov and R. Dahyot Irish Machine Vision and Image Processing conference (IMVIP 2018 <https://www.ulster.ac.uk/conference/imvip-2018>), 2018. Published in IMVIP e-book of proceedings with ISBN 978-0-9934207-3-3. 

## Dependencies

1. TensorFlow.
2. Keras.
3. Matconvnet.  

## Generating data

1. reduce the quality of images at different levels (JPEG_Quality = 10 or 20) by using 'Quality' function. 
2. The training set is 400 images from the BSDS500 (The Berkeley Segmentation Dataset).

## Training

1. generate training patches using Matlab: run generate_train.m and generate_test.m which in train folder, and then put this folder in the network folder you want to train (._train as SA-CAR6_train).
2. use Keras with TensorFlow (tf) as a backend to train any model (DACAR3, DACAR4, DACAR5 or SA-CAR6); Adam is used to optimizing the network for fast convergence: run DACAR_train.py or SACAR_train.py to produce DACAR model / or SACAR model.
3. convert Keras model to .Mat for testing using Matconvnet: run load_save.py first, then run save_model.m to produce Matconvnet model. 
4. run DACAR_SACAR_test.m in “test” folder to test the model; Live1 (which contains 29 images) and BSD100 (which contains 100 images) are used as testing data.

## Some Results

Qualitative evaluation of reconstruction quality using different networks for JPEG quality q = 10.
![q10](https://user-images.githubusercontent.com/16929158/46288025-5f487200-c57c-11e8-8ab5-92da9f873e9a.jpg)

##  Note
This code is based on Keras-2.
