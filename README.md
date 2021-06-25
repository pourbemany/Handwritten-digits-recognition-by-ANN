# Handwritten-digits-recognition-by-ANN-SLP
This project aims to use an artificial neural network (ANN) to train a simple
system (*single-layer perceptron (SLP)*) for the recognition of handwritten digits (0,
1, …, 9).

We design a fully connected network structure of 784 input nodes and ten output
nodes.  
The input to the single-layer network architecture will be a set of binary
pixels representing a  
28×28 image of handwritten digits. The output should indicate which digits
(0,....,9) are in the input image.

This project uses the MNIST database of handwritten digits for testing and
training the system attached here as "mnist_handwritten.csv." It contains 60k
images that are sorted as one vector, each with its labels.

To begin with, we select a subset of the MNIST database consisting of around 500
images of handwritten digits (0,...,9) for training the system and use another
100 images for testing the system. Next, we create binary or bipolar images of
handwritten digits from grayscale images available in MNIST by simple
thresholding.

To have an overview of our data, first, I plotted the distribution of labels
(digits). As you can see in the figure below, overall, it is okay.

![Chart, bar chart Description automatically
generated](media/159acf51765a28824250566e0e5238a0.png)

I used the threshold value 150 to convert the gray images to binary images.
After selecting 500 images as the training set and 100 images as the testing
set, by considering beta 0.01, epsilon 0.001, and maximum iteration 3000, I
followed the steps below to learn the network:

1.  Initializing the weights by np.ones((10,784)) \* 0.01 (we have 784 input and
    10 output for each training row)

2.  Computing v by dot product of w and x np.dot(w,np.transpose(x)) + w0

3.  Applying the activation function and computing y
    activation_function(v,kernel\_type)

4.  Calculating the erroe e e = np.transpose(d)-y

5.  Computing ∆w dw = np.dot(beta\*e,x)

6.  Modifing w w = w + dw

7.  Repeating step 2 to 6 until e\<0.01 or iteration\>3000

Then I used the calculated w and the test set to test the model. As the figure
below shows, the accuracy is 65%, the mean square error (MSE), and the
percentage error (PE) are 0.054 and 0.36, respectively.

A confusion matrix and classification report are provided for all the scenarios
to understand the results better.

![](media/71539fadef3d10878c1f0a4c9e52df97.png)

Then we plot a learning curve that illustrates the mean square error versus
iterations.

| ![Chart Description automatically generated](media/5c4edda4ae9513b94924a29b1bab6cb3.jpg) |
|------------------------------------------------------------------------------------------|

The result shows that the mean square error (MSE) decreases significantly at the
first iteration. It decreases more by having more iterations.  
We can also plot the percentage error in testing the handwritten digit
recognition system as a bar chart. It shows the mean error that occurred while
testing each digit with the test data.

| ![Chart, bar chart Description automatically generated](media/4a2ce64787108f52ebb01c60f470d15e.jpg) | ![Chart, bar chart Description automatically generated](media/ac53b092221b8b82afab9468af58a409.jpg) |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|

We see a significant error at digits 5 and 8. If we depict the distribution of
digits in the training data, we can find biased data on 5 and 8 (see the figure
below). So, it makes sense to have more errors on 5 and 8.

![Chart, bar chart Description automatically
generated](media/7e27c51a36e3a9017285649d31921eb1.jpg)

Then we repeat this experiment for different learning rate parameters. We start
with a large value and gradually decrease to a small value. I considered 5
values {0.01, 0.2, 0.5, 0.7, 0.9} for the learning rate. There is no significant
change in the results, but by decreasing beta, the MSE and PE decrease, and the
training time increases (see the results below). Indeed, the learning rate can
control the model's speed. Lower learning rates need more training epochs. Too
large learning rates lead the model to converge too quickly to a non-optimal
weight. Also, the too-small ones lead to a very slow model.

Training size 500 – Beta 0.9

![](media/77a50779d4b4bedf986e03700f314731.png)

Training size 500 – Beta 0.7

![](media/3da7b98167e5c66fd816347f783019e5.png)

Training size 500 – Beta 0.5

![](media/7f0b9b7db89f92cc6aba2f9781f64a96.png)

Training size 500 – Beta 0.2

![](media/f24ec78d5bd85590d1e5603b7b69f800.png)

Training size 500 – Beta 0.1

![](media/71539fadef3d10878c1f0a4c9e52df97.png)

![](media/c2dd4cb7bf8afba9b2a78e925cdd1c0c.jpeg)

| Threshold – Training size 500 – Various learning rate      |                                                                |
|------------------------------------------------------------|----------------------------------------------------------------|
| ![](media/f4a7cc90bb50fda89f9a58c8b7b857ac.jpg)            | ![](media/af4e1c4fd50de681085e86bde2e59f0b.jpg)                |
| Iteration \#: 67 Traning time: 0.18236017227172852 seconds | Mean Square Error: 0.057 Percentage Error: 0.37                |
| ![](media/6b3c91b1bbe23945b209598822ea94a6.jpg)            | ![](media/5f4775b9ffe31e27f33c630f20bff400.jpg)                |
| Iteration \#: 67 Traning time: 0.20390677452087402 seconds | Mean Square Error: 0.057 Percentage Error: 0.37                |
| ![](media/bbdc3de51d5794669a926e31c8a624ab.jpg)            | ![](media/03e545997680b7b494abfb08f127e874.jpg)                |
| Iteration \#: 67 Traning time: 0.18638300895690918 seconds | Mean Square Error: 0.057 Percentage Error: 0.37                |
| ![](media/e686d0b8c5eebaaed9d4daf4ef5c5b14.jpg)            | ![](media/6f2a3f50439bc07b7bab3ca49263e39e.jpg)                |
| Iteration \#: 66 Traning time: 0.16989684104919434 seconds | Mean Square Error: 0.05600000000000001 Percentage Error: 0.36  |
| ![](media/4a2ce64787108f52ebb01c60f470d15e.jpg)            | ![](media/ac53b092221b8b82afab9468af58a409.jpg)                |
| Iteration \#: 67 Traning time: 0.17978572845458984 seconds | Mean Square Error: 0.054000000000000006 Percentage Error: 0.36 |

| ![](media/869e1ac615d1e4af44a763dbe2e230fd.jpg) | ![](media/0e79cb5e4147e993f4431fd174777478.jpg) |
|-------------------------------------------------|-------------------------------------------------|
| ![](media/f7bb90dc3f2995879bac18fc0fb9a860.jpg) | ![](media/ebd600dd5354ba7ebdab3e77275260f9.jpg) |
| ![](media/5c4edda4ae9513b94924a29b1bab6cb3.jpg) |                                                 |

Also, there is no significant change in the iteration number.

Let's repeat the experiment with a more extensive database; the first 10000
images for training (image indexes from 0-10000) and test with another 1000
images (image indexes from 20000-21000).

By increasing the training size, the accuracy and the training time increase.
Changing the beta, we can have minor improvement on the accuracy. The training
time is around 135 seconds, and the accuracy is at most 73%. Compared with the
previous training set, this training set needs much more iterations to reach an
error less than epsilon.

Training size 10000 – Beta 0.9

*![](media/f5a97632addb6be0947b28f53577c45e.png)*

Training size 10000 – Beta 0.7

*![](media/775fb720631ae6cb024061a97cdc3cf4.png)*

*  
*Training size 10000 – Beta 0.5

*![](media/a5d42f8ff7f957fff4345d7f20224955.png)*

Training size 10000 – Beta 0.2

*![](media/eb1bdba23e720bddf4fa5f297b549a6c.png)*

*  
*

Training size 10000 – Beta 0.1

*![](media/1c1e3ee0f5768160bd76e73eea836fab.png)*

![](media/96d8c9a2833da126569f8dcb83c2c4d4.jpeg)

Comparing to the SVM, ANN is less accurate on the selected training set. Also,
the processing time of ANN is much more than SVM, 134 versus 5 seconds.

| Threshold – Training size 10000 – Various learning rate     |                                                                |
|-------------------------------------------------------------|----------------------------------------------------------------|
| ![](media/e9aad0dc899657aea5ec9a5796f90af7.jpg)             | ![](media/1eae6a0f96c20d6911073c476ce8287c.jpg)                |
| Iteration \#: 3001 Traning time: 137.32243943214417 seconds | Mean Square Error: 0.0446 Percentage Error: 0.28               |
| ![](media/68e94faebd8310b1b1717f80eec5efb2.jpg)             | ![](media/6a271af5c8e9ec2be85d008a8c8c4ca9.jpg)                |
| Iteration \#: 3001 Traning time: 134.11536717414856 seconds | Mean Square Error: 0.0493 Percentage Error: 0.287              |
| ![](media/7d19ef09bf1df437db799d8d078f0b02.jpg)             | ![](media/8aa1bf7c6a92ef4c9e7fe2eac9a137fb.jpg)                |
| Iteration \#: 3001 Traning time: 133.62726068496704 seconds | Mean Square Error: 0.0528 Percentage Error: 0.316              |
| ![](media/94915665c2b08f4fe2336a2b805943d8.jpg)             | ![](media/f4928c3e11664881a58ad612654a77df.jpg)                |
| Iteration \#: 3001 Traning time: 134.44923448562622 seconds | Mean Square Error: 0.05090000000000001 Percentage Error: 0.309 |
| ![](media/4f3d68bd76478bdc7025170292988b8b.jpg)             | ![](media/556cdeb05c8e6604a6c0af0443457228.jpg)                |
| Iteration \#: 3001 Traning time: 134.25897192955017 seconds | Mean Square Error: 0.045 Percentage Error: 0.28                |

| ![](media/f0cc32cbce59c8400155f82050366870.jpg) | ![](media/ec20fd604af93f9a845522ca0500b5b2.jpg) |
|-------------------------------------------------|-------------------------------------------------|
| ![](media/33897bfe6a0f1c224b4899cd62f8fddd.jpg) | ![](media/cf30578cfca5d67c46ff23d5d238d4c6.jpg) |
| ![](media/9c51b1ae8789f41aceb8bd3df89b8fc5.jpg) |                                                 |

*  
*

What will happen if we repeat the experiment with multilevel data while
normalizing the input data and using the sigmoid function for output
thresholding (without thresholding the input data)?

This model is more accurate, but the training time is half of the previous
experience. In addition, sigmoid increases convergence speed, so the network can
more quickly reach smaller errors.

Sigmoid – Beta 0.9

![](media/01cf502889f3e28ecb97e8f6310ac518.png)

Sigmoid – Beta 0.7

![](media/5c94b5930807a5c85bf014a037d134d1.png)

Sigmoid – Beta 0.5

![](media/83d4cf7e4496fe62b8d6ae4a46ebe501.png)

Sigmoid – Beta 0.2

![](media/ee5c1ca15659e1392eb8620804655bb9.png)

Sigmoid – Beta 0.1

![](media/286140bf28311f5b7d871cbcf04b0aae.png)

![](media/d3d26cfdde7c7ab397c15ebca7af971e.jpeg)

|  Sigmoid – training size 10000 – Learning rate 0.9          |                                                                 |
|-------------------------------------------------------------|-----------------------------------------------------------------|
| ![](media/726a225a4414a8612e09f51823120439.jpg)             | ![](media/f61de204f957e2b39c97c4b4ba2d97df.jpg)                 |
| Iteration \#: 3001 Traning time: 40.01509189605713 seconds  | Mean Square Error: 0.032600000000000004 Percentage Error: 0.219 |
| Sigmoid – training size 10000 – Learning rate 0.7           |                                                                 |
| ![](media/ccffb1d82affc01c805eaa26a0b1e3f4.jpg)             | ![](media/8641ffe5db93a9bd92de027388c98b69.jpg)                 |
| Iteration \#: 3001 Traning time: 41.2118194103241 seconds   | Mean Square Error: 0.0417 Percentage Error: 0.252               |
| Sigmoid – training size 10000 – Learning rate 0.5           |                                                                 |
| ![](media/e83ce957a62e618b997e372e4f6bce17.jpg)             | ![](media/a0313753c9dd7374d6715022c3931fd5.jpg)                 |
| Iteration \#: 3001 Traning time: 42.169373512268066 seconds | Mean Square Error: 0.04 Percentage Error: 0.253                 |
| Sigmoid – training size 10000 – Learning rate 0.2           |                                                                 |
| ![](media/43a9615a3c5307556ae37161022b6ed6.jpg)             | ![](media/3e9e6127c106c57643d408962c9fedfd.jpg)                 |
| Iteration \#: 3001 Traning time: 42.01747918128967 seconds  | Mean Square Error: 0.0378 Percentage Error: 0.234               |
| Sigmoid – training size 10000 – Learning rate 0.1           |                                                                 |
| ![](media/abd0b7782e947bc548509c7c8605703b.jpg)             | ![](media/afe15bcf0003e83b7cf1ee6ea9ea2a9c.jpg)                 |
| Iteration \#: 3001 Traning time: 42.08747053146362 seconds  | Mean Square Error: 0.0407 Percentage Error: 0.26                |

| ![](media/f0cc32cbce59c8400155f82050366870.jpg) | ![](media/ec20fd604af93f9a845522ca0500b5b2.jpg) |
|-------------------------------------------------|-------------------------------------------------|
| ![](media/33897bfe6a0f1c224b4899cd62f8fddd.jpg) | ![](media/cf30578cfca5d67c46ff23d5d238d4c6.jpg) |
| ![](media/9c51b1ae8789f41aceb8bd3df89b8fc5.jpg) |                                                 |

Considering the same training and testing dataset, SVM has a better performance
in comparison with ANN. I think the main reason is that we used a single-layer
ANN network, so it can be improved by adding some hidden layers. On the other
hand, decreasing the epsilon and increasing the iteration number can lead to a
more accurate model. Here, the SVM model is much faster and more accurate than
ANN.

SVM – Training size 10000

**![](media/a0b3e1c2499700d4869fc6d333e1c5ba.png)**

ANN – Training size 10000

**![](media/b97d0eefe8df9378036de1e38b6c3bfd.png)**

![](media/bfa597bd5c1929b346707aaa6c30e470.jpeg)

To visualize the model's result, I randomly select six images and their
predicted labels. As you can see in the figure below, the model's performance is
as we expected.

| ![](media/689c90a275b8acd9c23c58532332a07c.jpg)                                      | ![](media/369d82d36d7d672109f1fb41f776aa84.jpg) | ![](media/415c95103c7ef97207cb68605a29b504.jpg) |
|--------------------------------------------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| ![](media/13e404bdf07fbe0a910571e98b39162d.jpg)                                      | ![](media/3d9e664650403e597a300da36258cb21.jpg) | ![](media/277b2bf9c1042579689852e4511fc726.jpg) |
| *Comparing the predicted label by ANN (sigmoid – multilevel) with the actual label*  |                                                 |                                                 |

In conclusion, the size of the training set has a significant effect on the
model's performance. The learning rate can control convergence speed, and the
sigmoid has a better performance than the simple threshold activation function.
Overall, single layer ANN has a weaker performance as compared to the SVM.

**References:**

<https://www.kaggle.com/mirichoi0218/ann-slp-making-model-for-multi-classification>

<https://www.kaggle.com/shivamb/a-very-comprehensive-tutorial-nn-cnn>
