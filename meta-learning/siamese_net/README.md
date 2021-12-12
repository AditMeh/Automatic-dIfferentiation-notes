- This is a paper implementation of Siamese Neural Networks for One-shot Image Recognition by Gregory Koch, Richard Zemel and Ruslan Salakhutdinov

- Here is a link to the paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Details: 

### Model:
- As presented in the paper, the siamese convolutional network is structure which consists of two sister CNNs which share the same weights. The final L-2 layer is a dense layer which takes in the flattened output of the last convolutional or pooling layer and outputs a N-dimensional vector, where N is the number of output neurons.

![alt text](https://github.com/AditMeh/Siamese-NN-Implementation/blob/master/images/siamese_architecture.png)

- Given two images, both sister networks output an N-dimensional vector which represents the processed version of the images that were fed in. Then the weighted component wise distance is calculated and passed through a sigmoid, giving a probablity that the two images are of the same class. 

![alt text](https://github.com/AditMeh/Siamese-NN-Implementation/blob/master/images/final_layer_math.png)

### Loss function: 
- The model uses a binary cross entropy loss with regularization, where the positive class (y = 1) is when both images are the same class and the negative class (y = 0) is when the images are from different classes


### Ommniglot dataset:

- Here is a link to the ommniglot dataset, which is used for this implementation: https://github.com/brendenlake/omniglot
