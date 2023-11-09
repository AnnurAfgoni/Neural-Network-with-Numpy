# NumpyNeuralNetwork: Building Neural Networks from Scratch with NumPy

Numpy Neural Network, where I dive deep into the world of neural networks using NumPy! In this repository, you'll find my Code for creating neural networks from the ground up, without relying on external libraries.

## Neural Network Formula (FeedForward)

$z = \sigma \sum_{i=1}^{n}\left ( \vec{x_i}\vec{w_i} \right ) + b$

* $\sigma$ : Activation Function
* $x_i$ : Input Data
* $w_i$ : Weight
* $b$ : Bias

## Lost / Cost Function (Mean Squared Error)

$MSE = \frac{1}{n}\sum_{i = 1}^{n}(Y_i - \hat{Y_i})^2$

* $Y_i$ : real value
* $\hat{Y_i}$ : predicted value

## Update weight (gradient descent)

$W' = W + \alpha \frac{dL}{dW}$
$$b' = b + \alpha \frac{dL}{db}$$

* $\alpha$ : learning rate
* $L$ : Lost function
