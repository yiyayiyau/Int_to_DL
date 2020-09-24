# introtodeeplearning
Notes of learning DL: [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)
[Quelle: pdf](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf)
[Quelle: video](https://www.youtube.com/watch?v=njKP3FqW3Sk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)
## L1
### Common Activation Functions
* Sigmoid Function
* Hyperbolic Tangent
* Rectified Linear Unit(ReLU)
```python
tf.math.sigmoid(z)
tf.math.tanh(z)
tf.nn.relu(z)
```
The purpose of **activation functions** is to introduce **non-linearities** into the network. Linear activation functions produce linear decisions no matter the network size. Non-lineaities allow us to approximate arbitrarily complex functions.
## Building Neural Networks with Perceptrons
 Inputs Weights Sum Non-Linearity Output
Because all inputs are **densely connected** to all outputs, these layers are called **Dense layers**
### Dense layer from scratch

```python
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, input_dim, output_dim):
		super(MyDenseLayer, self).__init__()
		# Initialize weights and bias
		self.W = self.add_weight([input_dim, output_dim])
		self.b = self.add_weight([1, output_dim])
	def call(self, inputs):
		# Forward propagate the inputs
		z = tf.matmul(inputs, self.W) + self.b
		# Feed through a non-linear activation
		output = tf.math.sigmoid(z)
		return outputs
```
### Single Layer Neural Network_ Multi Output Perceptron
```python
import tensorflow as tf
layer = tf.keras.layers.Dense(units=2) # 2 outputs, 0 hidden layers
```
### Multi Layer Neural Netwok_ Multi Output Perceptron
```python
import tensorflow as tf
model = tf.keras.Sequential(
	[tf.keras.layers.Dense(n), # 1 hidden layers with n Neuros
	tf.keras.layers.Dense(2)]) # 2 outputs
```
### Deep Neural Network
```python
import tensorflow as tf
model = tf.keras.Sequential(
	[tf.keras.layers.Dense(n_1), # 1 hidden layers with n Neuros
	tf.keras.layers.Dense(n_2),
	...
	tf.keras.layers.Dense(2)# 2 outputs
	]) 
```
### Quantifying Loss
The **loss** of our network measures the cost incurred from incorrect predictions. 
我们网络的“亏损”衡量了由于错误的预测而产生的费用。
The **empirical loss** measures the total loss over our entire dataset.
经验损失衡量了整个数据集的总损失。--求和再平均
**Cross entropy loss** can be used with models that output a probability between 0 and 1.
交叉熵损失可与输出0到1之间的概率的模型一起使用。拓展--交叉熵损失常用于(二，多)分类问题中，经常和sigmoid, softmax一起使用。
```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predicted))
```
