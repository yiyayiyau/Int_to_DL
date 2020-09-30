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
* The **loss** of our network measures the cost incurred from incorrect predictions. 
我们网络的“亏损”衡量了由于错误的预测而产生的费用。

* The **empirical loss** measures the total loss over our entire dataset.
经验损失衡量了整个数据集的总损失。--求和再平均

* Binary Cross entropy loss
Cross entropy loss can be used with models that output a probability between 0 and 1.
交叉熵损失可与输出0到1之间的概率的模型一起使用。拓展--交叉熵损失常用于(二，多)分类问题中，经常和sigmoid, softmax一起使用。
预测对的乘以log对的加上预测错的乘以log错的 求和再平均
```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predicted))
```
* Mean Squared Error Loss
Mean squared error loss can be used with regression models that output continuous real numbers. 真实值与预测值的差的平方求和再平均
```python
loss = tf.reduce_mean( tf.square(tf.subtract(y, predicted)) ) 
```

### Training Neural Networks
#### Loss Optimiation
我们要找的是权重，其只带来最小的损失。权重是高维的，在高维曲面中最小值即损失最小，所以该问题转换为找该高维空间的最小值。可采用梯度下降法Gradient Descent。 原理: 反向传播 Backpropagation (with chain rule)
```python
import tensorflow as tf
weights = tf.Variable([tf.random.normal()])

while True:
	with tf.GradientTape() as g:
		loss = compute_loss(weights)
		gradient = g.gradient(loss, weights)

	weights = weights - lr*gradient
# lr 是学习率，0...1
```
### Neural Networks in Practice: Optimization
Loss Functions Can Be Difficult to Optimize. **Small learning rate** converges slowly and gets stuck in false local minima. **Large learning rates** overshoot, become unstable and diverge. **Stable learning rates** converge smoothly and avoid local minima.
Idea 1: Try lots of different learning rates and see what works "just right".
Idea 2: Design an adaptive learning rate that "adapts" to the landscape.
Gradient Descent Algorithms
* SGD 		
```python
tf.keras.optimizers.SGD
```
* Adam
```python		
tf.keras.optimizers.Adam
```
* Adadelta	
```python
tf.keras.optimizers.Adadelta
```
* Adagrad	
```python
tf.keras.optimizers.Adagrad
```
* RMSProp	
```python
tf.keras.optimizers.RMSProp
```

```python
import tensorflow as tf
model = tf.keras.Sequential([...])
optimizer = tf.keras.optimizers.SGD()

while True:
	prediction = model(x)
	with tf.GradientTape() as tape:
		loss = compute_loss(y,prediction)
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

```
### Mini-batches while training
计算所有点的grandient太慢，可以只计算一部分点。
* More accurate estimation of gradient
* Smoother convergence
* Allows for larger learning rates
* Mini-batches lead to fast training. 
* Can parallelize computation + achieve significat spped increases on GPU's.

### The Problem of Overfitting
#### Regularization正则化1: Dropout
'drop' 50% of activations in layer

#### Regularization正则化2: Early Stopping
训练集的损失曲线会持续降低，测试集的损失曲线会先降低后增大。在测试集曲线即将增大的地方中止训练。