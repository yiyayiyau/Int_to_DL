# RNN
单词: 

descendant 后裔

individual 个人,个体

involve 涉及

conveniently 方便地

derivative 导数，衍生物

mitigate 减轻 alleviate 减轻

vanishing 消失

address this problem 解决这个问题

shrinking and shrinking 不断缩小

eventually 最终

to perform our tasks of interest 执行我们感兴趣的任务

to make this concrete 具体一点说

obvious 明显的

gap 间隙，距离

prevent 防止

significant consideration 重要考虑

get rid of 摆脱

bog down 停滞

re-emphasize 重新强调

stuck 卡住

regulate 调节

bootcamp 训练营

periodic summaries 定期总结

distill down 蒸馏下来(取其精华的意思)

ultimately 最终

maintain 保持

brand new 全新的

composer 作曲家

encourage 鼓励

backbone 骨干

bottleneck 瓶颈

approach 方法

taken off 起飞

impactful 有影响力的

Trajectory 轨迹

Humidity 湿度

particulates 微粒

## Sequence data
预测句子中的下一个字或词。输入是前面的文字，需要预测下一个文字。

问题: 句子的长度不是确定的，输入的长度没法确定。

方法一: 确定输入窗口的长度，但是重要信息可能会被忽略掉。

无法建立有长期依赖性的模型: 我们需要的信息离要预测的位置较远

方法二: 使用整个序列作为一组计数, bag of words

计数不能保留秩序

方法三: 使用很大的固定窗口

不能分享参数，每个输入都有独立的参数。不同顺序的输入(相同意思)有不同的参数。

没有参数共享, 如果有关序列的知识出现在序列中的其他位置，则不会转移。

### 序列建模：设计准则

需要: 
* 解决输入长度不同的问题
* 解决长句子之间的依赖关系
* 顺序关系的问题
* 整句中信息共享的问题

一些神经网络的结构和应用: 

One (input) to One (output): "Vanilla" neural network 简单的"香草"神经网络

Many to One: Sentiment Classification 情感分类 

Many to Many: Music Generation 音乐生成

...


## RNN Concept
Recurrence relation: 在每个时间步骤应用递归关系以处理序列。 当前状态取决于前一个状态和当前输入。

注意：每个时间步使用相同的功能和参数集。

## RNN Intuition: 
```python
my_rnn = RNN()
hidden_state = [0,0,0,0]
sentence = ["I", "love", "recurrent","neural"]
for word in sentence:
	prediction, hidden_state = my_rnn(word, hidden_state)
next_word_prediction = prediction
```

## RNN State Update and Output

Input Vector: x_t

Update Hidden State: h_t = tanh(W_hh * h_t-1 + W_xh * x_t) (后文中的cell state)

Output Vector: y_t = W_hy * h_t

## RNNs: Computational Graph Across Time 跨时间的计算图

表示为跨时间展开的计算图。

在每个时间步重复使用相同的权重矩阵。

我们可以计算每一步的损失，然后相加得到总体的损失。

## RNNs from Scratch
```python
class MyRNNCell(tf.keras.layers.Layer):
	def __init__(self, rnn_units, input_dim, output_dim):
		super(MyRNNCell, self).__init__()
		# initialize weight matrix
		self.W_xh = self.add_weight([rnn_units, input_dim])
		self.W_hh = self.add_weight([rnn_units, rnn_units])
		self.W_hy = self.add_weight([output_dim, rnn_units])
		# initialize hidden state to zeros
		self.h = tf.zeros([rnn_units, 1])
	def call(self, x):
		# update the hidden state
		self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)
		# compute the output
		output = self.W_hy * self.h
		# return the corrent output and hidden state
		return output, self.h
```
### RNN Implementation in TensorFlow
```python
tf.keras.layers.SimpleRNN(rnn_units)
```

## Backpropagation Through Time(BPTT)

反向传播算法:
1. 对每个参数取损失的导数(梯度)
2. 移位参数以最小化损耗

随时间的反向传播:
1. 在每个单独的时间步进行反向传播，然后从当前位置一直到序列的开始一直跨所有时间步
2. 在每个时间步长之间，我们需要执行矩阵与矩阵W的乘法运算，单元更新也是由非线性激活函数引起的，这意味着梯度的计算是损耗相对于W的导数。一直追溯到我们的初始状态的参数需要对该权重矩阵进行多次重复乘法，以及重复使用激活函数的导数。这可能导致以下问题: 
	* 梯度爆炸: 出现原因: 当很多值(梯度)大于1。解决方法: 梯度裁剪以缩放大的梯度。
	* 梯度消失: 出现原因: 很多值小于1(梯度很小)。解决方法: 
		1. 激活函数 
		2. 初始化权重 
		3. 网络结构

为什么梯度消失是一个问题呢?

连续小数字的乘法运算->更进一步的后退导致更小的梯度->参数会偏差以捕获短期依赖性

*will actually end up biasing our network to capture more short term dependencies which may not always be a problem* 
==> 就不能解决长句中的信息共享问题(或有前后依赖关系的句子)

解决方法: 
1. 使用ReLU激活函数: for x>0: ReLU = 1,  tanh < 1,  sigmoid < 1. 使用ReLU防止梯度变小当 x>0 时。
2. 初始化参数权重为单位矩阵，初始化偏差biases为0。
3. 门控单元Gated Cells: 使用更复杂的带门的递归单元去控制哪些信息应该通过。比如LSTM，GRU等等。(更robust，更有效率)

Long Short Term Memory(LSTMs) 长短期记忆网络依靠门控单元来跟踪许多时间步长中的信息。

## Long Short Term Memory(LSTMs) Networks
```python
tf.keras.layers.LSTM(num_units)
```
信息是通过称为门的结构添加或删除的。 Gates可选地让信息通过，例如通过sigmoid神经网络层和逐点乘法。
工作原理: 
1. Forget: LSTMs会忘记先前状态的无关部分。通过例如Sigmoid函数(0-1)，决定一个信息会保留多少。
2. Store: LSTMs会保存新信息到状态中
3. Update: LSTMs会选择性的更新状态的值(这里的状态应该是internal cell state C)
4. Output: output gate 控制着哪些信息会传递到下一个时间段中。
LSTMs: Key Concepts
1. 保持与输出状态不同的 单元状态(cell state)
2. 使用门来控制信息的流动
	1. Forget gate 摆脱无关的信息
	2. 从当前的输入中保存相关的信息
	3. 有选择的更新 单元状态
	4. Output gate返回一个过滤版本的单元状态
3. 随时间的反向传播,梯度流不间断。(训练LSTM的关键在于保持这种独立的独立单元状态可以有效地训练LSTM随时间反向传播==>附录3)

## RNN Applications
### Example Task: Music Generation
Input: sheet music

Output: next character in sheet music
### Example Task: Sentiment Classification
Input: sequence of words

Output: probability of having positive sentiment
```python
loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)
```
### Example Task: Machine Translation
encoder decoder

编码的输入: 一种语言

编码的输出: 编码表示

解码的输入: 该编码表示

解码的输出: 另一种语言

如果输入过大，压缩成一个向量将会是困难重重的。因此有了"注意机制"。

Attention Mechanisms: 神经网络中的注意机制提供了可学习的内存访问。


### 另外的例子: Trajectory Prediction: 轨迹预测, 自动驾驶

### 另外的例子: Environmental Modeling

## 总结
1. RNNs 非常适合序列建模任务
2. 序列建模建立在递归关系的基础上
3. 通过 随时间的反向传播 训练RNNs
4. Gated cells (例如LSTMs) 让模型可以 长期依赖
5. 模型 生成音乐，分类，机器翻译等等


## 附录: 
1. 关于时间的反向传播的描述:
*our forward pass through the network also consists
of going forward across time updating the cell state based on the input and the previous state generating an output Y at that time step computing a loss at that time step and then finally summing these losses from the individual time steps to get our total loss and so that means instead of back propagating errors through a single feed-forward network at a single time set in RNNs errors are back propagated at each individual time step and then across all time steps all the way from where we are currently to the beginning of the sequence and this is the reason why it's called back propagation through time  because as you can see all the errors are flowing back in time to the beginning of our data sequence and so if we take a quick closer look at how gradients actually flow this chain of repeating modules you can see that between each time step we need to perform a matrix and multiplication involving this way matrix W and remember that this the cell update also results from a nonlinear activation function and that means that this computation of gradient that is the derivative of the loss with respect to the parameters tracing all the way back to our initial state requires many repeated multiplications of this weight matrix as well as repeated use of the derivative of our activation function.*

2. 关于LSTM的描述:
*forgetting irrelevant history, storing what's new and what's important, using that to update the internal state and generating an output*

3. 关于不间断训练LSTM的描述: 
*the keypoint in terms of training LSTM is that maintaining this separate independent cell state allows for the efficient training of LSTM to backpropagation through time*