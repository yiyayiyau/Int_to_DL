# L4_deep_ganerative_models
[link](https://www.youtube.com/watch?v=rZufA635dq4&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4)

## 单词: 

emerging 新兴的

resemble 类似

encounter 遭遇

leverage 利用

harsh 苛刻

harsh weather 恶劣的天气

pedestrians 行人

alluded 暗示

drastically 剧烈地

slight twist 轻微扭曲

enforce 执行

evenly 均匀地

penalise 惩罚

divergence 分歧

inferred latent distribution 推断潜在分布

amateur 业余

compact 紧凑

to obtain 获得

associated with 与。。。有关

reverse inference 反推

capturing the pixel wise difference 捕捉像素差异

constraints 约束

resembles as  像

essentially 本质上

adopts 采用

drive this point home 把这一点带回家(彻底理解这一点)

captures the divergence 捕捉分歧

latent perturbation 潜在扰动

distinct semantic meaning 独特的语义

interpret 解释

walking along two axes 沿两个轴行走

impose on 把...强加于

tune to 调到

disentanglement 解开

Debiasing 去偏

uncovering 揭露

capable of 能够

underlying 底层的

distinguish 区分

homogeneous 相同性质的

diverse 多种

imitation 仿制

breakthrough 突破

convey 传达

progressively 逐步

celebrity 名人

recent advances 近期进展

supplemental 补充的

synthesize 合成

appreciate 欣赏

competition 竞争

applause 掌声

问题: 怎样检测一些新出现的或者不常见的情况?

策略：利用生成模型，检测分布中的异常值。

在训练过程中使用离群值以进一步改善。

Latent Variable Models: 
* Autoencoders and Variational Autoencoders(VAEs)
* Generative Adversarial Networks(GANs)

什么是潜在的变量: 即隐藏在可以观测到的变量后面的变量。
我们在生成模型中的目标就是根据可以观测的数据(输入)，找到这些潜在的变量(真正的解释因素)。

## Autoencoders 自动编码器
从未标记的训练数据中学习低维特征表示的无监督方法。
* 编码器(Encoder)学习从数据x到低维潜在空间z的映射
* 解码器(Decoder)学习从潜在空间z到重构观测x_^的映射
对图像而言，损失函数就可以是真实数据与重构数据的差的平方。损失函数不使用任何标签。

潜在空间的维度决定了重构的质量。以MNIST数据集来说，5维的潜在空间比2维的潜在空间重构出来的更清晰。
自动编码器是压缩的一种形式。小的潜在空间将面临更大的训练瓶颈。

用于表示学习的自动编码器:
* 瓶颈隐藏层(Bottleneck hidden layer)迫使网络学习压缩的潜在表示(a compressed latent representation)。
* 重建损失(Reconstruction loss)迫使潜在表示捕获（或编码）关于数据的尽可能多的信息。
* 自动编码器 = 自动 编码 数据。 **Autoencoding** = **Auto**matically **encoding** data

## Variational Autoencoders (VAEs)
VAEs是自动编码器的概率扭曲。VAEs不直接产生潜在空间，从均值和标准差中采样以计算潜在采样，产生一个潜在空间的概率表示。编码器产生一个潜在空间的概率分布p_phi(z|x), 解码器q_theta(x|z)。损失函数是重构损失和正则化部分损失。重构损失和自动编码器中的相同，即重构结果与输入之间的均方误差。

由于VAE学习这些概率分布，我们希望对这种概率分布的计算方式以及该概率分布在正则化和训练网络过程中的表现方式做一些限制，因此，要做的是将先验值(prior)放入潜在分布，即p(z)。

因此，这是关于z分布看起来是什么的一些初步假设，这实际上有助于使学习的z遵循该先前分布(prior distribution)的形状。

我们这样做的原因是为了帮助网络不会过度拟合，因为如果没有这种正则化，它可能会过度拟合潜在空间的某些部分，但是如果我们强制每个潜在变量采用与此类似的内容，那么它将有助于平滑空间和学习的分布。

这个正则化项是一个函数，它捕获推断的潜在分布与我们放置在此的固定先验之间的差异，因此，正如我提到的，此先验分布的一个常见选择是正态高斯，均值0和标准差1。

我们能够做的是，得出关于网络可以做的最佳范围的一些非常好的属性。通过事先选择此高斯正态分布，可以鼓励编码器将潜在变量均匀地分布在该潜在空间的中心周围，从而平滑地分配编码，实际上，网络在尝试作弊和当聚类点超出了这种平滑的高斯分布时会学会惩罚自己，因为它可能是过拟合或试图记住数据的特定实例。

在选择正态高斯作为先验的情况下，也可以得出这个特定的距离函数，这称为KL散度，特别是当先验分布为标准的高斯分布时，它衡量了我们推断的潜在分布与该先验分布之间的距离，采用了这种特殊形式。

常选择的先验分布: 标准高斯分布
* 鼓励编码器将潜在变量均匀地分布在该潜在空间的中心周围，从而平滑地分配编码
* 网络通过在特定的区域聚集点(记住数据的特定实例)来作弊时会惩罚自己。

此时，遇到一个问题，这时的网络不能通过反向传播来训练，因为此时的潜在空间是随机过程。(我们无法通过采样层反向传播渐变。)

### 重新设定采样层
考虑到采样潜在向量z是
* 固定向量 miu
* 固定向量 sigma, 由先前分布得出的随机常数缩放
的总和。 ==> z = miu + sigma dotproduct epsilon, epsilon 是标准高斯分布，替代之前的标准高斯分布z

### 潜在扰动
* 理想情况下，我们需要彼此不相关的潜在变量。 
* 在潜在变量上强制对角线，以鼓励独立性。
==> Disentanglement

### 为什么要生成模型？ 去偏

能够发现数据集中潜在的(底层的)潜在变量。

我们如何使用潜在分布来创建公平且具有代表性的数据集？

Mitigating bias through learned latent structure
* Learn latent structure
* Estimate disribution
* Adaptively resample data
* learn from fair data distribution (Latent distributions used to creat fair and representative dataset)

### VAE总结
1. Compress representation of world to something we can use to learn
2. Reconstruction allows for unsupervised learning (no labels)
3. Reparameterization trick to train end-to-end
4. Interpret hidden latent variables using perturbation
5. Generating new examples

## GAN(Generative adversarial Network) 生成对抗网络

想法: 不要显式地对密度建模，而只是采样以生成新实例。

问题: 想从复杂的分布中取样-无法直接执行此操作

结果: 从简单的样本中学习，转变为学习分布

GANs是通过使两个神经网络相互竞争来生成生成模型的方法。**生成器**将噪声截断成数据的模仿物，以试图欺骗鉴别器。**鉴别器**尝试从生成器创建的伪造品中识别真实数据。

网络中的生成器随机产生噪声并经过训练以学习从该噪声到训练分布的变换，我们的目标是我们希望此生成的伪样本尽可能接近真实数据。

生成器生成假数据，然后和真实数据一起传给判别器，判别器尝试分辨哪个是真的哪个是假的。生成器尝试生成更好的假数据，直到判别器分不出来。

经过训练，生成器可以产生以前从未出现过的新的数据。

### GANs近期进展
* 逐步发展，层数加深，可以有更好的结果
* CycleGAN通过未配对的数据学习跨域的转换。应用比如视频领域，一段马的视频可以转换成一段斑马的视频。声音领域，一个人的声音转换成另一个人的声音(一个人的视频转换成另一个人的视频)。

### 深度生成模型总结
* AEs and VAEs: 学习低维潜在空间,采样以生成输入重构。
* GANs: 竞争的生成器和鉴别器网络。



## 附录
### 自动编码器: 
*Autoencoders: backgraound: Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data.*

### Loss of regularization term:
*because the VAE is learning these probability distributions, we want to place some constraints on how this probability distribution is computed and what the probability distribution resembles as a part of regularizing and training our network and so the way that's done is by placing a prior on the latent distribution and that's p of z. 
And so that's some initial hypothesis or guess about what the distribution of z looks like and this essentially helps enforce the learn z's to follow the shape of that prior distribution. And the reason that we do this is to help the network not over fit because without this regularization it may over fit on certain parts of the latent space but if we enforce that each latent variable adopts something similar to this prior it helps smooth out the landscape of the lane space and the lerned distributions. 
This regularization term is a function that captures the divergence between the inferred latent distribution and this fixed prior that we've placed so as I mentioned a common choice for this prior distribution is a normal Gaussian which means that we center it around with a mean of 0 and a standard deviation 1.*

*what this enables us to do is derive some very nice properties about the optimal bounds of how well our network can do. And by choosing this normal Gaussian prior what is done is the encoder is encouraged to sort of put the distribute the latent variables evenly around the center of this latent space distributing the encoding smoothly and actually the network will learn to penalise itself when it tries to cheat and cluster points outside sort of this smooth Gaussian distribution as it would be the case if it was overfitting or trying to memorize particular instances of the data.
And what also can be derived in the instance of when we choose a normal Gaussian as our prior is this specific distance function which is formulated here and this is called the KL divergence, and this is specifically in the case when the prior distribution is a zero one Gaussian the divergence that measures the separation between our inferred latent distribution and this prior takes this particular form.*

### latent perturbation 潜在扰动
*Ideally, we want latent variables that are uncorrelated with each other. Enforce diagonal prior on the latent variables to encourage independence Disentanglement.*

### VAEs
[CSDN](https://blog.csdn.net/mieleizhi0522/article/details/83822414)

*现在我们还不能产生任何未知的东西，因为我们不能随意产生合理的潜在变量。因为合理的潜在变量都是编码器从原始图片中产生的。我们可以对编码器添加约束，就是强迫它产生服从单位高斯分布的潜在变量。正式这种约束，把VAE和标准自编码器给区分开来了。现在，产生新的图片也变得容易：我们只要从单位高斯分布中进行采样，然后把它传给解码器就可以了。
事实上，我们还需要在重构图片的精确度和单位高斯分布的拟合度上进行权衡。
我们可以让网络自己去决定这种权衡。对于我们的损失函数，我们可以把这两方面进行加和。一方面，是图片的重构误差，我们可以用平均平方误差来度量，另一方面。我们可以用KL散度来度量我们潜在变量的分布和单位高斯分布的差异。
为了优化KL散度，我们需要应用一个简单的参数重构技巧：不像标准自编码器那样产生实数值向量，VAE的编码器会产生两个向量:一个是均值向量，一个是标准差向量。
当我们计算解码器的loss时，我们就可以从标准差向量中采样，然后加到我们的均值向量上，就得到了编码去需要的潜在变量。
VAE除了能让我们能够自己产生随机的潜在变量，这种约束也能提高网络的产生图片的能力。
编码越有效，那么标准差向量就越能趋近于标准高斯分布的单位标准差。
这种约束迫使编码器更加高效，并能够产生信息丰富的潜在变量。这也提高了产生图片的性能。而且我们的潜变量不仅可以随机产生，也能从未经过训练的图片输入编码器后产生。
VAE好处，就是我们可以通过编码解码的步骤，直接比较重建图片和原始图片的差异，但是GAN做不到。
另外，VAE的一个劣势就是没有使用对抗网络，所以会更趋向于产生模糊的图片。*

### KL散度(Kullback–Leibler divergence)
*KL散度是度量两个分布之间差异的函数。在各种变分方法中，都有它的身影。*

### 端到端学习 end-to-end
*而深度学习模型在训练过程中，从输入端（输入数据）到输出端会得到一个预测结果，与真实结果相比较会得到一个误差，这个误差会在模型中的每一层传递（反向传播），每一层的表示都会根据这个误差来做调整，直到模型收敛或达到预期的效果才结束，这是端到端的。*

### GANs
*GANs are a way to make a generative model by having tow neural networks compete with each other.
The generator truns noise into an imitation of the data to try to trick the discriminator.
The discriminator tries to identify real data from fakes created by the generator.*

*this generator in network is trained to learn a transformation going from that noise to the training distribution and our goal is we want this generated fake sample to be as close to the real data as possible*