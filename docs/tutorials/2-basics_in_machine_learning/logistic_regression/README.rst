

章节
~~~~~~~~

-  `介绍 <#介绍>`__
-  `描述整体过程 <#描述整体过程>`__
-  `如何在代码中做到这一点？ <#如何在代码中做到这一点？>`__
-  `总结 <#总结>`__

使用TensorFlow做Logistic Regression
------------------------------------

本教程是关于通过TensorFlow训练logistic regression的
二进制分类。

介绍
------------

使用 `TensorFlow <http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html>`__
做线性回归这篇文章中，我们描述了如何预测连续参数使用线性建模系统。 那么如果目标是在二者中选一个呢？答案很简单:我们是在处理一个分类问题。 在这篇教程中，目标是使用 Logistic Regression决定输入的照片是“0”还是“1” 换句话说，是数字“1”或者不是！全部的源代码在相关 `Github  <https://github.com/Machinelearninguru/Deep_Learning/tree/master/TensorFlow/machine_learning_basics/logistic_regression>`__ 仓库可见。

数据集
-------

本教程使用的数据集是
`MNIST <http://yann.lecun.com/exdb/mnist/>`__ 数据集。主要的数据集
包括55000个训练图像和10000个测试图像。 图像是28x28x1的，每一张图片是一个手写的从“0”到“9”的数字图片。每张图片就有784维特征向量。在本次测试中，我们仅使用“0” 和“1”图像。

Logistic Regression
-------------------

在线性回归中，努力的方向是预测连续的结果值利用线性函数y=W^T*x。 另一方面，在logistic regression中，我们做的是预测一个二分类标签（0或1）我们使用了和线性回归不同的过程。在logistic regression中，预测的输出结果为1，则表示输入的样本数属于目标类的，比如在我们这次测试中的图像“1”。 在一个二分类问题中， 显然如果x在目标类内的概率为m，那么x不在目标类的概率为1-m。 所以假设可以以如下方式被创建:

$$P(y=1\|x)=h\_{W}(x)={{1}\\over{1+exp(-W^{T}x)}}=Sigmoid(W^{T}x) \\ \\
\\ (1)$$ $$P(y=0\|x)=1 - P(y=1\|x) = 1 - h\_{W}(x) \\ \\ \\ (2)$$

在上述方程中，Sigmoid 方程将预测结果映射到[0,1]的概率空间中。 主要目的是找到一个模型，使用的时候 如果输入的图像是“1”则输出的可能性就高，否则输出的可能性就低。首要目的是设计合适的损失函数以最小化输出时的损失。（译者注：即设计模型降低错误率） 损失函数可以被定义为如下方式： 

$$Loss(W) =
\\sum\_{i}{y^{(i)}log{1\\over{h\_{W}(x^{i})}}+(1-y^{(i)})log{1\\over{1-h\_{W}(x^{i})}}}$$

As it can be seen from the above equation, the loss function consists of
two term and in each sample only one of them is non-zero considering the
binary labels.（此部分暂未翻译）

到目前为止，我们定义了logistic regression的公式和优化函数。 在下一部分，我们将演示如何在代码中使用mini-batch优化。

描述整体过程
----------------------------------

首先，我们处理数据集，并且只抽取“0”和“1”的图像。 实现logistic regression的代码很大程度上受到了 
`Train a Convolutional Neural Network as a
Classifier <http://www.machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html>`__
这篇文章的启发。 我们参考了前面的文章，以获得更好的效果
理解实现细节。 在这篇教程中，我们仅仅解释如何处理数据集并且如何实现logistic regression
，剩余部分可以从早些时间发布的文章CNN classifier处了解。

如何在代码中做到这一点？
---------------------

在这部分，我们解释如何从数据集抽取所需的样本并且使用Softmax实现logistic regression。

处理数据集
~~~~~~~~~~~~~~~

首先，我们需要从MNIST数据集中抽取 "0" and "1" 数字对应的图片:

.. code:: python

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

    ########################
    ### Data Processing ####
    ########################
    # Organize the data and feed it to associated dictionaries.
    data={}

    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels

    # Get only the samples with zero and one label for training.
    index_list_train = []
    for sample_index in range(data['train/label'].shape[0]):
        label = data['train/label'][sample_index]
        if label == 1 or label == 0:
            index_list_train.append(sample_index)

    # Reform the train data structure.
    data['train/image'] = mnist.train.images[index_list_train]
    data['train/label'] = mnist.train.labels[index_list_train]


    # Get only the samples with zero and one label for test set.
    index_list_test = []
    for sample_index in range(data['test/label'].shape[0]):
        label = data['test/label'][sample_index]
        if label == 1 or label == 0:
            index_list_test.append(sample_index)

    # Reform the test data structure.
    data['test/image'] = mnist.test.images[index_list_test]
    data['test/label'] = mnist.test.labels[index_list_test]

代码看起来很冗长，但实际上非常简单。所有我们想要实现的，在第28-32行中实现了，即抽取所需样本。
接下来，我们需要深入挖掘logistic regression的体系构造。

Logistic Regression实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

logistic regression结构只是简单的从前面“喂”入输入向量，通过全连接层，最后一层只有两个类。 全连接网络的构造定义如下： 

.. code:: python

        ###############################################
        ########### Defining place holders ############
        ###############################################
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)

        ##################################################
        ########### Model + Loss + Accuracy ##############
        ##################################################
        # A simple fully connected with two class and a Softmax is equivalent to Logistic Regression.
        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')

前面几行是定义占位符，用于在graph中存放值。具体请参考 `这篇文章 <http://www.machinelearninguru.com/deep_learning/tensorflow/neural_networks/cnn_classifier/cnn_classifier.html>`__
。 损失函数使用TensorFlow可以轻易实现，脚本如下： 

.. code:: python

        # Define loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

        # Accuracy
        with tf.name_scope('accuracy'):
            # Evaluate the model
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))

            # Accuracy calculation
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.nn.softmax\_cross\_entropy\_with\_logits这个函数做了这件工作。它以一种微妙的方式优化了前面定义的损失函数。它生成了两个输入，即使输入的样本是数字“0”，相应的概率会很高。
It generates two inputs in which even if the sample is digit
"0", the correspondent probability will be high. So
tf.nn.softmax\_cross\_entropy\_with\_logits function, for each class
predict a probability and inherently on its own, makes the decision.（此部分暂未翻译）

总结
-------

在这篇教程中，我们描述了logistic regression，并且演示了如何用代码实现它。我们将问题扩展到两个类每个类预测自身的概率，而不是基于输出的目标类的可能性来决定。 在未来的文章中，我们会扩展这个问题到多分类问题，并且我们会展示这可以用类似的方法做到。
