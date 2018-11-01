
章节
~~~~~~~~

-  `介绍 <#介绍>`__
-  `描述整体过程 <#描述整体过程>`__
-  `如何在代码中做到这一点？ <#如何在代码中做到这一点？>`__
-  `总结 <#总结>`__

使用TensorFlow做线性回归
----------------------------------

本教程是关于通过TensorFlow训练线性模型以拟合数据。 另外，你也可以查看此链接 `博客链接 <blogpostlinearregression_>`_ 。

.. _blogpostlinearregression: http://www.machinelearninguru.com/deep_learning/tensorflow/machine_learning_basics/linear_regresstion/linear_regression.html



介绍
------------

在机器学习和统计中，线性回归是对诸如Y的变量和至少一个独立变量X之间的关系的建模。 在线性回归中，线性
关系将由一个预测函数来建模参数由数据估计，称为线性模型。线性回归算法的主要优点是使用简单
，直接了当的解释新模型并把数据映射进一个新的空间。在本文中，我们将介绍如何使用TensorFLow训练线性模型以及如何展示生成的模型。

描述整体过程
----------------------------------

为了训练模型，TensorFlow循环遍历数据，它应该找到符合数据的最佳直线（因为我们有一个线性模型）。 通过设计适当的优化问题来估计X，Y的两个变量之间的线性关系，其中需求是适当的损失函数。 数据集可从
`Stanford course CS
20SI <http://web.stanford.edu/class/cs20si/index.html>`__: TensorFlow
for Deep Learning Research 获得。

如何在代码中做到这一点？
---------------------

通过加载必要的库和数据集来启动该过程：

.. code:: python


    # Data file provided by the Stanford course CS 20SI: TensorFlow for Deep Learning Research.
    # https://github.com/chiphuyen/tf-stanford-tutorials
    DATA_FILE = "data/fire_theft.xls"

    # read the data from the .xls file.
    book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    num_samples = sheet.nrows - 1

    #######################
    ## Defining flags #####
    #######################
    tf.app.flags.DEFINE_integer(
        'num_epochs', 50, 'The number of epochs for training the model. Default=50')
    # Store all elements in FLAG structure!
    FLAGS = tf.app.flags.FLAGS

然后我们继续定义和初始化必要的变量：
.. code:: python

    # creating the weight and bias.
    # The defined variables will be initialized to zero.
    W = tf.Variable(0.0, name="weights")
    b = tf.Variable(0.0, name="bias")

在那之后，我们应该定义必要的功能。不同的标签演示了定义的功能：

.. code:: python

    def inputs():
        """
        Defining the place_holders.
        :return:
                Returning the data and label lace holders.
        """
        X = tf.placeholder(tf.float32, name="X")
        Y = tf.placeholder(tf.float32, name="Y")
        return X,Y

.. code:: python

    def inference():
        """
        Forward passing the X.
        :param X: Input.
        :return: X*W + b.
        """
        return X * W + b

.. code:: python

    def loss(X, Y):
        """
        compute the loss by comparing the predicted value to the actual label.
        :param X: The input.
        :param Y: The label.
        :return: The loss over the samples.
        """

        # Making the prediction.
        Y_predicted = inference(X)
        return tf.squared_difference(Y, Y_predicted)

.. code:: python

    # The training function.
    def train(loss):
        learning_rate = 0.0001
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

接下来，我们将循环遍历不同的数据时期并执行优化过程：

.. code:: python

    with tf.Session() as sess:

        # Initialize the variables[w and b].
        sess.run(tf.global_variables_initializer())

        # Get the input tensors
        X, Y = inputs()

        # Return the train loss and create the train_op.
        train_loss = loss(X, Y)
        train_op = train(train_loss)

        # Step 8: train the model
        for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
            for x, y in data:
              train_op = train(train_loss)

              # Session runs train_op to minimize loss
              loss_value,_ = sess.run([train_loss,train_op], feed_dict={X: x, Y: y})

            # Displaying the loss per epoch.
            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))

            # save the values of weight and bias
            wcoeff, bias = sess.run([W, b])

在上面的代码中， sess.run(tf.global\_variables\_initializer())
全局初始化所有已定义的变量。train\_op 是建立在 train\_loss 之上并且每一步更新。 最后线性模型的参数, 例如, 多项式系数（w）偏差（b）, 会被返回。
为了评估，预测的线和原始数据会被展示，以显示模型如何与数据匹配：

.. code:: python

    ###############################
    #### Evaluate and plot ########
    ###############################
    Input_values = data[:,0]
    Labels = data[:,1]
    Prediction_values = data[:,0] * wcoeff + bias
    plt.plot(Input_values, Labels, 'ro', label='main')
    plt.plot(Input_values, Prediction_values, label='Predicted')

    # Saving the result.
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

结果如下图所示：

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/2-basics_in_machine_learning/linear_regression/updating_model.gif
   :scale: 50 %
   :align: center

**图 1:** 原始数据与估计的线性模型

上面的动画GIF显示了模型的一些微小运动演示更新过程。 可以观察到，线性的
模特当然不是最好的! 然而，正如我们提到的，简单是它的优点!

总结
-------

在本教程中，我们使用了TensorFlow创建线性模型。 训练后找到的那条线是不能保证的
做最好那条。不同的参数会影响收敛性的准确性。 线性模型使用随机优化和其简洁性使我们的工作变得很简单。
