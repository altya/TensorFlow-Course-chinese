============================
欢迎来到TensorFlow世界
============================

.. _this link: https://github.com/astorfi/TensorFlow-World/tree/master/codes/0-welcome

本节教程是进入TensorFlow世界的开始。

本节教程是进入TensorFlow世界的开始。

我们使用Tensorboard来可视化输出。 TensorBoard是TensorFlow提供的图形可视化工具。 用 Google的话来说: “你将使用TensorFlow来做的运算- 比如训练一张巨大的深度神经网络 - 可能是复杂且混乱的。 为了更容易理解、调试和优化TensorFlow程序，我们包含了一组可视化工具，称为TensorBoard。” 本教程中使用了一个简单的Tensorboard实现。

**注意:*** 
     
     * 有关摘要操作、Tensorboard及其优点的详细信息超出了本教程的范围，将在更高级的教程中介绍。


--------------------------
准备环境
--------------------------

首先，我们需要引入必须的库文件。

.. code:: python
    
       from __future__ import print_function
       import tensorflow as tf
       import os

由于我们的目的是去使用Tensorboard, 我们需要一个目录去存放信息 (操作以及相应的输出，如果用户需要的话). 该信息被TensorFlow导出到 ``event files`` 。 event files可以被转化成可视化数据以便用户能够评估体系结构和操作。  存储这些event files 的``路径``以如下方式被定义:

.. code:: python
    
       # The default path for saving event files is the same folder of this python file.
       tf.app.flags.DEFINE_string(
       'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
       'Directory where event logs are written to.')

       # Store all elements in FLAG structure!
       FLAGS = tf.app.flags.FLAGS

``os.path.dirname(os.path.abspath(__file__))`` 命令获取到当前文件夹的名称。  ``tf.app.flags.FLAGS`` 指向所有被定义的flags使用 ``FLAGS`` 指示器. 从此flags 可以被调用，使用 ``FLAGS.flag_name``。

为方便起见，使用 ``绝对路径``非常有效。 使用以下脚本，用户使用绝对路径 ``log_dir`` 目录。

.. code:: python

    # The user is prompted to input an absolute path.
    # os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
    #       Example: '~/logs' equals to '/home/username/logs'
    if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
        raise ValueError('You must assign absolute path for --log_dir')

--------
基础
--------

一些基础的数学运算可以在TensorFlow中定义:

.. code:: python

     # Defining some constant values
     a = tf.constant(5.0, name="a")
     b = tf.constant(10.0, name="b")

     # Some basic operations
     x = tf.add(a, b, name="add")
     y = tf.div(a, b, name="divide")
    
 ``tf.`` 运算符执行特定操作， 并且输出会是一个 ``Tensor`` 。属性 ``name="some_name"`` 被定义为了Tensorboard更好的可视化，具体在本教程后面会看到。

-------------------
Run the Experiment
-------------------

 ``session``，是运行操作的环境, 执行命令如下:

.. code:: python

    # Run the session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
        print("output: ", sess.run([a,b,x,y]))

    # Closing the writer.
    writer.close()
    sess.close()

``tf.summary.FileWriter`` 被定义用来写总结到 ``event files``。 ``sess.run()``命令 必须被用来评价 ``Tensor`` 否则操作不会被执行。 最后通过使用 ``writer.close()``, summary writer会被关闭。
    
--------
结果
--------

在终端中运行的结果如下:

.. code:: shell

        [5.0, 10.0, 15.0, 0.5]


如果我们运行Tensorboard使用 ``tensorboard --logdir="absolute/path/to/log_dir"`` 命令。我们得到以下可视化 ``Graph``:

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/1-basics/basic_math_operations/graph-run.png
   :scale: 30 %
   :align: center

   **Figure 1:** The TensorFlow Graph.

