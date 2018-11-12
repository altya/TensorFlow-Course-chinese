
===========================
欢迎来到TensorFlow 世界 
===========================

这篇文档专注于解释如何运行本教程的python脚本。

---------------------------
Test TensorFlow Environment
---------------------------

``警告:`` 如果TensorFlow在任何环境安装了（虚拟环境，...），它必须先被激活。 所以，首先使用以下语句，保证tensorFlow在当前环境可用： 

.. code:: shell

    cd code/
    python TensorFlow_Test.py
    
--------------------------------
如何在终端运行代码？
--------------------------------

    
请到 ``code/`` 目录，并且以以下形式运行脚本： 

.. code:: shell
    
    python [python_code_file.py] --log_dir='absolute/path/to/log_dir'
    

一个样例代码可以以如下方式执行：

.. code:: shell
    
    python 1-welcome.py --log_dir='~/log_dir'

 ``--log_dir`` 这个flag提供了事务文件存放的地址（用于tensorboard可视化）。  ``--log_dir`` 标志不是必须的，因为他的默认值在以下源代码处已经设置：

.. code:: python
    
    tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

----------------------------
如何在IDEs中运行代码?
----------------------------

由于代码是立即可执行的，只要TensorFlow能够在IDE编辑器中(Pycharm, Spyder,..)被调用，代码就可以成功被执行。


----------------------------
如何运行Tensorboard?
----------------------------
.. _Google’s words: https://www.tensorflow.org/get_started/summaries_and_tensorboard
TensorBoard是一个由TensorFlow提供的图形化工具。 用 `Google’的话来说`_: “你会用tensorflow做的运算-比如计算一个复杂的深度神经网络-可能是复杂且令人困惑的。为了使之更容易理解，调式，并且优化TensorFlow编程，我们包含了一个名叫TensorBoard的可视化工具套件。”

Tensorboard可以使用以下命令来运行在命令行之中:

.. code:: shell
    
    tensorboard --logdir="absolute/path/to/log_dir"


 



