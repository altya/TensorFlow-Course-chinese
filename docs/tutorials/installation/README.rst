==================================
从源安装TensorFlow
==================================

.. _TensorFlow: https://www.tensorflow.org/install/
.. _Installing TensorFlow from Sources: https://www.tensorflow.org/install/install_sources
.. _Bazel Installation: https://bazel.build/versions/master/docs/install-ubuntu.html
.. _CUDA Installation: https://github.com/astorfi/CUDA-Installation
.. _NIDIA documentation: https://github.com/astorfi/CUDA-Installation



安装教程见 `TensorFlow`_ 。 推荐使用源文件安装，因为用户可以根据特定的体系结构构建所需的TensorFlow二进制文件。这种方式安装有更好的系统兼容性，且运行起来会快很多。使用源文件安装可以点击此链接（需要富强） `Installing TensorFlow from Sources`_ 。TensorFlow官方的解释简明扼要。但是，在安装的时候却没什么用。（...）  我们尝试一步步来，避免弄混。 下面各节为安装教程。

假设安装TensorFlow的环境为： ``Ubuntu`` 下安装，开启 ``GPU 支持`` 。 使用的python版本为 ``Python2.7`` 。

**注意** 安装过程视频可点击此链接查看。 `link <youtube_>`_ （为方便不能富强的同学，已将视频转至国内源） 

.. _youtube: http://v.youku.com/v_show/id_XMzg5Mzc3NDA0OA==.html?spm=a2h3j.8428770.3416059.1

------------------------
准备环境
------------------------

以下步骤应按序执行:
 
    * TensorFlow Python依赖安装
    * Bazel安装
    * TensorFlow GPU先决条件安装

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow Python依赖安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为了安装所需的依赖项，必须在终端执行以下命令:

.. code:: bash

    sudo apt-get install python-numpy python-dev python-pip python-wheel python-virtualenv
    sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel python3-virtualenv
    
第二行是为了 ``python3`` 安装。

~~~~~~~~~~~~~~~~~~~
Bazel安装
~~~~~~~~~~~~~~~~~~~

请参照 `Bazel Installation`_。

``警告:`` Bazel安装可能会改变GPU支持的内核!在那之后你可能需要刷新你的GPU安装，或者更新它，否则 你在安装 TensorFlow时可能会出现以下错误:

.. code:: bash

    kernel version X does not match DSO version Y -- cannot find working devices in this configuration
    
为了解决这个问题，你可能需要卸载所有NVIDIA驱动，并且重新安装。 详细可参照 `CUDA Installation`_ 。


    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow GPU先决条件安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下要求必须被满足:

    * NVIDIA's Cuda Toolkit和相关驱动(推荐version 8.0 )。安装解释见 `CUDA Installation`_ 。
    * cuDNN库(推荐 version 5.1)。 更多细节参照 `NIDIA documentation`_ 。
    * 安装 ``libcupti-dev`` 使用以下命令: ``sudo apt-get install libcupti-dev``

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
创建虚拟环境(可选)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

假设需要在“python虚拟环境”中安装TensorFlow。 首先，我们需要创建一个目录来容纳所有环境。 在命令行界面执行以下命令即可:

.. code:: bash

    sudo mkdir ~/virtualenvs

现在使用 ``virtualenv`` 命令, 虚拟环境就能被创建:

.. code:: bash

    sudo virtualenv --system-site-packages ~/virtualenvs/tensorflow

**环境激活**

目前为止, 被称为 *tensorflow* 的虚拟环境已经被创建。 为了激活环境，还需执行以下命令:

.. code:: bash

    source ~/virtualenvs/tensorflow/bin/activate

然而，这些命令太冗长了！

**Alias**

解决方法是使用alias，使其变得容易！让我们执行以下命令:

.. code:: bash

    echo 'alias tensorflow="source $HOME/virtualenvs/tensorflow/bin/activate" ' >> ~/.bash_aliases
    bash

再执行完以上命令后，请关闭命令行界面并重新打开。 现在运行以下脚本，TensorFlow环境就会被激活。

.. code:: bash

    tensorflow
    
**检查 ``~/.bash_aliases`` **

再次检查， ``~/.bash_aliases`` 在命令行界面使用 ``sudo gedit ~/.bash_aliases`` 命令。 该文件应该包含以下内容:

.. code:: shell

    alias tensorflow="source $HO~/virtualenvs/tensorflow/bin/activate" 
    

**检查 ``.bashrc`` **

同样的，让我们检查 ``.bashrc`` shell脚本，使用 sudo gedit ~/.bashrc 命令。 该脚本应该包含以下内容:
 
.. code:: shell

    if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
    fi
 

    
---------------------------------
安装配置
---------------------------------

首先，需要克隆Tensorflow仓库:

.. code:: bash

     git clone https://github.com/tensorflow/tensorflow 

在准备好环境之后，必须配置安装。 配置的“标志”非常重要，因为它们决定了如何安装和兼容TensorFlow !! 首先我们需要跳转到TensorFlow的根目录:

.. code:: bash

    cd tensorflow  # cd to the cloned directory

他与配置环境一起标记如下所示：

.. code:: bash

    $ ./configure
    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
    Do you wish to use jemalloc as the malloc implementation? [Y/n] Y
    jemalloc enabled
    Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
    No Google Cloud Platform support will be enabled for TensorFlow
    Do you wish to build TensorFlow with Hadoop File System support? [y/N] N
    No Hadoop File System support will be enabled for TensorFlow
    Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] N
    No XLA JIT support will be enabled for TensorFlow
    Found possible Python library paths:
      /usr/local/lib/python2.7/dist-packages
      /usr/lib/python2.7/dist-packages
    Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
    Using python library path: /usr/local/lib/python2.7/dist-packages
    Do you wish to build TensorFlow with OpenCL support? [y/N] N
    No OpenCL support will be enabled for TensorFlow
    Do you wish to build TensorFlow with CUDA support? [y/N] Y
    CUDA support will be enabled for TensorFlow
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
    Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
    Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5.1.10
    Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size.
    [Default is: "3.5,5.2"]: "5.2"


**注意:**
     * cuDNN版本必须使用/usr/local/cuda相关的版本。 
     * 计算能力与系统架构中的“可用GPU模型”相关。 例如 ``Geforce GTX Titan X`` GPUs 有 5.2的计算能力。
     *  推荐使用 ``bazel clean``  如果需要再次配置。

**警告:**
     * 如果需要在虚拟环境中安装TwnsorFlow，则必须在运行 ./configure 脚本前激活环境。
     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
测试Bazel (可选)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们可以运行 ``Bazel`` 测试，来确保一切环境都正常:

.. code:: bash

    ./configure
    bazel test ...

---------------------
构建the .whl包
---------------------

设置完成后，需要由Bazel构建pip包。
    
构建支持GPU的TensorFlow包，可执行以下命令：

.. code:: bash

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    
``bazel build`` 构建了一个脚本名叫 build_pip_package。在~/tensorflow_package目录下运行以下脚本构建 a .whl文件  :

.. code:: bash

    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_package





-------------------------------
安装Pip包
-------------------------------

两种安装方法可以被使用。使用系统原生安装，或者使用虚拟环境安装。 

~~~~~~~~~~~~~~~~~~~~~~~~~~~
原生安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下命令会安装Bazel构建的pip包：

.. code:: bash

    sudo pip install ~/tensorflow_package/file_name.whl
    

~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用虚拟环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先，虚拟环境必须被激活。由于我们已经定义了环境alias  ``tensorflow``，通过在命令行执行简单命令 ``tensorflow``，环境就会被激活。 然后像之前部分一样，我们执行以下命令:

.. code:: bash
    
    pip install ~/tensorflow_package/file_name.whl

**警告**:
           * 通过使用虚拟环境安装工具，sudo不应再被使用。因为如果我们使用sudo，它就会指向系统原生包而不是虚拟环境中的。
           * 由于 ``sudo mkdir ~/virtualenvs`` 命令是用来创建虚拟环境的。如果使用 ``pip install`` 返回 ``permission error``。 在这种情况下，环境文件夹的权限必须更改，使用以下命令 ``sudo chmod -R 777 ~/virtualenvs`` 。
    
--------------------------
验证安装
--------------------------

在终端，运行以下命令( ``在家目录下``) ，必须显示完全正确没有error或者warning:

.. code:: bash

    python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))

--------------------------
常见错误
--------------------------

TensorFlow编译和运行过程中遇到的不同的错误。

   * ``Mismatch between the supported kernel versions:`` 这个错误在文档前面部分提到过。简单的解决方案是重新安装CUDA驱动程序。
   * ``ImportError: cannot import name pywrap_tensorflow:`` 这个错误通常是Python从错误的目录加载tensorflow库，例如，不是用户在根目录下安装的版本。 首先确保我们在系统根目录中，以便正确使用python库。 所以，基本上我们可以重新打开一个新的终端，并且尝试再次安装TensorFlow。
   * ``ImportError: No module named packaging.version":`` 最有可能是与 ``pip`` 安装有关。 使用 ``python -m pip install -U pip`` 或者 ``sudo python -m pip install -U pip`` 命令重新安装，可能会解决！

--------------------------
总结
--------------------------

在这篇教程中，我们介绍了如何用源码方式安装TensorFlow，优点是系统兼容性更好。同样介绍了Python虚拟环境下安装为了和其他环境相隔离。 Conda环境也可以Python虚拟环境使用，在另一篇文章中会解释conda。在任何情况下，用源方式安装的TensorFlow 比安装预编译完的安装包快很多，虽然这种安装方式也增加了安装的复杂度。



