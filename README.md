# 跬步千里

**Weekly Report.**

The best moments usually occur when a person's body or mind is stretched to its limits in a voluntary effort to accomplish something difficult and worthwhile.

## 2023/07/05~2023/07/12 第一周

### 概述

这周的工作主要集中在Stanford的一篇文章的上，需要对文章的一些实验进行一个复现，然后第二个就是要完全读懂文章和它对应的代码。

### 完成的任务

- 代码基本完全扫过一边，基本读懂了；花了大概两三天的时间。
- 完成了自己电脑环境的配置；大概一天
- 复现了文章提供的demo模拟实验；一个下午

### 进展和挑战

- XML物理环境构建，比较难阅读，这方面的基础比较差，需要进一步了解

- 现在思考Mujoco仿真到Issac Gym的转变

### 下周计划

- 对Learning Bimanual Manipulation这个任务的深度理解，能够提出问题，甚至是改进方法
- PowerPoint准备下周的group meeting.

## 2023/07/13 ~ 2023/07/19 第二周

### Summary

The work I focused on this week is to continue to explore the paper from Stanford. To comprehend it deeply, I read papers about BERT and DETR, whose idea is used in that paper. CLS token has a representation from the global information level, which can be used as downstream tasks. In this paper, CLS output is used as style variant, namely latent code. As for DETR, it's originally used in object detection. But here, its brief or easy structure has also great performance.

### Completed Tasks

- Understand why it uses BERT and DETR ideas to solve problems in that paper.
- Passed group meeting successfully.
- Watched RSS conference live stream replay, some paper in RL implementation.

### Challenges

- Is there any idea to improve the current performance?
- Is DETR the best structure to fit this environment?
- I still can't use Isaac Gym simulation platform to do some simulation tasks.

### Next Week Plan

- Simulate the environment that the paper said totally. Fix the camera problem, and environment issues.
- Continue reading the state-of-art papers.

## 2023/07/20 ～ 2023/07/26 第三周

### 概述

这周的工作主要集中在思考如何把代码移植到Isaac gym上面，完成了以下一些事情：1. MJCF以及URDF这两类机器人描诉语言的学习，具体而言这两种实际上都是xml描述文件，并且主要是针对于树形结构的描述，描述语言中的<font color="blue">各个变量的物理意义基本都能够理解</font>，对于MJCF/URDF文件阅读没有很大问题；2. <font color="blue">实现MJCF文件在Isaac Gym仿真环境的导入</font>，解决了一系列的奇怪的问题，认识到Mujoco和Gym的不同，Gym最好是一个actor一个actor进行仿真建模的，而Mujoco是对整个场景进行导入（对整个场景进行建模在gym中会有问题）；3. ROS的学习，学URDF的时候不可避免的要部分ROS知识，同时解决了URDF导入的小问题。

### 完成的任务

- 仿真前MJCF/URDF机器人文件的导入
- ROS的学习
- Isaac Gym的学习

### 遇到的困难

- 问题1：MJCF文件它在Isaac Gym上有一些困难：worldbody只能有一个body。 此外，它们不能同时属于父体和子体，除非父体是世界体。
  - 方案：对整个场景重新梳理树形结构，重写xml文件
- 问题2：Isaac Gym会忽略一些MJCF文件的一些物理量的设定（之后对比的时候明显发现摩擦系数有很大区别，mjcf导入的模型再有重力的条件下下垂后会不停摆动，而urdf会停下）
  - 方案：用URDF进行仿真模型文件
- 问题3：Isaac Gym貌似不支持直接在URDF中设定texture（texture路径不识别的问题）
  - 方案：Gym提供了api，需要在python文件中进行修改
- 问题4：官方URDF的夹子与原文的夹子不同，原文是3D打印的新夹子，是为了更好的执行一些精细任务
  - 方案：修改URDF文件，浪费了好多时间

### 下周任务

- 实现在Isacc gym上进行仿真

## 2023/07/27 ～ 2023/08/02 第四周

### 概述

这周考虑到任务的复杂程度，以及目标只是为了去利用isaac gym平台得到一些方针数据，所以首先考虑利用它提供的一些例子代码，我主要考虑Franka -> stack cube这个任务去收集它的observations和actions。过程中最大的困难就是isaac gym envs代码量很大，需要完全吃透整个工程的代码结构才好进行修改，目前位置基本上是<font color="blue">掌握了isaac gym进行RL的过程以及整个的代码结构</font>。

### 完成的任务

- isaacgymenvs仓库（gym仿真的包）的代码结构的理解
- 获得了fanka进行stack cube这个任务的数据（摄像头的图片、joint positions、actions），但是这个actions有很大不同

### 遇到的困难

- 问题1：isaacgymenvs进行RL的逻辑结构

​	`cfg`里面包含的是配置文件，通过hydra进行导入，这里包括task配置（env、sim），强化学习算法配置PPO，pbt是如果需要进行多gpu并行训练的时候需要添加的配置，cfg文件下还有一个config.yaml文件这个是用来进行override一些配置参数的。

​	`learning`里面包含的是强化学习算法，train设计使用的是a2c_continuous的RL算法。train是agent，test是player。

​	`pbt`它是pbt算法（Population-Based Training）

​	`	tasks`任务定义，定义observation维度，action维度。它继承vec_task他的任务就是创建仿真环境，其中step很重要，它是在执行action后更新环境的函数，返回obs, rewards, resets(是否完成)，info（超时信息）

​	`	train.py`利用runner进行train or test， 这里只关注players，这里面实现了get_action，使用model得到包括action等信息，players继承player，run函数是核心，在这里是信息agent和env互动的函数，在这里导出整个操作的数据。

​	buffer这些在执行过程中的一些内容信息，会在task中或者说vec_task的继承中更新。

- 问题2：obs怎么定义？
  - 方案：obs在compute_observations中实现，它实际上是直接从states中抽出来，可以进行自定义，比如说添加摄像头的handles
- 问题3：如何实现读取env或者说创建env的？
  - 方案：isaacgymenvs.make里面会调用一个map，实际上env就是自己写的task类，后面进行play的时候需要利用好env的attribution
- 问题4：step和run它们的作用分别是什么？
  - 方案：run会调用step，这是进行整个task可以这样认为；那么step就是这个task过程的一帧的update或者是物理环境在一个timestep的更新
- 问题5：franka的人物没有用到摄像头，怎样添加这个camera？
  - 方案：所谓的states并没有那么有很大的作用，RL算法使用到的数据是observations进行reward的计算，所以初步认为，image可以放在state，然后camera可以在create_sim中进行添加；

### 下周任务

- 用ACT框架跑一跑franka的任务
- 这次是修改了package源代码，可能以后会有问题，最好写出一版不要改package源代码的代码
- 实现VX300S机械臂的仿真代码



## 2023/08/03 ～ 2023/08/9 第五周

### 概述

这种的工作完成了ACT框架的修改，能够导入stack cube的task配置文件，并且实现了这个任务的train和inference步骤，为了更好的实验效果，对cube的初始位置进行了固定，以及franka的初始姿态也固定在一定范围，让它不是一个完全随机的过程。最终的结果并不是很理想，实验效果比较差。

### 完成的任务

- franka机械臂stack cube任务在ACT网络框架下的训练和测试

### 遇到的困难

- Q1: 数据纬度需要进行统一
  - S1: 人为定义一个上限 然后对所有文件进行整体padding

- Q2: isaacgymenvs.make有问题
  - S2: 项目有对hydra的依赖，如果直接进行make一个环境会有问题，按照原项目的train函数进行模仿重新写一个train和inference的函数

## 2023/08/10～2023/08/16

Diffusion Policy

## 2023/08/17～2023/08/23

Implement ACT in `DP` code frame. 

## 2023/08/24~2023/08/30

Comparative Experiments. 

- [x] rewrite `experiments.md`, e.g. training, formulations, environment settings, and etc.
- [x] Use IBC to execute push-t task.
- [ ] epoch-success rate png.

