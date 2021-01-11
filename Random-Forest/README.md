# README

## 1. 使用方法

### 1.1 数据要求

* 要求训练数据中有一列名为`index`，表示每一个样本的唯一标识码，int型；每个特征的特征名都不相同，每个特征都用一列表示，int型；样本的`label`也用一列表示；
* 要求测试数据有一列名为`index`，表示每一个样本的唯一标识码，int型；特征名和训练数据的特征名相同，每个特征都用一列表示，int型。

### 1.2 参数设置

* `max_depth`：决策树的最大深度，可以控制决策树的深度来防止决策树过拟合。默认是None，如果不是None，要求是大于0的整数。通常情况下，数据少或特征少时可以不限制`max_depth`；如果模型的样本量多且特征也多的情况下，推荐限制`max_depth`，具体的取值取决于数据的分布，常用取值在10~100之间。

* `min_samples_split`：内部结点再划分所需最小样本数。当结点的样本数少于`min_sample_split`时，不再继续分裂。默认是2，要求是大于0的整数。通常情况下，如果样本量不大，可以不限制`min_samples_split`，如果样本量数量级非常大，则推荐增大这个值。

* `min_impurity_split`：结点划分的最小不纯度。当结点$Gini$系数必须小于`min_impurity_split`时，该结点不再继续分裂。默认值为1e-7，不推荐改动默认值，要求大于0。

* `max_features`：随机森林划分时考虑的最大特征数。默认是None，表示划分时考虑所有的特征数；如果是“log2”，表示划分时最多考虑$\lceil log_{2}^{N} \rceil$个特征；如果是"sqrt"，表示划分时最多考虑$\lceil \sqrt n \rceil$个特征；如果$0 < max\_features < 1$，表示划分时最多考虑$\lceil max\_features*N \rceil$个特征。

* `n_estimators`：随机森林中决策树的数量。默认是10，要求为大于0的整数。

### 1.3 具体步骤

​		1. 需要将训练数据集放在`data/train/`目录下，将测试数据集放在`data/test/`目录下；

​		2. 在`code/config.json`中设置参数`is_train`，如果设置为`true`，则需要设置1.2中提到的参数来构建随机森林模型；如果设置为`false`，则需要设置参数`model_file`为训练好的模型路径；

​		3. 在`code/main.py`中修改数据集的路径，运行`main.py`即可。

## 2. 算法思想

### 2.1 基于分类回归树（CART）的决策划分

#### 2.1.1 算法组成

​		本项目采用的决策树分类算法是基于分类回归树（CART）的决策划分，由**决策树生成**和**决策树剪枝**两部分组成：

* 决策树生成：基于训练数据集生成决策树，生成的决策树要尽量的大；
* 决策树剪枝：用验证数据集对以生成的树进行剪枝并选择最优子树，这时用损失函数最小作为剪枝的标准。

#### 2.1.2 算法特点

​		CART算法有以下三个方面的特点：

* **二分**：在每次判断过程中，都是对样本数据进行二分。CART算法是一种二分递归分割技术，把当前样本划分为两个子样本，使得生成的每个非叶子结点都有两个分支，因此CART算法生成的决策树是结构简洁的二叉树。由于CART算法构成的是一个二叉树，它在每一步的决策时只能是“是”或者“否”，即使一个feature有多个取值，也是把数据分为两部分。
* **单变量分割**：每次**最优划分**都是针对**单个变量**。
* **剪枝策略**：CART算法的关键点，也是整个Tree-Based算法的关键步骤。对于不同的划分标准生成的最大树，在剪枝之后都能够保留最重要的属性划分，差别不大。反而是剪枝方法对于最优树的生成更为关键。CART树生成就是递归的构建二叉决策树的过程，对回归使用平方误差最小化准则，对于分类树使用**基尼指数**准则，进行特征选择，生成二叉树。

#### 2.1.3 基尼指数

​		基尼指数是描述混乱的程度。基尼指数 $Gini(D)$ 表示的是集合 $D$ 的不确定性，基尼指数 $Gini(D, A)$ 表示数据集 $D$ 经过特征 $A$ 划分以后集合 $D$ 的不确定性，基尼指数越大说明集合的不确定性越大。

​		给定样本集$D$，假设有$K$个类，样本点属于第$k$个类的概率为$p_{k}$，则概率分布的基尼指数定义为：
$$
Gini(D)=\sum_{k=1}^{K}p_{k}(1-p_{k})=1-\sum_{k=1}^{K}p_{k}^{2}.
$$
​		根据基尼指数定义，可以得到样本集合$D$的基尼指数，其中$C_{k}$表示数据集$D$中属于第$k$类的样本子集。
$$
Gini(D)=1-\sum_{k=1}^{K}(\frac{|C_{K}|}{D})^{2}
$$
​		基尼值反映了从数据集中随机抽取两个样本，其类别标记不一致的概率，因此基尼值越小，则数据集纯度越高。

​		对属性$a$进行划分，则属性$a$的基尼指数定义为：
$$
Gini\_index(D, a)=\sum_{v=1}^{V}\frac{|D^{v}|}{|D|}Gini(D^{v}).
$$
​		如果数据集$D$根据特征$A$在某一取值$a$上进行分割，得到$D_{1}$，$D_{2}$两部分后，那么在特征$A$下集合$D$的基尼系数：
$$
Gini\_index(D, a)=\frac{D_{1}}{D}Gini(D_{1})+\frac{D_2}{D}Gini(D_{2}).
$$
​		于是，我们在候选属性集合$A$中，选择哪个使得划分后基尼指数最小的属性作为最优划分属性，即
$$
a_{*}=argmin_{a \in A}Gini\_index(D, a).
$$
​		其中，算法停止的条件有：

* 节点中的样本个数小于预定阈值；

* 样本集的$Gini$系数小于预定阈值（此时样本基本属于同一类）；
* 没有更多特征。

### 2.2 基于随机森林的决策分类

#### 2.2.1 算法概述

​		随机森林是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树。集成学习就是使用一系列学习器进行学习，并将各个学习方法通过某种特定的规则进行整合，以获得比单个学习器更好的学习效果的一种机器学习方法。集成学习通过建立几个模型，并将它们组合来解决单一预测问题。它的工作原理主要是生成多个分类器或模型，各自独立的学习和做出预测。

​		随机森林是由**多棵决策树**构成的。对于每棵树，它们使用的**训练集**是采用**放回**的方式从总的训练集中采样出来的。而在训练每棵树的节点时，**使用特征**是从所有特征中，采用按照**一定比例随机的无回放的**方式抽取的。

#### 2.2.2 算法优缺点

随机森林的优点：

* 随机森林算法能解决分类与回归两种类型的问题，并在这两个方面都有相当好的估计表现。
* 随机森林对于高维数据集的处理能力令人兴奋，它可以处理成千上万的输入变量，并确定最重要的变量，因此被认为是一个不错的降维方法。此外，该模型能够输出变量的重要性程度，这是一个非常便利的功能。
* 在对缺失数据进行估计时，随机森林是一个十分有效的方法。就算存在大量的数据缺失，随机森林也能较好地保持精确性。
* 当存在分类不平衡的情况时，随机森林能够提供平衡数据集误差的有效方法。
* 模型的上述性能可以被扩展运用到未标记的数据集中，用于引导无监督聚类、数据透视和异常检测。
* 随机森林算法中包含了对输入数据的重复自抽样过程，即所谓的bootstrap抽样。这样一来，数据集中大约三分之一将没有用于模型的训练而是用于测试，这样的数据被称为out of bag samples，通过这些样本估计的误差被称为out of bag error。研究表明，这种out of bag方法的与测试集规模同训练集一致的估计方法有着相同的精确程度，因此在随机森林中我们无需再对测试集进行另外的设置。
* 训练速度快，容易做成并行化方法。

随机森林的缺点：

* 随机森林在解决回归问题时并没有像它在分类中表现的那么好，这是因为它并不能给出一个连续型的输出。当进行回归时，随机森林不能够作出超越训练集数据范围的预测，这可能导致在对某些还有特定噪声的数据进行建模时出现过度拟合。
* 对于许多统计建模者来说，随机森林给人的感觉像是一个黑盒子——你几乎无法控制模型内部的运行，只能在不同的参数和随机种子之间进行尝试。

#### 2.2.3 随机森林的构造方法

​		随机森林的建立基本由随机采样和完全分裂两部分组成：

* **随机采样**：随机森林对输入的数据进行行、列的采样，但两个采样的方法有所不同。对于行采样，采用的方法是有回放的采样，即在采样得到的样本集合中，可能会有重复的样本。假设输入样本为N个，那么采样的样本也是N个，这样使得在训练时，每棵树的输入样本都不是全部的样本，所以相对不容易出现over-fitting。对于列采样，采用的方式是按照一定的比例无放回的抽取，从M个feature中，选择m个样本（m<<M）。
* **完全分裂**：在形成决策树的过程中，决策树的每个节点都要按完全分裂的方式来分裂，直到节点不能再分裂。采用这种方式建立出的决策树的某一叶子节点要么是无法继续分裂的，要么里面的所有样本都是指向同一个分类。

​		对于每棵树的构造方法，过程如下：

* 用N表示训练集的个数，M表示变量的数目。
* 用m来表示当在一个节点上做决定时会用到的变量的数量。
* 从N个训练案例中采用可重复取样的方式，取样N次，形成一组训练集，并使用这棵树来对剩余变量预测其类别，并对误差进行计算。
* 对于每个节点，随机选择m个基于词典上的变量。根据这m个变量，计算其最佳的分割方式。
* 对于森林中的每棵树都不用采用剪枝技术，每棵树都能完整生长。

​		森林中任意两棵的相关性与森林中棵树的分类能力是影响随机森林分类效果（误差率）的两个重要因素。任意两棵树之间的相关性越大，错误率越大，每棵树的分类能力越强，整个森林的错误率越低。

## 3. 具体实现

### 3.1 CART决策树实现

#### 3.1.1 结点类TreeNode

> 代码位于`code/tree_node.py`

```python
class TreeNode:
    def __init__(self):
        self.left_node = None
        self.right_node = None
        self.type = None
        self.split_feature = None
        self.split_feature_value = None
        self.category = None

    def predict(self, x):
        tree_node = self
        while tree_node.type != "leaf":
            if tree_node.split_feature_value == x[tree_node.split_feature]:
                tree_node = tree_node.left_node
            else:
                tree_node = tree_node.right_node
        return tree_node.category
```

​		`TreeNode`类是CART决策树的节点类，具有以下属性：

* `left_node`：该结点的左子结点。
* `right_node`：该结点的右子结点。
* `type`：该结点的类型。“internal”表示内部结点，“leaf”表示该结点为叶子结点。
* `split_feature`：当该结点为内部结点时，表示该结点的划分特征。
* `split_feature_value`：当该结点为内部结点时，表示该结点的划分特征的取值。如果样本的`split_feature`值等于`split_feature_value`，则该样本属于该结点的左分支；否则该样本属于结点的右分支。
* `category`：当该结点为叶子结点时，表示该结点所属的类别`label`。

​		同时，`TreeNode`类还有`predict`方法，只要将一个样本传到一个内部结点的`predict`方法，就可以从上往下对样本进行分类。如果该样本的`split_feature`值等于`split_feature_value`，就将该样本传到当前结点的左子结点的`predict`方法；否则，就传到当前结点的右子结点的`predict`方法。以此类推，直到当前结点变成叶子结点，则当前结点的`category`就是样本所属的类别`label`，样本分类成功。

​		在大多数情况下，都是将样本传给一个决策树的根结点，最终根结点`predict`得到的结果就是样本的分类结果。

#### 3.1.2 CART决策树类DecisionTreeCART

> 代码位于`code/decision_tree.py`

```python
class DecisionTreeCART:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_split=1e-7, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.root_node = TreeNode()

    def _tree_generate(self, tree_node, train_data, depth):
        # 判断是否继续分裂
        if train_data.shape[0] < self.min_samples_split \
                or train_data.shape[1] <= 2 \
                or self.max_depth is not None and depth == self.max_depth \
                or compute_gini(train_data['label']) < self.min_impurity_split:
            tree_node.type = 'leaf'
            tree_node.category = train_data['label'].value_counts(ascending=False).keys()[0]
            return
        # 统计所有feature及其可能的取值
        feature2values = dict()
        for i in range(len(train_data.columns)):
            feature = train_data.columns[i]
            if feature != 'index' and feature != 'label' and pd.unique(train_data[feature]).shape[0] > 1:
                feature2values[feature] = list(train_data[feature].value_counts().keys())
        # 根据max_features选择决策树划分时选取的特征
        chosen_features = list(feature2values.keys())
        chosen_features_num = len(chosen_features)
        if self.max_features == "log2":
            chosen_features_num = np.log2(len(chosen_features))
        elif self.max_features == "sqrt":
            chosen_features_num = np.sqrt(len(chosen_features))
        elif 0 < self.max_features < 1:
            chosen_features_num = len(chosen_features) * self.max_features
        chosen_features = np.random.choice(chosen_features, size=np.ceil(chosen_features_num).astype(int))
        # 枚举选取最优划分特征和特征值
        min_gini_index = 1
        split_feature = None
        split_feature_value = 0
        for feature in chosen_features:
            for value in feature2values[feature]:
                dv = train_data[train_data[feature] == value]
                dv_nums = dv.shape[0]
                gini_dv = compute_gini(dv['label'])
                dv_not = train_data[train_data[feature] != value]
                dv_not_nums = dv_not.shape[0]
                gini_dv_not = compute_gini(dv_not['label'])
                gini_index = (dv_nums * gini_dv + dv_not_nums * gini_dv_not) / (dv_nums + dv_not_nums)
                if gini_index < min_gini_index:
                    min_gini_index = gini_index
                    split_feature = feature
                    split_feature_value = value
        # 划分样本集
        left_data = train_data[train_data[split_feature] == split_feature_value].drop([split_feature], axis=1)
        right_data = train_data[train_data[split_feature] != split_feature_value]
        if pd.unique(right_data[split_feature]).shape[0] <= 1:
            right_data = right_data.drop([split_feature], axis=1)
        # 设置当前结点的属性
        tree_node.type = "internal"
        tree_node.split_feature = split_feature
        tree_node.split_feature_value = split_feature_value
        tree_node.left_node = TreeNode()
        tree_node.right_node = TreeNode()
        # 递归构建左右子结点
        self._tree_generate(tree_node=tree_node.left_node, train_data=left_data, depth=depth+1)
        self._tree_generate(tree_node=tree_node.right_node, train_data=right_data, depth=depth+1)

    def fit(self, train_data):
        self._tree_generate(tree_node=self.root_node, train_data=train_data, depth=0)

    def predict(self, x_data):
        y_data = np.zeros((x_data.shape[0], 1))
        for i in range(x_data.shape[0]):
            y_data[i] = self.root_node.predict(x_data.iloc[i])
        return y_data
```

​			**`__init__`**方法初始化了CART决策树的相关参数（已在“1.2 参数设置”中做了相关阐述）和`TreeNode`类的树根结点。



​		**`_tree_generate`**内部方法实现了递归构建二叉决策树的过程，参数`tree_node`表示当前正在构建的结点，`train_data`表示待当前结点划分的样集，`depth`表示当前结点的深度。步骤如下：

* *判断是否继续分裂*

  ​        递归要设定边界条件，在本算法中，边界条件就是判断当前结点是否需要继续划分，根据`CART`决策树的特性和参数设定，有以下不再继续分裂的条件：

  * 结点的样本数小于`min_sample_split`；
  * 样本没有更多特征；
  * 树的深度达到决策树的`max_depth`；
  * 结点样本集的$Gini$系数小于`min_impurity_split`。

  ​        当满足以上条件时，当前结点就是叶子结点，该叶子结点的`category`就是当前样本集中`label`的众数。

  ​       其中，出现了$Gini$系数的计算，可通过静态函数`compute_gini`实现，代码如下。`value_counts`可以统计该列中出现的值的计数占比。

  ```python
  def compute_gini(y):
      gini = 1
      for cate in y.value_counts(1):
          gini -= cate * cate
      return gini
  ```

* *统计所有feature及其可能取值*

  ​        将当前样本集的所有feature和可能取值记录在名为`feature2values`的字典中，便于最优划分。需要注意的是，如果当前样本集的一个feature只有一个取值，不需要统计，因为它不可能作为划分样本集的特征。

* *根据max_features选择决策树划分时选取的特征*

  ​        由于设定了参数`max_features`，因此首先要根据`max_feature`的取值确定划分该结点时考虑的特征数`chosen_features_num`，并从所有待选择的特征`chosen_features`中随机选取`chosen_features_num`个特征，作为当前结点待选择的特征`chosen_features`。

* *选取最优划分特征和特征值*

  ​        初始化最小的$Gini$系数`min_gini_index`为1，就开始枚举所有待选择的特征和可能的特征值。

  ​        假设当前样本集根据特征`feature`在某一取值`value`上进行分割，`dv`表示特征`feature`等于`value`的样本集，`dv_nums`表示样本集`dv`的样本数量，`gini_dv`表示样本集`dv`的$Gini$系数；`dv_not`表示特征`feature`不等于`value`的样本集，`dv_not_nums`表示样本集`dv_not`的样本数量，`gini_dv_not`表示样本集`dv_not`的$Gini$系数。

  ​       那么样本集根据特征`feature`在某一取值`value`上进行分割得到的$Gini$系数为`gini_index = (dv_nums * gini_dv + dv_not_nums * gini_dv_not) / (dv_nums + dv_not_nums)`。`gini_index`越小，表示经过这样分类后，集合的不确定性越小，所以如果`gini_index`小于`min_gini_index`，则更新`min_gini_index`为`gini_index`。全部枚举完毕后，最小的`gini_index`对应的`feature`和`value`就是最优划分的特征和特征值。

* *划分样本集*

  ​       在得到最优划分后，就可以根据它划分样本集了。如果样本的`split_feature`等于`split_feature_value`，就属于当前结点的左分支；否则，就属于当前结点的右分支。此外，由于左分支的`split_feature`已经确定，可以删除该特征；如果右分支的`split_feature`也只有1个取值，也可以删该特征。

* *设置当前结点的属性*

​		       此时，当前结点的属性已经确定，可以对当前结点`tree_node`的相关属性进行赋值。

* *递归构建左右子结点*



​		**`fit`**方法就是给定训练数据集，调用内部方法`_tree_generate`从根结点递归构建二叉决策树的过程。



​		**`predict`**方法就是给点数据集，使用搭建好的CART决策树对数据分类的过程，最终返回分类结果。

### 3.2 随机森林实现

#### 3.2.1 随机森林类RandomForest

>  代码位于`code/random_forest.py`

```python
class RandomForest:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_split=1e-7, max_features=None,
                 n_estimators=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.decision_trees = []

    def fit(self, train_data):
        drop_columns = []
        for i in range(len(train_data.columns)):
            feature = train_data.columns[i]
            if feature != 'index' and feature != 'label' and pd.unique(train_data[feature]).shape[0] == 1:
                drop_columns.append(feature)
        train_data = train_data.drop(drop_columns, axis=1)
        for i in range(self.n_estimators):
            decision_tree = DecisionTreeCART(max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_impurity_split=self.min_impurity_split,
                                             max_features=self.max_features)
            decision_tree.fit(train_data)
            print(f"tree {i} done")
            self.decision_trees.append(decision_tree)

    def predict(self, x_data):
        y_data = np.zeros((x_data.shape[0], 1))
        print("prediction start...")
        for i in range(x_data.shape[0]):
            if i % 1000 == 0:
                print(f"{i} samples done")
            pred_dict = dict()
            for j in range(len(self.decision_trees)):
                pred_rlt = self.decision_trees[j].root_node.predict(x_data.iloc[i])
                if pred_rlt not in pred_dict:
                    pred_dict[pred_rlt] = 0
                pred_dict[pred_rlt] += 1
            y_data[i] = max(pred_dict, key=pred_dict.get)
        print(f"{x_data.shape[0]} samples all done")
        print("prediction finish...")
        return y_data

    def save_model(self):
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        model_file = open(f'../model/{now}_rf_model.txt', 'w')
        model_file.write("{} {} {} {} {}\n".format(self.max_depth if self.max_depth is not None else "None",
                                                   self.min_samples_split,
                                                   self.min_impurity_split,
                                                   self.max_features if self.max_features is not None else "None",
                                                   self.n_estimators))
        for i in range(len(self.decision_trees)):
            model_file.write(f"tree {i}:\n")
            node_list = [self.decision_trees[i].root_node]
            node2num = {node_list[0]: 0}
            num = 1
            while len(node_list) != 0:
                cur_node = node_list.pop(0)
                if cur_node.left_node is not None:
                    node2num[cur_node.left_node] = num
                    num += 1
                    node_list.append(cur_node.left_node)
                if cur_node.right_node is not None:
                    node2num[cur_node.right_node] = num
                    num += 1
                    node_list.append(cur_node.right_node)
                model_file.write('{} {} {} {} {} {} {}\n'.format(node2num[cur_node], cur_node.split_feature,
                                                                 cur_node.split_feature_value,
                                                                 cur_node.type,
                                                                 node2num[cur_node.left_node] if cur_node.left_node is not None else "None",
                                                                 node2num[cur_node.right_node] if cur_node.right_node is not None else "None",
                                                                 cur_node.category))
            model_file.write(f"end tree {i}\n")
        model_file.close()

    def load_model(self, model_file_name):
        model_file = open(config["model_file"], 'r')
        lines = model_file.readlines()
        for i in range(self.n_estimators):
            self.decision_trees.append(DecisionTreeCART(max_depth=self.max_depth,
                                                        min_samples_split=self.min_samples_split,
                                                        min_impurity_split=self.min_impurity_split,
                                                        max_features=self.max_features))
        line_no = 1
        tree_cnt = 0
        for tree in self.decision_trees:
            num2node = {0: tree.root_node}
            if lines[line_no][0] == "t":
                line_no += 1
            while line_no < len(lines) and lines[line_no][0] != "e":
                node_info = lines[line_no].replace("\n", '').split(' ')
                num = int(node_info[0])
                cur_node = num2node[num]
                cur_node.split_feature = node_info[1] if node_info[1] != "None" else None
                cur_node.split_feature_value = int(node_info[2]) if node_info[2] != "None" else None
                cur_node.type = node_info[3]
                if node_info[4] != "None":
                    left_node = TreeNode()
                    cur_node.left_node = left_node
                    num2node[int(node_info[4])] = left_node
                if node_info[5] != "None":
                    right_node = TreeNode()
                    cur_node.right_node = right_node
                    num2node[int(node_info[5])] = right_node
                if node_info[6] != "None":
                    cur_node.category = int(node_info[6])
                line_no += 1
            print(f"building tree {tree_cnt} finish...\n")
            tree_cnt += 1
            line_no += 1
```

​		**`__init__`**方法初始化了随机森林的相关参数（已在“1.2 参数设置”中做了相关阐述）和`DecisionTreeCART`类的决策树列表`decision_trees`。



​		**`fit`**方法的任务是根据训练数据搭建随机森林，即搭建`n_estimators`棵CART决策树，并添加该`RandomForest`类对象的决策树列表中。



​		**`predict`**方法的任务就是对样本进行分类，给定一个样本集，对其中每一个样本，用决策树列表`decision_trees`中的所有CART决策树进行分类，并投票决定该样本所属的`laebel`。



​		**`save_model`**方法的任务是保存训练好的模型，将所有决策树的结点信息保存下来，方便下次分类时，直接加载模型。



​		**`load_model`**方法的任务就是加载模型，搭建随机森林用于样本分类。

### 3.3 主函数实现

```python
if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    if config["is_train"]:
        x_train = pd.read_csv("../data/train/x_train.csv")
        y_train = pd.read_csv("../data/train/y_train.csv")
        train_dataset = pd.merge(x_train, y_train)
        train_data, eval_data = train_test_split(train_dataset, test_size=0.2)
        random_forest = RandomForest(max_depth=config["max_depth"],
                                     min_samples_split=config["min_samples_split"],
                                     min_impurity_split=config["min_impurity_split"],
                                     max_features=config["max_features"],
                                     n_estimators=config["n_estimators"])
        random_forest.fit(train_data)
        random_forest.save_model()
        pred_label = random_forest.predict(eval_data)
        print("ac: ", accuracy_score(eval_data['label'], pred_label))
        print("precision: ", precision_score(eval_data['label'], pred_label))
        print("recall: ", recall_score(eval_data['label'], pred_label))
        print("f1_score: ", f1_score(eval_data['label'], pred_label))
    else:
        x_data = pd.read_csv("../data/train/x_train.csv")
        y_data = pd.read_csv("../data/train/y_train.csv")
        test_data = pd.merge(x_data, y_data)
        model_file = open("../model/{}".format(config["model_file"]), 'r')
        lines = model_file.readlines()
        params = lines[0].split(' ')
        max_depth = int(params[0]) if params[0] is not None else None
        min_samples_split = int(params[1])
        min_impurity_split = float(params[2])
        max_features = params[3] if params[3] == "None" or params[3] == "log2" or params[3] == "sqrt" else float(params[3])
        n_estimators = int(params[4])
        random_forest = RandomForest(max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_impurity_split=min_impurity_split,
                                     max_features=max_features, n_estimators=n_estimators)
        random_forest.load_model(config["model_file"])
        pred_label = random_forest.predict(test_data)
        print("ac: ", accuracy_score(test_data['label'], pred_label))
        print("precision: ", precision_score(test_data['label'], pred_label))
        print("recall: ", recall_score(test_data['label'], pred_label))
        print("f1_score: ", f1_score(test_data['label'], pred_label))
```

​		如果需要训练模型的话，就读取数据，并使用`sklearn`包中的`train_test_split`划分训练集和验证集。将训练集传入模型进行训练，并用训练好的模型去对验证集进行分类，并使用`sklearn`包中的`f1_score`等函数评价模型的预测性能。

​		如果不需要训练模型，就直接加载模型，用训练好的模型直接对测试集进行分类，并使用`sklearn`包的`f1_score`等函数评价模型的预测性能。

## 4. 参考资料

[1] [机器学习算法之决策树分类](https://www.biaodianfu.com/decision-tree.html)