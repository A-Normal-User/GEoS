## 语言
*   [English](README.md)
*   [中文](README_CN.md)

## 简介
*   本代码库包含了GEoS构建训练集和进行训练的全部代码，可完整地用于模型的训练。
*   关于测试集，考虑到其中包含了部分商业数据，无法公开提供。因此仅提供了各个模型的计算结果。
*   此外，我们也将拟合得到的PC-SAFT参数提供给大家，同时给出了其拟合状态等信息，包括拟合MSE、拟合时间等，方便大家测试参考。

## 文件信息
*   `Feature`文件夹中包含了特征选择的结果和确定最终特征的代码。
    *  `RHO-Boruta.csv`包含了特征选择的结果。
    *  `EM.ipynb`包含了特征选择使用的EM算法代码。
*   `Database`文件夹中包含了构建训练集的代码。
    *  `features.xlsx`包含特征信息。
    *  `feature_statistics.xlsx`包含训练特征的平均值和标准差信息。
    *  `build.ipynb`是构建训练集的代码。
    *  `Database.zip`是在我们训练集GEoS使用的训练集，由于文件较大，无法直接上传，解压后包含了`database_new_max.pt`和`database_new_max_val.pt`文件。不过，训练集的构建代码我们也已经完整提供，也可根据`build.ipynb`自行构建训练集。
    *  运行`build.ipynb`将会得到`database_new_max.pt`和`database_new_max_val.pt`文件。其中`database_new_max.pt`是最终的训练集，`database_new_max_val.pt`是最终的验证集（实际训练中抽取了其一半的数据放入了训练集中）。
*   `Test`文件夹中包含了测试集的计算结果。
    *  `pcsaft_regression_results.csv`包含了拟合的PC-SAFT参数。
*   `Training`文件夹中包含了模型训练的代码。
    *  `train.py`是训练代码。
    *  `trainall.sh`是训练所有模型的脚本。
    *  `fine.py`是微调代码。
    *  `fineall.sh`是微调所有模型的脚本。
    *  `model.py`是模型定义代码。
    *  `img`文件夹中包含了训练过程中的损失曲线。
    *  `log`文件夹中包含了训练过程中的日志信息。
    *  `model`文件夹中包含了训练得到的模型文件。

## 运行指南
1. 下载代码库
   ```bash
   git clone https://github.com/A-Normal-User/GEoS.git
   cd GEoS
   ```

2. 安装依赖文件
   ```bash
   pip3 install -r requirements.txt
   ```

3. 查看特征筛选结果
   ```bash
   cd Feature
   jupyter notebook EM.ipynb
   cd ..
   ```

4. 构建训练集
   ```bash
   cd Database
   jupyter notebook build.ipynb
   cd ..
   ```

5. 进行预训练模型的构建
    ```bash
    cd Training
    bash trainall.sh
    cd ..
    ```

6. 完成预训练模型构建后可进行微调模型的构建
    ```bash
    cd Training
    bash fineall.sh
    cd ..
    ```

### 相关提示
*   `build.ipynb`与论文中训练集构建过程的代码一致，然而可能因为硬件运行环境的差异导致构建的训练集文件存在极其微小的差异，在此我们给出`.pt`文件的哈希值供参考。
    *   `database_new_max.pt`的MD5哈希值为`dec7b213c5413ee6e8e703adfe282568`
    *   `database_new_max_val.pt`的MD5哈希值为`f6710f57386d59e67091f61eab6c5094`

*   模型训练在`Ubuntu 22.04.4 LTS`操作系统下，使用`Python 3.10.12`和`2.5.1+cu121`进行训练。训练过程中使用了2张`NVIDIA GeForce RTX 3090`显卡，`index`为偶数的在第一张显卡上完成训练，`index`为奇数的在第二张显卡上完成训练。
    *   考虑到不同硬件环境下的训练结果可能存在微小差异，因此我们在`Training/model`文件夹中给出了我们训练得到的模型。

## 文件树
```
|   README.md
|   README_CN.md
|
+---Feature
|       EM.ipynb
|       RHO-Boruta.csv
|
+---Database
|       build.ipynb
|       features.xlsx
|       feature_statistics.xlsx
|
+---Test
|       pcsaft_regression_results.csv
|
\---Training
    |   fine.py
    |   fineall.sh
    |   model.py
    |   train.py
    |   trainall.sh
    |
    +---img
    +---log
    \---model
            model_DGM_fine_1131796.pth
            model_DGM_fine_1211200801.pth
            model_DGM_fine_157564161.pth
            model_DGM_fine_18094361.pth
            model_DGM_fine_218105633.pth
            model_DGM_fine_26317076.pth
            model_DGM_fine_26318864.pth
            model_DGM_fine_448585456.pth
            model_DGM_fine_539035672.pth
            model_DGM_fine_609641183.pth
            model_max_1131796.pth
            model_max_1211200801.pth
            model_max_157564161.pth
            model_max_18094361.pth
            model_max_218105633.pth
            model_max_26317076.pth
            model_max_26318864.pth
            model_max_448585456.pth
            model_max_539035672.pth
            model_max_609641183.pth
```