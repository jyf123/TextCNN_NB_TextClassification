# 中文文本分类系统

##文件结构
    named_entity_recognition                        
    |____ data                      #数据集
         |____ class.txt            #类别
         |____ train.txt            #训练集
         |____ dev.txt              #验证集
         |____ test.txt             #测试集
         |____ scu_stopwords.txt    #四川大学机器智能实验室停用词表
         |____ vocab.pkl            #根据训练集生成的字表
    |____ save_model                #保存的训练好的TextCNN模型
    |____ Nbayes.py                 #训练并评估朴素贝叶斯模型 
    |____ utils.py                  #TextCNN模型数据集处理的函数
    |____ TextCNN.py                #TextCNN模型的基本构建
    |____ train_eval.py             #训练、测试、评估TextCNN模型的函数
    |____ cnn_main.py               #训练并评估TextCNN模型
    

    
##环境    
torch==1.8.1+cu102

numpy==1.20.2

jieba==0.42.1

pandas==1.2.4

tqdm==4.61.0

scikit_learn==0.24.2

       
## 数据集
使用了[huwenxing](https://github.com/649453932) 在[
Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) 项目中的数据集。

其中包含[THUCNews](http://thuctc.thunlp.org/) 数据集中的20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

|  数据集  |  数据量  |
|  ----  | ------ | 
| 训练集  | 180000 | 
| 验证集  | 10000 |
| 测试集 | 10000 | 

## 快速开始
```
#安装依赖项：
pip install -r requirements.txt

#训练并评估朴素贝叶斯模型：
python Nbayes.py

#训练并评估TextCNN模型:
python cnn_main.py

```


## 运行结果

朴素贝叶斯模型和TextCNN模型的性能结果如下表：

|      | 朴素贝叶斯    | TextCNN   | 
| ---- | ------ | ------ |     
| 精确率  | 86.67% | 90.32% | 
| 召回率  | 86.56% | 90.28% | 
| F1值 | 86.57% | 90.29% | 
| 准确率 | 86.56% |90.28% | 




















