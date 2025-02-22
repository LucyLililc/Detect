from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
from torch import manual_seed, cuda, device, cat, Tensor, load, clamp, normal, norm, where, nn, from_numpy, zeros, topk
import matplotlib.pyplot as plt
import warnings
import copy
import torch
from sklearn.metrics import roc_curve, auc
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import scale, RobustScaler, PowerTransformer
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, pairwise_distances
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 该函数用来加载MNIST、CIFAR10数据集
def get_data(dataset, batch_size=256, train_transform=transforms.Compose([transforms.ToTensor()]),
             test_transform=transforms.Compose([transforms.ToTensor()]), is_dataloader=True):
    """
    :param dataset: 加载的数据集名称
    :param batch_size: 批量大小
    :param train_transform: 对训练集数据所做的预处理，默认为转换成Tensor张量。
    :param test_transform: 对测试集数据所做的预处理，默认为转换成Tensor张量。
    :param is_dataloader: 用来控制返回的数据集是Dataset类型的还是Dataloader类型的
    （这样做是因为有些地方会用Subset来划分子集，而Subset需要传入的数据集参数需要是Dataset类型的，要不使用Subset划分好的子集或报错：
    TypeError: 'DataLoader' object is not subscriptable）
    :return: 训练集与测试集
    """
    # 仅支持MNIST和CIFAR10两种数据集
    dataset_classes = {
        'MNIST': datasets.MNIST,
        'CIFAR10': datasets.CIFAR10,
    }
    # 异常处理
    if dataset not in dataset_classes:
        raise ValueError(f"Dataset {dataset} is not supported. Available datasets are: {list(dataset_classes.keys())}")
    # 加载数据集
    train_dataset = dataset_classes[dataset](root=f'../Dataset/{dataset}', train=True,
                                             transform=train_transform, download=True,)
    test_dataset = dataset_classes[dataset](root=f'../Dataset/{dataset}', train=False,
                                            transform=test_transform, download=True,)
    # 分批次
    if is_dataloader:
        train_dataset = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,)
        test_dataset = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,)
    # 返回训练集与测试集
    return train_dataset, test_dataset


# 保持实验的一致性
def seed_everything(seed=25):
    """
    :param seed: 随机种子默认为25
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)


# 设置训练时使用的设备
def get_device():
    """
    :return: 支持的训练设备，gpu/cpu
    """
    use_device = device("cuda" if cuda.is_available() else "cpu")
    return use_device


# 该函数用来加载模型
def get_model(model_name):
    """
    :param model_name: 需要加载的模型名称
    :return: 模型
    """
    # 支持加载LeNet5模型架构
    model_classes = {
        'LeNet5': 'LeNet5_MNIST',
        'ModelCifar10': 'ModelCifar10_CIFAR10',
        'ResNet18': 'ResNet18_CIFAR10',
        'ResNet34': 'ResNet34_CIFAR10'
    }
    # 异常处理
    if model_name not in model_classes:
        raise ValueError(f"model {model_name} is not supported. Available models are {list(model_classes.keys())}")
    # 加载对应模型并返回
    model = load(f"../Pretrained_Model/{model_classes[model_name]}.pt")
    return model


# 该类用来装载自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        """
        Args:
            data (torch.Tensor): 数据张量
            label (torch.Tensor): 标签张量
            transform (callable, optional): 应用于每个样本的变换
        """
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# 可视化数据集前十六张图片（一定要是DataLoader加载完的数据集）
def show(dataset, dataset_name):
    """
    :param dataset: 想要可视化的数据集
    :param dataset_name: 可视化数据集名称，方便展示图像
    :return:
    """
    # 显示前十六张图
    num_images_to_show = 16
    # 创建4x4的子图
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for images, labels in dataset:
        for i in range(num_images_to_show):
            if dataset_name == "MNIST":
                # 移除多余的维度,使张量变为28x28
                img = images[i].squeeze()
                label = labels[i]
                axes[i].imshow(img, cmap='gray')
                # axes[i].set_title(f'Label: {label}')
                # 隐藏坐标轴
                axes[i].axis('off')
            elif dataset_name == "CIFAR10":
                # imshow 期望的图像数据形状是(height, width, channels)
                img = images[i].permute(1, 2, 0)
                label = labels[i]
                axes[i].imshow(img)
                # axes[i].set_title(f'Label: {label}')
                # 隐藏坐标轴
                axes[i].axis('off')
        break

    plt.tight_layout()
    plt.show()


# 计算数据集在特定模型上的准确率（一定要是DataLoader加载完的数据集）
def accuracy(model, dataset, display=True):
    """
    :param model: 预训练模型
    :param dataset: 需要计算预测准确率的数据集
    :param display: 默认为True，显示准确率结果
    :return: 预测的标签
    """
    # 设置训练设备
    use_device = get_device()
    # 记录总数与预测正确个数
    all_sum = 0
    cor_sum = 0
    # 记录所有预测标签值
    pre_labels = Tensor().to(use_device)

    # 计算准确率
    for images, labels in dataset:
        images = images.to(use_device)
        labels = labels.to(use_device)
        results = model(images)

        all_sum += len(labels)
        cor_sum += (results.argmax(axis=1) == labels).sum()
        pre_labels = cat((pre_labels, results.argmax(axis=1)))
    if display:
        print(f"预测准确率为：{round(float(cor_sum)/all_sum*100, 2)}%")
    # 将该tensor转到cpu上并将元素转为整型
    return pre_labels.to("cpu").int()


def accuracy_2(model, dataset, display=True):
    """
    :param model: 预训练模型
    :param dataset: 需要计算预测准确率的数据集
    :param display: 默认为True，显示准确率结果
    :return: 预测的标签
    """
    # 设置训练设备
    use_device = get_device()
    # 记录总数与预测正确个数
    all_sum = 0
    cor_sum = 0
    # 记录所有预测标签值
    pre_labels = Tensor().to(use_device)

    # 计算准确率
    for images, labels in dataset:
        images = images.to(use_device)
        labels = labels.to(use_device)
        results = model(images)

        all_sum += len(labels)
        cor_sum += (results.argmax(axis=1) == labels).sum()
        pre_labels = cat((pre_labels, results.argmax(axis=1)))
    if display:
        print(f"预测准确率为：{round(float(cor_sum)/all_sum*100, 2)}%")
    # 将该tensor转到cpu上并将元素转为整型
    return round(float(cor_sum)/all_sum*100, 2)


# Define a new class that inherits from the original ResNet18
class ResNet18WithoutFC(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithoutFC, self).__init__()
        # Copy all layers except the last fully connected layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        # self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = flatten(x, 1)
        return x


class ResNet18WithBn1(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithBn1, self).__init__()
        # 只保留前面的卷积层和基本块，移除 layer4 中的第二个 BasicBlock 中的 BN、ReLU 和 Conv2
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            # 保留 layer4 中第一个 BasicBlock
            original_model.layer4[0],
            original_model.layer4[1].conv1,
            original_model.layer4[1].bn1,
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet18WithBn2(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithBn2, self).__init__()
        # 去掉 avgpool 和 fc 层
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        # 前向传播时，只经过卷积层和基本块，去掉平均池化和全连接层
        x = self.features(x)
        return x


class ResNet18WithConv1(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithConv1, self).__init__()
        # 只保留前面的卷积层和基本块，移除 layer4 中的第二个 BasicBlock 中的 BN、ReLU 和 Conv2
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            # 保留 layer4 中第一个 BasicBlock
            original_model.layer4[0],
            original_model.layer4[1].conv1,
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet18WithConv2(nn.Module):
    def __init__(self, original_model):
        super(ResNet18WithConv2, self).__init__()
        # 只保留前面的卷积层和基本块，移除 layer4 中的第二个 BasicBlock 中的 BN、ReLU 和 Conv2
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            # 保留 layer4 中第一个 BasicBlock
            original_model.layer4[0],
            original_model.layer4[1].conv1,
            original_model.layer4[1].bn1,
            original_model.layer4[1].relu,
            original_model.layer4[1].conv2,
        )

    def forward(self, x):
        x = self.features(x)
        return x


# 高斯噪声尺度大小由对抗样本的平均扰动大小所决定
STDEVS = {
    'CIFAR10': {'FGSM': 0.021, 'BIM': 0.021, 'PGD': 0.021, 'CW': 0.021, "Deepfool": 0.021, "MIFGSM": 0.021,
                'APGD': 0.021, 'UPGD': 0.021, "PIFGSM": 0.021, "TIFGSM": 0.021, "TPGD": 0.021, "Auto": 0.021},
}


# 得到噪声样本
def get_noisy_samples(test_dataset, dataset, attack, batch_size=256):
    """
    :param test_dataset: 默认是在测试集上构造噪声样本
    :param dataset: 数据集名称
    :param attack: 攻击方法名称
    :param batch_size: 噪声数据集批量大小
    :return:
    """
    seed_everything()
    warnings.warn("使用预设的高斯噪声尺度大小来构造噪声样本，如果改变了构造对抗样本时的扰动/扰动轮次参数，需要更新预设的高斯噪声尺度大小。",
                  UserWarning)
    # 记录正常输入与标签值
    normal_inputs = Tensor()
    true_labels = Tensor()
    for images, labels in test_dataset:
        normal_inputs = cat((normal_inputs, images), dim=0)
        true_labels = cat((true_labels, labels))
    true_labels = true_labels.int()
    # 添加高斯噪声
    noisy_inputs = clamp(normal_inputs + normal(mean=0, std=STDEVS[dataset][attack], size=normal_inputs.shape),
                         min=0, max=1)
    # 构造噪声数据集
    noisy_dataset = CustomDataset(noisy_inputs, true_labels)
    noisy_dataset = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # show(noisy_dataset, dataset)
    # show(test_dataset, dataset)

    # 返回噪声数据集
    return noisy_dataset


# 计算L2损失大小
def l2_diff(dataset_1, dataset_2):
    """
    :param dataset_1: 数据集1一般为正常样本
    :param dataset_2: 数据集2一般为对抗样本或噪声样本
    :return: 数据集1与数据集2的L2损失
    """
    inputs_1 = Tensor()
    inputs_2 = Tensor()
    for (images_1, _), (images_2, _) in zip(dataset_1, dataset_2):
        inputs_1 = cat((inputs_1, images_1), dim=0)
        inputs_2 = cat((inputs_2, images_2), dim=0)

    l2_loss = norm(inputs_1.reshape(len(inputs_1), -1) - inputs_2.reshape((len(inputs_2), -1)),
                   p=2, dim=1).mean()
    return l2_loss


# 仅保留那些正常样本的预测标签与真实标签相符合的样本
def pre_equal_true(model, test_dataset, noisy_dataset, adv_dataset):
    """
    :param model: 用来进行预测的模型
    :param test_dataset: 测试集
    :param noisy_dataset: 噪声集
    :param adv_dataset: 对抗样本集
    :return:
    """
    use_device = get_device()
    # 分别记录对应的数据集的输入
    test_inputs = Tensor().to(use_device)
    noisy_inputs = Tensor()
    adv_inputs = Tensor()
    # 真实标签
    true_labels = Tensor().to(use_device)
    # 正常样本的预测标签
    test_labels = Tensor().to(use_device)

    for (test_images, labels), (noisy_images, _), (adv_images, _) in zip(test_dataset, noisy_dataset, adv_dataset):
        test_images = test_images.to(use_device)
        labels = labels.to(use_device)

        results = model(test_images)

        true_labels = cat((true_labels, labels))
        test_labels = cat((test_labels, results.argmax(axis=1)))

        test_inputs = cat((test_inputs, test_images), dim=0)
        noisy_inputs = cat((noisy_inputs, noisy_images), dim=0)
        adv_inputs = cat((adv_inputs, adv_images), dim=0)

    test_inputs = test_inputs.to("cpu")
    true_labels = true_labels.to("cpu")
    test_labels = test_labels.to("cpu")

    # 记录预测正确的样本下标
    index_correct = where(true_labels == test_labels)[0]
    # 保留预测正确的样本的标签值
    labels_correct = true_labels[index_correct]

    # 只保留对应下标的样本
    test_inputs = test_inputs[index_correct]
    noisy_inputs = noisy_inputs[index_correct]
    adv_inputs = adv_inputs[index_correct]

    return [test_inputs, noisy_inputs, adv_inputs, labels_correct]


# 计算深度特征表示值（logits层）
def get_deep_representations(model, dataset, is_numpy=False):
    """
    :param model:
    :param dataset: Dataloader加载好的数据集
    :param is_numpy: 控制返回结果是numpy数组还是Tensor张量
    :return: logits层数据
    """
    use_device = get_device()
    outputs = Tensor()
    for images, _ in dataset:
        images = images.to(use_device)
        results = model(images)
        results = results.reshape(results.shape[0], -1)
        # 去掉梯度并放到cpu上，要不显存不够
        results = results.detach().to("cpu")
        outputs = cat((outputs, results), dim=0)
    if is_numpy:
        outputs = outputs.numpy()
    return outputs


# 将输入转换为DataLoader加载好的数据
def inputs_to_dataset(inputs, labels, batch_size, shuffle=False, drop_last=False, transform=None):
    dataset = CustomDataset(inputs, labels, transform)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataset


# 具体的核密度计算函数
def score_point(tup):
    """
    :param tup: 函数score_samples中传入的元组
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


# 对测试集，噪声集，对抗集样本进行核密度估计
def score_samples(kdes, features, preds):
    """
    :param kdes:训练好的核密度估计器
    :param features: 深层特征表示
    :param preds: 预测标签
    :return:
    """
    results = np.asarray(
        list(
            map(
                score_point,
                [(x, kdes[i]) for x, i in zip(features, preds)]
            )
        )
    )

    return results


# 将核密度估计值，不确定性值，语义空间距离标准化（标准化的意义是不让某项的值过大从而影响分类器的训练效果）
def normalize(test_features, noisy_features, adv_features):
    """
    TODO
    :param test_features: 测试集的检测特征
    :param noisy_features: 噪声集的检测特征
    :param adv_features: 对抗集的检测特征
    :return: 标准化后的结果列表
    """
    n_samples = len(test_features)
    # 标准化
    total = scale(np.concatenate((test_features, noisy_features, adv_features)))

    # 以列表形式返回结果
    return [total[:n_samples], total[n_samples:2 * n_samples], total[2 * n_samples:]]


def robust(test_features, noisy_features, adv_features):
    n_samples = len(test_features)

    robust_scale = RobustScaler()
    total = robust_scale.fit_transform(np.concatenate((test_features, noisy_features, adv_features)).reshape(-1, 1))\
        .reshape(-1)

    # 以列表形式返回结果
    return [total[:n_samples], total[n_samples:2 * n_samples], total[2 * n_samples:]]


def power(test_features, noisy_features, adv_features):
    n_samples = len(test_features)

    power_scale = PowerTransformer()
    total = power_scale.fit_transform(np.concatenate((test_features, noisy_features, adv_features)).reshape(-1, 1))\
        .reshape(-1)

    # 以列表形式返回结果
    return [total[:n_samples], total[n_samples:2 * n_samples], total[2 * n_samples:]]


# 整理检测指标的数据，便于训练检测器
def build_detect_data(densities_results=None, uncertain_results=None, distance_results=None, squeezing_results=None,
                      md_results=None, iso_results=None, cs_results=None, gmm_results=None, z_results=None,
                      lof_results=None, ma_results=None):
    """
    :param densities_results: 核密度估计结果
    :param uncertain_results: 贝叶斯不确定性结果
    :param distance_results: 语义空间距离结果
    :param squeezing_results: 位深压缩结果
    :param md_results:马氏距离结果
    :param iso_results:孤立森林值
    :param cs_results:余弦相似度值
    :param gmm_results:
    :param z_results
    :param lof_results
    :param ma_results
    :return:
    """
    # 将所有检测指标放进一个列表中
    detect_index = [densities_results, uncertain_results, distance_results, squeezing_results, md_results, iso_results,
                    cs_results, gmm_results, z_results, lof_results, ma_results]
    # 记录负类的指标值（测试、噪声样本集所产生的检测指标）
    values_neg = np.array([])
    # 记录正类的指标值（对抗样本集所产生的检测指标）
    values_pos = np.array([])
    # 遍历列表中的检测指标，完成负类与正类值的填充
    for value_index in detect_index:
        # 若当前检测指标存在（即build_detect_data函数被赋值相应参数）
        if value_index:
            # 负类（即测试、噪声样本集所产生的检测指标，放在列表中下标为0、1的位置）
            if values_neg.any():
                values_neg = np.concatenate(
                    (values_neg, np.concatenate((value_index[0], value_index[1])).reshape(1, -1)), axis=0
                )
            else:
                values_neg = np.concatenate((value_index[0], value_index[1])).reshape(1, -1)
            # 正类（即对抗样本集所产生的检测指标，放在列表中下标为2的位置）
            if values_pos.any():
                values_pos = np.concatenate(
                    (values_pos, value_index[2].reshape(1, -1)), axis=0
                )
            else:
                values_pos = value_index[2].reshape(1, -1)
    # 行列互换
    values_neg = values_neg.transpose([1, 0])
    values_pos = values_pos.transpose([1, 0])

    # 合并正负类数据
    values = np.concatenate((values_neg, values_pos))
    # 前三分之二为负类（测试集与噪声集的检测指标），所以标签值为0；后面的三分之一为对抗集的检测指标，为正类，所以标签值为1
    labels = np.concatenate((np.zeros([values_neg.shape[0]]), np.ones([values_pos.shape[0]])))

    # 返回整理好的检测指标数据
    return values, labels


# 根据整理好的检测数据来训练检测器
def train_detector(values, labels, model_name, attack, dataset_name, is_saved=True):
    print("训练检测器>>>>>")
    # 创建检测器
    classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=23,
            # 缓解过拟合
            reg_alpha=10,  # L1 正则化，增加稀疏性
            reg_lambda=10,  # L2 正则化，防止权重过大
            min_child_weight=5,  # 增大叶子节点的最小样本权重，抑制复杂分割
            subsample=0.8,  # 每棵树使用 80% 的样本
            colsample_bytree=0.8,  # 每棵树使用 80% 的特征
            gamma=1,
    )
    # 训练检测器
    classifier.fit(values, labels)
    if is_saved:
        classifier.save_model(f"../Detector/Detector_{model_name}_{attack}_{dataset_name}.json")
        print("检测器保存成功")
    # 返回训练好的检测器
    return classifier


# 根据训练好的检测器来进行预测（训练数据，测试数据的预测准确率，ROC曲线图的展示）
def detect_effect(classifier, train_values, train_labels, test_values, test_labels, model_name, attack, dataset_name):
    print("展示检测效果>>>>>")
    # 训练集的预测标签
    train_preds = classifier.predict(train_values)
    print(f"训练集的检测准确率：{round(np.sum(np.asarray(train_preds)==train_labels)/len(train_preds)*100, 2)}%")
    # 训练集的预测结果属于类别1的概率（用于绘制ROC曲线图）
    train_probs = classifier.predict_proba(train_values)[:, 1]
    # 测试集的预测标签
    test_preds = classifier.predict(test_values)
    print(f"测试集的检测准确率：{round(np.sum(np.asarray(test_preds)==test_labels)/len(test_preds)*100, 2)}%")
    # 测试集的预测结果属于类别1的概率（用于绘制ROC曲线图）
    test_probs = classifier.predict_proba(test_values)[:, 1]

    # 绘制ROC曲线图以及计算AUC值
    # 训练集与测试集的前2/3是负类（正常与噪声集），后1/3是正类（对抗集），所以这样划分
    train_index = int(len(train_probs)*2/3)
    test_index = int(len(test_probs)*2/3)
    auc_score = compute_roc(
        train_probs[:train_index],
        train_probs[train_index:],
        test_probs[:test_index],
        test_probs[test_index:],
        model_name,
        attack,
        dataset_name
    )
    print(f'Detector ROC-AUC score-train: {auc_score[0]} test:{auc_score[1]}')


# ROC曲线图的绘制
def compute_roc(probs_neg_train, probs_pos_train, probs_neg_test, probs_pos_test, model_name, attack, dataset_name,
                plot=True):
    probs_train = np.concatenate((probs_neg_train, probs_pos_train))
    probs_test = np.concatenate((probs_neg_test, probs_pos_test))
    labels_train = np.concatenate((np.zeros_like(probs_neg_train), np.ones_like(probs_pos_train)))
    labels_test = np.concatenate((np.zeros_like(probs_neg_test), np.ones_like(probs_pos_test)))
    fpr_train, tpr_train, _ = roc_curve(labels_train, probs_train)
    auc_score_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(labels_test, probs_test)
    auc_score_test = auc(fpr_test, tpr_test)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr_train, tpr_train, color='blue',
                 label='train (AUC = %0.4f)' % auc_score_train)
        plt.plot(fpr_test, tpr_test, color='red',
                 label='test (AUC = %0.4f)' % auc_score_test)
        plt.legend(loc='lower right')
        plt.title(f"ROC Curve\n{model_name}_{attack}_{dataset_name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"../Detect_effect/ROC_Curve_{model_name}_{attack}_{dataset_name}.png")
        plt.show()

    return [auc_score_train, auc_score_test]


# 将模型的Dropout层打开
def set_dropout_train(module):
    if isinstance(module, nn.Dropout):
        module.train()


# 获取贝叶斯不确定性
def get_mc_predictions(model, inputs, nb_iter, model_name, batch_size=256):
    """
    :param model: 用于获取蒙特卡洛方差(即贝叶斯不确定性)结果的模型
    :param inputs: 送入模型的输入
    :param nb_iter: 默认采样轮数为50轮
    :param model_name: 使用的模型名称，若为ResNet18,则需要关闭批量归一化层，开启Dropout层(LeNet5和ModelCifar10无批量归一化层)
    :param batch_size: 批量大小默认256
    :return:
    """
    use_device = get_device()
    # 模型为训练状态,dropout保持开启
    if model_name == "ResNet18":
        model.eval()
        model.apply(set_dropout_train)
    else:
        model.train()

    def predict():
        # 创建输入的数据集
        dataset = DataLoader(inputs, batch_size=batch_size, shuffle=False, drop_last=False)
        # 存储模型输出的结果
        output = Tensor()
        for image in dataset:
            image = image.to(use_device)
            results = model(image)
            results = results.detach().to("cpu")
            output = cat((output, results), dim=0)
        # 返回本次调用模型的输出结果(转为numpy数据再返回)
        return output.numpy()

    # 记录所有迭代轮次(nb_iter)所产生的预测结果
    preds_mc = []
    for _ in tqdm(range(nb_iter)):
        preds_mc.append(predict())
    preds_mc = np.asarray(preds_mc)
    # 返回保存的nb_iter轮的预测结果
    return preds_mc


# 按指定比例划分训练数据和测试数据(numpy数组的划分,不是Dataloader型,也不是Tensor型)
def train_test_split(index, test, noisy, adv):
    """
    :param index: int型,用来决定划分的下标
    :param test: 测试数据(numpy.array)
    :param noisy: 噪声数据(numpy.array)
    :param adv: 对抗数据(numpy.array)
    :return:
    """
    # 计算划分的下标
    split_index = int(index*test.shape[0])
    # 划分训练数据与测试数据,并进行了标准化工作
    inputs = normalize(test, noisy, adv)
    train_inputs = [inputs[0][:split_index], inputs[1][:split_index], inputs[2][:split_index], ]
    test_inputs = [inputs[0][split_index:], inputs[1][split_index:], inputs[2][split_index:], ]
    # 返回结果
    return train_inputs, test_inputs


def train_test_split_robust(index, test, noisy, adv):
    """
    :param index: int型,用来决定划分的下标
    :param test: 测试数据(numpy.array)
    :param noisy: 噪声数据(numpy.array)
    :param adv: 对抗数据(numpy.array)
    :return:
    """
    # 计算划分的下标
    split_index = int(index*test.shape[0])
    # 划分训练数据与测试数据,并进行了标准化工作
    train_inputs = robust(test[:split_index], noisy[:split_index], adv[:split_index])
    test_inputs = robust(test[split_index:], noisy[split_index:], adv[split_index:])

    # 返回结果
    return train_inputs, test_inputs


def train_test_split_power(index, test, noisy, adv):
    """
    :param index: int型,用来决定划分的下标
    :param test: 测试数据(numpy.array)
    :param noisy: 噪声数据(numpy.array)
    :param adv: 对抗数据(numpy.array)
    :return:
    """
    # 计算划分的下标
    split_index = int(index*test.shape[0])
    # 划分训练数据与测试数据,并进行了标准化工作
    train_inputs = power(test[:split_index], noisy[:split_index], adv[:split_index])
    test_inputs = power(test[split_index:], noisy[split_index:], adv[split_index:])

    # 返回结果
    return train_inputs, test_inputs


# 计算特征压缩（色深压缩）前后的结果差值
def squeezing_difference(model, inputs, squeezing_inputs, batch_size, p=2):
    """
    :param model: 用于计算结果的模型
    :param inputs: 特征压缩前的数据
    :param squeezing_inputs: 特征压缩后的数据
    :param batch_size:
    :param p: 决定范数格式
    :return:
    """
    # 设置训练设备
    use_device = get_device()
    # 批量加载数据
    inputs_dataset = DataLoader(inputs, batch_size=batch_size, shuffle=False, drop_last=False)
    squeezing_inputs_dataset = DataLoader(squeezing_inputs, batch_size=batch_size, shuffle=False, drop_last=False)
    # 记录结果差值
    differences = Tensor()
    # 计算差值
    for image, squeezing_image in zip(inputs_dataset, squeezing_inputs_dataset):
        image = image.to(use_device)
        squeezing_image = squeezing_image.to(use_device)

        result = model(image)
        squeezing_result = model(squeezing_image)

        diff = norm(result-squeezing_result, p=p, dim=1)
        diff = diff.detach().to("cpu")

        differences = cat((differences, diff))
    # 以numpy数组形式返回差值
    return differences.numpy()


# 定义一个函数来获取随机子集的索引
def get_random_subset_indices(dataset_size, subset_size):
    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)
    return all_indices[:subset_size]


# 随机抽取数据集中的部分数据，并返回组成的新的数据集
def get_sub_dataset(dataset, radio=0.1, batch_size=256):
    """
    :param dataset: 要抽取的数据集
    :param radio: 抽取的比例，默认为0.1
    :param batch_size: 批量大小，默认256
    :return: 新的子数据集
    """
    # 获取数据集中的样本个数
    num_samples = len(dataset.dataset)
    # 计算抽取的子集的大小
    subset_size = int(radio * num_samples)
    # 获取子集对应的随机下标位置
    subset_indices = get_random_subset_indices(num_samples, subset_size)
    # 获取子集
    subset_dataset = Subset(dataset, subset_indices)
    # 分批量
    subset_dataset = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # 返回子集
    return subset_dataset


# 去噪器对指定数据集进行去噪并返回去噪后的数据集（数据集需要为Dataloader形式）
def denoise_dataset(dataset, denoiser, batch_size):
    """
    :param dataset: 需要去噪的数据集
    :param denoiser:  去噪器
    :param batch_size:
    :return:
    """
    use_device = get_device()
    refact_inputs = Tensor()
    all_labels = Tensor()
    for images, labels in dataset:
        images = images.to(use_device)
        refact_images = denoiser(images)
        refact_images = refact_images.detach().to("cpu")
        refact_inputs = cat((refact_inputs, refact_images))
        all_labels = cat((all_labels, labels))
    refact_datasets = CustomDataset(refact_inputs, all_labels)
    refact_datasets = DataLoader(refact_datasets, batch_size=batch_size, shuffle=False, drop_last=False)

    return refact_datasets


ratio = 0.7


def mahalanobis_distance(train_features, test_features, noisy_features, adv_features):
    # 1. 计算均值向量
    mean_vector = np.mean(train_features, axis=0)
    # 2. 计算协方差矩阵
    cov_matrix = np.cov(train_features, rowvar=False)
    # 3. 计算协方差矩阵的逆
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    results = []

    strat_time = time.time()
    for input_features in [test_features, noisy_features, adv_features]:
        output = np.zeros(shape=(input_features.shape[0], ))
        for i in tqdm(range(len(input_features))):
            diff = input_features[i] - mean_vector
            output[i] = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
        results.append(output)
    end_time = time.time()
    print(f"孤立森林需要花费的时间：{end_time-strat_time}")

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_isolation_forest(train_features, test_features, noisy_features, adv_features):
    """
    使用孤立森林检测两个分布的差异。

    参数:
        train_features: np.ndarray, 形状为 (n_samples1, n_features)，训练数据
        input_features: np.ndarray, 形状为 (n_samples2, n_features)，输入数据

    返回:
        scores: np.ndarray, 每个输入样本的异常分数
    """
    # 初始化孤立森林模型
    clf = IsolationForest(n_estimators=100, random_state=42)

    # 使用训练数据训练孤立森林
    clf.fit(train_features)

    results = []

    for input_features in [test_features, noisy_features, adv_features]:
        # 计算输入数据的异常分数（负值，值越小越异常）
        scores = -clf.decision_function(input_features)

        results.append(scores)

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_cosine_similarity(train_features, test_features, noisy_features, adv_features):
    """
    计算每个输入样本与训练数据的余弦相似度。

    参数:
        train_features: np.ndarray, 形状为 (n_samples1, n_features)，训练数据
        input_features: np.ndarray, 形状为 (n_samples2, n_features)，输入数据

    返回:
        output: np.ndarray, 每个输入样本的平均余弦相似度
    """
    results = []

    start = time.time()
    for input_features in [test_features, noisy_features, adv_features]:
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(input_features, train_features)

        # 对每个输入样本，计算其与所有训练样本的平均相似度
        output = similarity_matrix.mean(axis=1)
        results.append(output)
    end = time.time()
    print(f"cos花费时间{end-start}")

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_gmm_log_likelihood(train_features, test_features, noisy_features, adv_features, n_components=5):
    """
    使用高斯混合模型（GMM）计算分布差异。

    参数:
        train_features: np.ndarray, 形状为 (n_samples1, n_features)，训练数据
        input_features: np.ndarray, 形状为 (n_samples2, n_features)，输入数据
        n_components: int, GMM的组件数（默认为5）

    返回:
        scores: np.ndarray, 每个输入样本的异常分数（基于GMM对数似然）
    """
    # 使用高斯混合模型拟合训练数据
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(train_features)

    results = []

    for input_features in [test_features, noisy_features, adv_features]:
        # 使用进度条计算每个输入样本的对数似然
        log_likelihood = np.zeros(input_features.shape[0])  # 初始化对数似然数组

        # 计算每个输入样本的对数似然，使用 tqdm 来显示进度条
        for i in tqdm(range(input_features.shape[0]), desc="Calculating GMM log-likelihood"):
            log_likelihood[i] = gmm.score_samples([input_features[i]])

        # 转换为异常分数（对数似然的负值表示异常）
        scores = -log_likelihood

        results.append(scores)

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_lof_score(train_features, test_features, noisy_features, adv_features):
    # 使用 LOF 计算局部离群因子 (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(train_features)

    results = []

    strat = time.time()
    for input_features in [test_features, noisy_features, adv_features]:
        scores = lof.score_samples(input_features)
        results.append(scores)
    end = time.time()
    print(f"lof花费时间：{end-strat}")

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_z_score(train_features, test_features, noisy_features, adv_features):
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)

    results = []

    start = time.time()

    for input_features in [test_features, noisy_features, adv_features]:
        scores = np.sum((input_features - mean) / std, axis=1)
        results.append(scores)

    end = time.time()
    print(f"z-score花费时间{end-start}")

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


# 计算与训练集相距最近的k个样本的语义空间距离均值
def distance_k(train_features, test_features, noisy_features, adv_features, k=10):
    use_device = get_device()
    # 转为tensor张量,并放置到gpu上
    train_features = from_numpy(train_features).to(use_device)

    results = []

    for input_features in [test_features, noisy_features, adv_features]:
        # 转化为Tensor张量来计算是否能快点，现在这计算一次就要十分钟(转换完后快多了，十几秒就好了)
        output = zeros(size=(input_features.shape[0],))
        epochs = len(input_features)
        input_features = from_numpy(input_features).to(use_device)
        # 计算topk均值
        for i in tqdm(range(epochs)):
            distance = norm(train_features - input_features[i], p=2, dim=1)
            _, index = topk(distance, k=k)
            output[i] = (distance[index]).to("cpu").mean()
        # 转为numpy数组
        output = output.numpy()
        results.append(output)

    # 按照9:1的比例划分训练集和测试集
    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    # 返回
    return train_inputs, test_inputs


def compute_manhattan_distance(train_features, test_features, noisy_features, adv_features, k=10):
    use_device = get_device()
    # 转为tensor张量,并放置到gpu上
    train_features = from_numpy(train_features).to(use_device)

    results = []

    for input_features in [test_features, noisy_features, adv_features]:
        output = zeros(size=(input_features.shape[0],), device=use_device)
        epochs = input_features.shape[0]
        input_features = from_numpy(input_features).to(use_device)
        # 计算每个样本与所有训练样本之间的 manhattan 距离
        for i in tqdm(range(epochs)):
            distance = torch.sum(torch.abs(train_features - input_features[i]), dim=1)
            _, index = topk(distance, k=k)
            output[i] = distance[index].mean()

        output = output.cpu().numpy()
        results.append(output)

    train_inputs, test_inputs = train_test_split(ratio, results[0], results[1], results[2])
    return train_inputs, test_inputs
