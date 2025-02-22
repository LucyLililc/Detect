import os.path
import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Utils.utils import get_data, get_device, get_model, CustomDataset, show, get_noisy_samples, \
    l2_diff, pre_equal_true, get_deep_representations, ResNet18WithoutFC, inputs_to_dataset, score_samples, \
    normalize, build_detect_data, train_detector, detect_effect, get_mc_predictions, train_test_split, distance_k, \
    squeezing_difference, mahalanobis_distance, compute_isolation_forest, compute_cosine_similarity, \
    accuracy, train_test_split_robust, train_test_split_power, compute_gmm_log_likelihood, \
    ResNet18WithBn2, compute_lof_score, compute_z_score, compute_manhattan_distance, ResNet18WithConv1, \
    ResNet18WithBn1, ResNet18WithConv2
from torch import load, round, tensor, nn, Tensor, cat, save, optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import copy
import numpy as np
import warnings
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale, RobustScaler, PowerTransformer, \
    QuantileTransformer
from sklearn.linear_model import LogisticRegression
import torch
from scipy.spatial import distance

device = get_device()


# 检测器的训练数据为：测试集以及测试集上生成的噪声集和对抗集的前百分之九十的数据（因此，去噪器也应该是由前百分之九十的数据训练生成的。
# 最后测试的时候拿后百分之十的数据来检测整个框架效果）
class DetectAdvSamples:
    def __init__(self, model_name, attack, dataset_name, batch_size=256):
        self.model_name = model_name
        self.attack = attack
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        # 加载预训练模型
        self.model = get_model(model_name)
        # 设为评估模式
        self.model.eval()
        # 加载对抗样本数据集
        self.adv_dataset = load(f"../Save_Adv/{attack}_{model_name}_{dataset_name}.pt")
        # 这样命名是为了和下面的self.adv_inputs区分开来
        self.adv_inputs_all = self.adv_dataset["adv_inputs"]
        self.true_labels = self.adv_dataset['labels']
        self.adv_dataset = CustomDataset(self.adv_inputs_all, self.true_labels)
        self.adv_dataset = DataLoader(self.adv_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # 加载正常样本数据集
        self.train_dataset, self.test_dataset = get_data(dataset_name)
        show(self.test_dataset, self.dataset_name)

        # 加载噪声数据集
        self.noisy_dataset = get_noisy_samples(self.test_dataset, self.dataset_name, self.attack, self.batch_size)

        # 准确率与扰动显示n
        # 模型在正常样本上的预测准确率
        print("正常样本：")
        accuracy(self.model, self.test_dataset)
        # 模型在对抗样本上的预测准确率
        print("对抗样本：")
        accuracy(self.model, self.adv_dataset)
        # 模型在噪声样本上的预测准确率
        print("噪声样本：")
        accuracy(self.model, self.noisy_dataset)
        # 计算对抗样本的L2扰动大小
        l2_adv = l2_diff(self.test_dataset, self.adv_dataset)
        print(f"对抗样本的L2扰动大小：{l2_adv}")
        # 计算噪声样本的L2扰动大小
        l2_noisy = l2_diff(self.test_dataset, self.noisy_dataset)
        print(f"噪声样本的L2扰动大小：{l2_noisy}")

        # 均以Tensor形式返回
        # 仅保留那些正常样本的预测标签与真实标签相符合的样本(注意这里仅返回对应的样本，不带标签，不是Dataloader类型)，并返回对应的标签值
        self.test_inputs, self.noisy_inputs, self.adv_inputs, self.labels_correct = \
            pre_equal_true(self.model, self.test_dataset,
                           self.noisy_dataset, self.adv_dataset)

        if not os.path.exists(f"../Save_detect_data/{self.attack}_train_values.npy"):
            # 计算logits层结果
            print("计算logits值>>>>>")
            # 训练集经过模型后输出的logits数据
            if not os.path.exists(f"../Features/logits_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.logits_train_features = get_deep_representations(self.model, self.train_dataset, is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.logits_train_features,
                     f"../Features/logits_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集logits值保存成功")
            else:
                self.logits_train_features = load(
                    f"../Features/logits_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集logits值加载成功")
            # 测试集经过模型后输出的logits数据
            self.logits_test_features = get_deep_representations(self.model, inputs_to_dataset(self.test_inputs,
                                                                                               self.labels_correct,
                                                                                               self.batch_size),
                                                                 is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的logits数据
            self.logits_noisy_features = get_deep_representations(self.model, inputs_to_dataset(self.noisy_inputs,
                                                                                                self.labels_correct,
                                                                                                self.batch_size),
                                                                  is_numpy=True)
            # 对抗集经过模型后输出的logits数据
            self.logits_adv_features = get_deep_representations(self.model, inputs_to_dataset(self.adv_inputs,
                                                                                              self.labels_correct,
                                                                                              self.batch_size),
                                                                is_numpy=True)
            # ************************************************************************************************
            print("计算hidden层特征表示>>>>>")
            # 由于要去除模型的最后若干层，从而得到last_hidden层数据，所以重建一个新模型，避免改变self.model
            self.model_logits = get_model(self.model_name)
            self.model_logits = ResNet18WithoutFC(self.model_logits)
            self.model_logits.eval()
            # 训练集经过模型后输出的last_hidden层数据
            if not os.path.exists(f"../Features/hidden_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.hidden_train_features = get_deep_representations(self.model_logits, self.train_dataset,
                                                                      is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.hidden_train_features,
                     f"../Features/hidden_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集hidden层特征表示保存成功")
            else:
                self.hidden_train_features = load(
                    f"../Features/hidden_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集hidden层特征表示加载成功")
            # 测试集经过模型后输出的last_hidden层数据
            self.hidden_test_features = get_deep_representations(self.model_logits,
                                                                 inputs_to_dataset(self.test_inputs,
                                                                                   self.labels_correct,
                                                                                   self.batch_size),
                                                                 is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的last_hidden层数据
            self.hidden_noisy_features = get_deep_representations(self.model_logits,
                                                                  inputs_to_dataset(self.noisy_inputs,
                                                                                    self.labels_correct,
                                                                                    self.batch_size),
                                                                  is_numpy=True)
            # 对抗集经过模型后输出的last_hidden层数据
            self.hidden_adv_features = get_deep_representations(self.model_logits,
                                                                inputs_to_dataset(self.adv_inputs,
                                                                                  self.labels_correct,
                                                                                  self.batch_size),
                                                                is_numpy=True)
            # ************************************************************************************************
            # 训练集经过模型后输出的bn1层数据
            print("计算bn1层的特征表示>>>>>")
            # 由于要去除模型的最后若干层，从而得到last_hidden层数据，所以重建一个新模型，避免改变self.model
            self.model_bn1 = get_model(self.model_name)
            self.model_bn1 = ResNet18WithBn1(self.model_bn1)
            self.model_bn1.eval()
            # 训练集经过模型后输出的bn1层数据
            if not os.path.exists(f"../Features/bn1_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.bn1_train_features = get_deep_representations(self.model_bn1, self.train_dataset, is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.bn1_train_features,
                     f"../Features/bn1_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集bn1层特征表示保存成功")
            else:
                self.bn1_train_features = load(
                    f"../Features/bn1_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集bn1层特征表示加载成功")
            # 测试集经过模型后输出的last_hidden层数据
            self.bn1_test_features = get_deep_representations(self.model_bn1, inputs_to_dataset(self.test_inputs,
                                                                                                self.labels_correct,
                                                                                                self.batch_size),
                                                              is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的last_hidden层数据
            self.bn1_noisy_features = get_deep_representations(self.model_bn1, inputs_to_dataset(self.noisy_inputs,
                                                                                                 self.labels_correct,
                                                                                                 self.batch_size),
                                                               is_numpy=True)
            # 对抗集经过模型后输出的last_hidden层数据
            self.bn1_adv_features = get_deep_representations(self.model_bn1, inputs_to_dataset(self.adv_inputs,
                                                                                               self.labels_correct,
                                                                                               self.batch_size),
                                                             is_numpy=True)
            # 标准化所有特征数据
            all_features = np.vstack([
                self.bn1_train_features,
                self.bn1_test_features,
                self.bn1_noisy_features,
                self.bn1_adv_features
            ])
            # 标准化
            scaler = StandardScaler()
            # 对所有特征进行标准化
            all_features_scaled = scaler.fit_transform(all_features)
            # 降维：PCA统一对四个特征进行降维
            print("对bn1层特征进行PCA降维")
            n_components = 200
            pca = PCA(n_components=n_components)
            # 先统一降维
            all_features_pca = pca.fit_transform(all_features_scaled)
            # 分离回各自的特征
            self.bn1_train_features = all_features_pca[:len(self.bn1_train_features)]
            self.bn1_test_features = all_features_pca[
                                     len(self.bn1_train_features):len(self.bn1_train_features) + len(
                                         self.bn1_test_features)]
            self.bn1_noisy_features = all_features_pca[len(self.bn1_train_features) + len(self.bn1_test_features):len(
                self.bn1_train_features) + len(self.bn1_test_features) + len(self.bn1_noisy_features)]
            self.bn1_adv_features = all_features_pca[
                                    len(self.bn1_train_features) + len(self.bn1_test_features) + len(
                                        self.bn1_noisy_features):]
            # ************************************************************************************************
            # 训练集经过模型后输出的bn2层数据
            print("计算bn2层的特征表示>>>>>")
            # 由于要去除模型的最后若干层，从而得到last_hidden层数据，所以重建一个新模型，避免改变self.model
            self.model_bn2 = get_model(self.model_name)
            self.model_bn2 = ResNet18WithBn2(self.model_bn2)
            self.model_bn2.eval()
            # 训练集经过模型后输出的bn2层数据
            if not os.path.exists(f"../Features/bn2_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.bn2_train_features = get_deep_representations(self.model_bn2, self.train_dataset, is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.bn2_train_features,
                     f"../Features/bn2_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集bn2层特征表示保存成功")
            else:
                self.bn2_train_features = load(
                    f"../Features/bn2_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集bn2层特征表示加载成功")
            # 测试集经过模型后输出的last_hidden层数据
            self.bn2_test_features = get_deep_representations(self.model_bn2, inputs_to_dataset(self.test_inputs,
                                                                                                self.labels_correct,
                                                                                                self.batch_size),
                                                              is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的last_hidden层数据
            self.bn2_noisy_features = get_deep_representations(self.model_bn2, inputs_to_dataset(self.noisy_inputs,
                                                                                                 self.labels_correct,
                                                                                                 self.batch_size),
                                                               is_numpy=True)
            # 对抗集经过模型后输出的last_hidden层数据
            self.bn2_adv_features = get_deep_representations(self.model_bn2, inputs_to_dataset(self.adv_inputs,
                                                                                               self.labels_correct,
                                                                                               self.batch_size),
                                                             is_numpy=True)
            # 标准化所有特征数据
            all_features = np.vstack([
                self.bn2_train_features,
                self.bn2_test_features,
                self.bn2_noisy_features,
                self.bn2_adv_features
            ])
            # 标准化
            scaler = StandardScaler()
            # 对所有特征进行标准化
            all_features_scaled = scaler.fit_transform(all_features)
            # 降维：PCA统一对四个特征进行降维
            print("对bn2层特征进行PCA降维")
            n_components = 200
            pca = PCA(n_components=n_components)
            # 先统一降维
            all_features_pca = pca.fit_transform(all_features_scaled)
            # 分离回各自的特征
            self.bn2_train_features = all_features_pca[:len(self.bn2_train_features)]
            self.bn2_test_features = all_features_pca[
                                     len(self.bn2_train_features):len(self.bn2_train_features) + len(
                                         self.bn2_test_features)]
            self.bn2_noisy_features = all_features_pca[len(self.bn2_train_features) + len(self.bn2_test_features):len(
                self.bn2_train_features) + len(self.bn2_test_features) + len(self.bn2_noisy_features)]
            self.bn2_adv_features = all_features_pca[
                                    len(self.bn2_train_features) + len(self.bn2_test_features) + len(
                                        self.bn2_noisy_features):]
            # ************************************************************************************************
            print("计算conv1层特征表示")
            # 由于要去除模型的最后若干层，从而得到last_hidden层数据，所以重建一个新模型，避免改变self.model
            self.model_conv1 = get_model(self.model_name)
            self.model_conv1 = ResNet18WithConv1(self.model_conv1)
            self.model_conv1.eval()
            # 训练集经过模型后输出的conv1层数据
            if not os.path.exists(f"../Features/conv1_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.conv1_train_features = get_deep_representations(self.model_conv1, self.train_dataset,
                                                                     is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.conv1_train_features,
                     f"../Features/conv1_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集conv1层特征表示保存成功")
            else:
                self.conv1_train_features = load(
                    f"../Features/conv1_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集conv1层特征表示加载成功")
            print(self.conv1_train_features.shape)
            # 测试集经过模型后输出的last_hidden层数据
            self.conv1_test_features = get_deep_representations(self.model_conv1, inputs_to_dataset(self.test_inputs,
                                                                                                    self.labels_correct,
                                                                                                    self.batch_size),
                                                                is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的last_hidden层数据
            self.conv1_noisy_features = get_deep_representations(self.model_conv1,
                                                                 inputs_to_dataset(self.noisy_inputs,
                                                                                   self.labels_correct,
                                                                                   self.batch_size),
                                                                 is_numpy=True)
            # 对抗集经过模型后输出的last_hidden层数据
            self.conv1_adv_features = get_deep_representations(self.model_conv1, inputs_to_dataset(self.adv_inputs,
                                                                                                   self.labels_correct,
                                                                                                   self.batch_size),
                                                               is_numpy=True)
            # 标准化所有特征数据
            all_features = np.vstack([
                self.conv1_train_features,
                self.conv1_test_features,
                self.conv1_noisy_features,
                self.conv1_adv_features
            ])
            # 标准化
            scaler = StandardScaler()
            # 对所有特征进行标准化
            all_features_scaled = scaler.fit_transform(all_features)
            # 降维：PCA统一对四个特征进行降维
            print("对conv1层特征进行PCA降维")
            n_components = 200
            pca = PCA(n_components=n_components)
            # 先统一降维
            all_features_pca = pca.fit_transform(all_features_scaled)
            # 分离回各自的特征
            self.conv1_train_features = all_features_pca[:len(self.conv1_train_features)]
            print(self.conv1_train_features.shape)
            self.conv1_test_features = all_features_pca[
                                       len(self.conv1_train_features):len(self.conv1_train_features) + len(
                                           self.conv1_test_features)]
            self.conv1_noisy_features = all_features_pca[
                                        len(self.conv1_train_features) + len(self.conv1_test_features):len(
                                            self.conv1_train_features) + len(self.conv1_test_features) + len(
                                            self.conv1_noisy_features)]
            self.conv1_adv_features = all_features_pca[
                                      len(self.conv1_train_features) + len(self.conv1_test_features) + len(
                                          self.conv1_noisy_features):]
            # ************************************************************************************************
            print("计算conv2层特征表示")
            # 由于要去除模型的最后若干层，从而得到last_hidden层数据，所以重建一个新模型，避免改变self.model
            self.model_conv2 = get_model(self.model_name)
            self.model_conv2 = ResNet18WithConv2(self.model_conv2)
            self.model_conv2.eval()
            # 训练集经过模型后输出的conv2层数据
            if not os.path.exists(f"../Features/conv2_train_features_{self.dataset_name}_{self.model_name}.pt"):
                self.conv2_train_features = get_deep_representations(self.model_conv2, self.train_dataset,
                                                                     is_numpy=True)
                # 保留训练集的深度特征表示
                save(self.conv2_train_features,
                     f"../Features/conv2_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集conv2层特征表示保存成功")
            else:
                self.conv2_train_features = load(
                    f"../Features/conv2_train_features_{self.dataset_name}_{self.model_name}.pt")
                print("训练集conv2层特征表示加载成功")
            print(self.conv2_train_features.shape)
            # 测试集经过模型后输出的last_hidden层数据
            self.conv2_test_features = get_deep_representations(self.model_conv2, inputs_to_dataset(self.test_inputs,
                                                                                                    self.labels_correct,
                                                                                                    self.batch_size),
                                                                is_numpy=True)
            # 注：噪声集和对抗集都是在测试集上生成的
            # 噪声集经过模型后输出的last_hidden层数据
            self.conv2_noisy_features = get_deep_representations(self.model_conv2,
                                                                 inputs_to_dataset(self.noisy_inputs,
                                                                                   self.labels_correct,
                                                                                   self.batch_size),
                                                                 is_numpy=True)
            # 对抗集经过模型后输出的last_hidden层数据
            self.conv2_adv_features = get_deep_representations(self.model_conv2, inputs_to_dataset(self.adv_inputs,
                                                                                                   self.labels_correct,
                                                                                                   self.batch_size),
                                                               is_numpy=True)
            # 标准化所有特征数据
            all_features = np.vstack([
                self.conv2_train_features,
                self.conv2_test_features,
                self.conv2_noisy_features,
                self.conv2_adv_features
            ])
            # 标准化
            scaler = StandardScaler()
            # 对所有特征进行标准化
            all_features_scaled = scaler.fit_transform(all_features)
            # 降维：PCA统一对四个特征进行降维
            print("对conv2层特征进行PCA降维")
            n_components = 200
            pca = PCA(n_components=n_components)
            # 先统一降维
            all_features_pca = pca.fit_transform(all_features_scaled)
            # 分离回各自的特征
            self.conv2_train_features = all_features_pca[:len(self.conv2_train_features)]
            print(self.conv2_train_features.shape)
            self.conv2_test_features = all_features_pca[
                                       len(self.conv2_train_features):len(self.conv2_train_features) + len(
                                           self.conv2_test_features)]
            self.conv2_noisy_features = all_features_pca[
                                        len(self.conv2_train_features) + len(self.conv2_test_features):len(
                                            self.conv2_train_features) + len(self.conv2_test_features) + len(
                                            self.conv2_noisy_features)]
            self.conv2_adv_features = all_features_pca[
                                      len(self.conv2_train_features) + len(self.conv2_test_features) + len(
                                          self.conv2_noisy_features):]

    def detect(self):
        if not os.path.exists(f"../Save_detect_data/{self.attack}_train_values.npy"):
            train_values = []
            test_values = []

            robust_scale = RobustScaler()
            power_scale = PowerTransformer()
            standard_scale = StandardScaler()
            quantile_scale_gauss = QuantileTransformer(output_distribution='normal')
            quantile_scale_uniform = QuantileTransformer(output_distribution='uniform')

            class MixScaler:
                def __init__(self):
                    self.standard_scale = StandardScaler()
                    self.robust_scale = RobustScaler()
                    self.power_scale = PowerTransformer()

                def fit_transform(self, data):
                    # 第一步：标准化
                    step1 = self.standard_scale.fit_transform(data)
                    # 第二步：鲁棒标准化
                    step2 = self.robust_scale.fit_transform(step1)
                    # 第三步：幂次变换
                    final_data = self.power_scale.fit_transform(step2)
                    return final_data

                def transform(self, data):
                    # 第一步：标准化
                    step1 = self.standard_scale.transform(data)
                    # 第二步：鲁棒标准化
                    step2 = self.robust_scale.transform(step1)
                    # 第三步：幂次变换
                    final_data = self.power_scale.transform(step2)
                    return final_data

            mix_scale = MixScaler()

            for train_features, test_features, noisy_features, adv_features in [
                [self.logits_train_features, self.logits_test_features, self.logits_noisy_features,
                 self.logits_adv_features],
                [self.hidden_train_features, self.hidden_test_features, self.hidden_noisy_features,
                 self.hidden_adv_features],
                [self.bn1_train_features, self.bn1_test_features, self.bn1_noisy_features,
                 self.bn1_adv_features],
                [self.bn2_train_features, self.bn2_test_features, self.bn2_noisy_features,
                 self.bn2_adv_features],
                [self.conv1_train_features, self.conv1_test_features, self.conv1_noisy_features,
                 self.conv1_adv_features],
                [self.conv2_train_features, self.conv2_test_features, self.conv2_noisy_features,
                 self.conv2_adv_features],
            ]:
                for scaler in [power_scale, robust_scale, standard_scale, quantile_scale_gauss, quantile_scale_uniform,
                               mix_scale]:
                    train_features = scaler.fit_transform(train_features)

                    test_features = scaler.transform(test_features)
                    noisy_features = scaler.transform(noisy_features)
                    adv_features = scaler.transform(adv_features)

                    # 获取语义空间距离检测数据
                    print("计算F范数>>>>>")
                    distance_train, distance_test = distance_k(train_features, test_features, noisy_features,
                                                               adv_features)
                    # 获取马氏距离值
                    print("计算马氏距离>>>>>")
                    md_train, md_test = mahalanobis_distance(train_features, test_features, noisy_features,
                                                             adv_features)
                    # 获取孤立森林得分值
                    print("计算孤立森林值>>>>>")
                    iso_train, iso_test = compute_isolation_forest(train_features, test_features, noisy_features,
                                                                   adv_features)
                    print("计算余弦相似度>>>>>")
                    cs_train, cs_test = compute_cosine_similarity(train_features, test_features, noisy_features,
                                                                  adv_features)
                    print("计算高斯混合模型分布差异>>>>>")
                    gmm_train, gmm_test = compute_gmm_log_likelihood(train_features, test_features, noisy_features,
                                                                     adv_features)
                    print("计算Z-score得分")
                    z_train, z_test = compute_z_score(train_features, test_features, noisy_features, adv_features)
                    print("计算lof得分")
                    lof_train, lof_test = compute_lof_score(train_features, test_features, noisy_features, adv_features)
                    print("计算曼哈顿距离")
                    ma_train, ma_test = compute_manhattan_distance(train_features, test_features, noisy_features,
                                                                   adv_features)

                    # 仅基于隐藏层的数据分布的检测方法
                    part_train_values, train_labels = build_detect_data(
                        distance_results=distance_train,
                        md_results=md_train,
                        iso_results=iso_train,
                        cs_results=cs_train,
                        gmm_results=gmm_train,
                        z_results=z_train,
                        lof_results=lof_train,
                        ma_results=ma_train,
                    )
                    part_test_values, test_labels = build_detect_data(
                        distance_results=distance_test,
                        md_results=md_test,
                        iso_results=iso_test,
                        cs_results=cs_test,
                        gmm_results=gmm_test,
                        z_results=z_test,
                        lof_results=lof_test,
                        ma_results=ma_test,
                    )
                    train_values.append(part_train_values)
                    test_values.append(part_test_values)

            train_values = np.column_stack(train_values)
            test_values = np.column_stack(test_values)

            # 保存训练数据
            np.save(f'../Save_detect_data/{self.attack}_train_values.npy', train_values)
            np.save(f'../Save_detect_data/{self.attack}_test_values.npy', test_values)

            # 保存标签
            np.save(f'../Save_detect_data/{self.attack}_train_labels.npy', train_labels)
            np.save(f'../Save_detect_data/{self.attack}_test_labels.npy', test_labels)
        else:
            # 加载训练数据
            train_values = np.load(f'../Save_detect_data/{self.attack}_train_values.npy')
            test_values = np.load(f'../Save_detect_data/{self.attack}_test_values.npy')

            # 加载标签
            train_labels = np.load(f'../Save_detect_data/{self.attack}_train_labels.npy')
            test_labels = np.load(f'../Save_detect_data/{self.attack}_test_labels.npy')

        print(train_values.shape)
        print(test_values.shape)
        # 根据训练数据来训练检测器
        classifier = train_detector(train_values, train_labels, self.model_name, self.attack, self.dataset_name,
                                    is_saved=False)
        # 查看检测器的检测效果
        # 最后的ROC图的命名应该和模型名，攻击名称，数据集名称有关
        detect_effect(classifier, train_values, train_labels, test_values, test_labels,
                      self.model_name, self.attack, self.dataset_name)


for att in ["FGSM"]:
    # ["FGSM", "BIM", "PGD", "CW", "Deepfool", "APGD", "MIFGSM", "UPGD", "PIFGSM", "TPGD", "TIFGSM", "Auto"]
    # ResNet18模型架构下的对抗样本检测
    detect_adv_samples = DetectAdvSamples("ResNet18", att, "CIFAR10")
    # 注意不同数据集修改位深参数
    # CIFAR10
    detect_adv_samples.detect()
