import torchattacks
from torch import load
from Utils.utils import get_data, CustomDataset, show, accuracy, seed_everything, get_model, get_device
from torch.utils.data import DataLoader


device = get_device()

# 目前仅支持这五种攻击方法生成对抗样本，若要添加新的请自行导入
attack_classes = {
    'FGSM': torchattacks.FGSM,
    'BIM': torchattacks.BIM,
    'PGD': torchattacks.PGD,
    'CW': torchattacks.CW,
    "Deepfool": torchattacks.DeepFool,
    "MIFGSM": torchattacks.MIFGSM,
    "APGD": torchattacks.APGD,
    "UPGD": torchattacks.UPGD,
    "TPGD": torchattacks.TPGD,
    "PIFGSM": torchattacks.PIFGSM,
    "TIFGSM": torchattacks.TIFGSM,
    "DIFGSM": torchattacks.DIFGSM,
    "Auto": torchattacks.AutoAttack,
    "SPSA": torchattacks.SPSA,
}


class CraftAdv:
    def __init__(self, attack_name, model_name, dataset):
        """
        :param attack_name:  攻击方法名称
        :param model_name:  攻击模型名称
        :param dataset: 所需数据集名称
        """
        self.attack_name = attack_name
        self.model_name = model_name
        self.dataset = dataset
        # 加载模型
        self.model = get_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        # 加载数据集（在测试集上生成对抗样本）
        _, self.test_dataset = get_data(self.dataset)

    def craft(self, eps=0.2, c=10, steps=100, lr=0.01, overshoot=0.02):
        """
        :param eps: 指定攻击的扰动大小
        :param c:
        :param steps:
        :param lr
        :param overshoot
        :return:
        """
        # 生成攻击方法(这里的参数可以重新设置扰动率等攻击参数)
        if self.attack_name in ["FGSM"]:
            attack = attack_classes[self.attack_name](self.model, eps=eps)
        elif self.attack_name in ["BIM", "PGD", "MIFGSM", "APGD", "UPGD", "TPGD", "TIFGSM", "DIFGSM", ]:
            attack = attack_classes[self.attack_name](self.model, eps=eps, steps=10)
        elif self.attack_name == "CW":
            print('Crafting cw adversarial samples. This may take a while...')
            attack = attack_classes[self.attack_name](self.model, c=c, steps=steps, lr=lr)
        elif self.attack_name in ["JSMA", "Auto"]:
            attack = attack_classes[self.attack_name](self.model)
        elif self.attack_name == "Deepfool":
            attack = attack_classes[self.attack_name](self.model, steps=steps, overshoot=overshoot)
        elif self.attack_name == "SPSA":
            attack = attack_classes[self.attack_name](self.model, eps=eps, lr=lr)
        elif self.attack_name == "PIFGSM":
            attack = attack_classes[self.attack_name](self.model, max_epsilon=eps)

        attack.save(self.test_dataset,
                    save_path=f"../Save_Adv/{self.attack_name}_{self.model_name}_{self.dataset}.pt")

    def visual(self, denoise=False):
        # 加载保存的对抗样本并计算在对应训练模型上的准确率（并进行适当的可视化展示）
        if not denoise:
            adv_dataset = load(f"../Save_Adv/{self.attack_name}_{self.model_name}_{self.dataset}.pt")
        if denoise:
            adv_dataset = load(f"../Save_Adv/Denoise_{self.attack_name}_{self.model_name}_{self.dataset}.pt")
        adv_inputs = adv_dataset['adv_inputs']
        # 注意，这里是真实标签
        true_labels = adv_dataset['labels']
        adv_dataset = CustomDataset(adv_inputs, true_labels)
        # 批量大小默认为256
        adv_dataset = DataLoader(adv_dataset, batch_size=256, shuffle=False, drop_last=False)
        # 计算正常样本的准确率
        print("干净样本：")
        accuracy(self.model, self.test_dataset)
        # 计算对抗样本的准确率，并接收返回的对抗样本的预测标签
        print("对抗样本：")
        adv_labels = accuracy(self.model, adv_dataset)
        # 干净样本的前十六张图表示
        show(self.test_dataset, self.dataset)
        # 对抗样本带真实标签的前十六张图表示
        show(adv_dataset, self.dataset)
        # 对抗样本带预测标签的前十六张图表示
        adv_dataset = CustomDataset(adv_inputs, adv_labels)
        adv_dataset = DataLoader(adv_dataset, batch_size=256, shuffle=False, drop_last=False)
        show(adv_dataset, self.dataset)


# 避免其他包调用时执行下面的代码
if __name__ == "__main__":
    seed_everything()
    # # 针对LeNet5构造对抗样本
    # craft_adv = CraftAdv('FGSM', 'LeNet5', 'MNIST')
    # craft_adv.craft(eps=0.30, c=200, steps=100, lr=1.30)
    # craft_adv.visual()

    # 针对ResNet18构造对抗样本 FGSM:eps=0.05 BIM,PGD eps=0.01, steps=10
    # CW c=25, lr=0.15, steps=10
    # (这些参数是为了保证所有攻击的扰动大小一致，用来验证多分类性能)
    # 扰动大小要控制在：1.1-1.2
    craft_adv = CraftAdv("SPSA", 'ResNet18', 'CIFAR10')
    craft_adv.craft()
    craft_adv.visual()
