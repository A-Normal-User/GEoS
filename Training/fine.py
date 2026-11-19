import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model import Net
import matplotlib.pyplot as plt
import sys
class Tee:
    def __init__(self, filename, mode='w', encoding=None):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.stdout.write(text)  # 控制台输出
        self.file.write(text)    # 文件写入
        self.flush()  # 确保及时写入文件

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def isatty(self):
        return False
    
    def close(self):
        if self.file and not self.file.closed:
            sys.stdout = self.stdout  # 恢复原sys.stdout
            self.file.close()
# 这是一个随机种子，便于复现训练结果
seed = [
    1131796,
    18094361,
    26317076,
    26318864,
    157564161,
    218105633,
    448585456,
    539035672,
    609641183,
    1211200801,
]
parser = argparse.ArgumentParser()
# 接收超参数
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()
index = args.index
seed = seed[index]
outputname = 'DGM_fine'
tee = Tee(f'./log/{seed}_{outputname}.txt', 'w')
# device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda:{index % 2}' if torch.cuda.is_available() else 'cpu')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.manual_seed(seed)
np.random.seed(seed)

train_len = 31 + 2 + 4
test_ratio = 0.01
data = torch.load('../Database/database_new_max.pt', weights_only=True, map_location='cpu')
data = data[~torch.isnan(data).any(dim=1)]

data_val = torch.load('../Database/database_new_max_val.pt', weights_only=True, map_location='cpu')
data_val = data_val[~torch.isnan(data_val).any(dim=1)]
# 从val中抽样一半到训练集中，剩下的一半作为新的验证集
num_val_samples = data_val.shape[0]
num_samples_to_move = num_val_samples // 2
# 从val中随机抽样num_samples_to_move个样本
indices = torch.randperm(num_val_samples)
data_to_move = data_val[indices[:num_samples_to_move]]
data_val = data_val[indices[num_samples_to_move:]]
data = torch.cat((data, data_to_move), dim=0)
X_train = data[:, :train_len]
y_train = data[:, train_len:]
X_val = data_val[:, :train_len]
y_val = data_val[:, train_len:]

# 创建TensorDataset
feature_count = 31
liquid_mask = X_train[:, feature_count+5] > 0
X_train_liquid = X_train[liquid_mask]
X_train_gas = X_train[~liquid_mask]
y_train_liquid = y_train[liquid_mask]
y_train_gas = y_train[~liquid_mask]
train_dataset_liquid = torch.utils.data.TensorDataset(X_train_liquid[:, 0:feature_count].detach().clone(), 
                                               X_train_liquid[:, feature_count:feature_count+1].detach().clone(), 
                                               X_train_liquid[:, feature_count+1:feature_count+2].detach().clone(), 
                                               X_train_liquid[:, feature_count+2:feature_count+3].detach().clone(), 
                                               X_train_liquid[:, feature_count+3:feature_count+4].detach().clone(), 
                                               X_train_liquid[:, feature_count+4:feature_count+5].detach().clone(),
                                               y_train_liquid.detach().clone())
train_dataset_gas = torch.utils.data.TensorDataset(X_train_gas[:, 0:feature_count].detach().clone(), 
                                             X_train_gas[:, feature_count:feature_count+1].detach().clone(), 
                                             X_train_gas[:, feature_count+1:feature_count+2].detach().clone(), 
                                             X_train_gas[:, feature_count+2:feature_count+3].detach().clone(), 
                                             X_train_gas[:, feature_count+3:feature_count+4].detach().clone(), 
                                             X_train_gas[:, feature_count+4:feature_count+5].detach().clone(),
                                             y_train_gas.detach().clone())
liquid_mask_val = X_val[:, feature_count+5] > 0
X_val_liquid = X_val[liquid_mask_val]
X_val_gas = X_val[~liquid_mask_val]
y_val_liquid = y_val[liquid_mask_val]
y_val_gas = y_val[~liquid_mask_val]
val_dataset_liquid = torch.utils.data.TensorDataset(X_val_liquid[:, 0:feature_count].detach().clone(), 
                                             X_val_liquid[:, feature_count:feature_count+1].detach().clone(), 
                                             X_val_liquid[:, feature_count+1:feature_count+2].detach().clone(), 
                                             X_val_liquid[:, feature_count+2:feature_count+3].detach().clone(), 
                                             X_val_liquid[:, feature_count+3:feature_count+4].detach().clone(), 
                                             X_val_liquid[:, feature_count+4:feature_count+5].detach().clone(), 
                                             y_val_liquid.detach().clone())
val_dataset_gas = torch.utils.data.TensorDataset(X_val_gas[:, 0:feature_count].detach().clone(), 
                                           X_val_gas[:, feature_count:feature_count+1].detach().clone(), 
                                           X_val_gas[:, feature_count+1:feature_count+2].detach().clone(), 
                                           X_val_gas[:, feature_count+2:feature_count+3].detach().clone(), 
                                           X_val_gas[:, feature_count+3:feature_count+4].detach().clone(), 
                                           X_val_gas[:, feature_count+4:feature_count+5].detach().clone(),
                                           y_val_gas.detach().clone())

# 创建DataLoader
train_loader_liquid = DataLoader(train_dataset_liquid, batch_size=1024, shuffle=True, pin_memory=True)
train_loader_gas = DataLoader(train_dataset_gas, batch_size=1024, shuffle=True, pin_memory=True)
val_loader_liquid = DataLoader(val_dataset_liquid, batch_size=2048*8, shuffle=False, pin_memory=True)
val_loader_gas = DataLoader(val_dataset_gas, batch_size=2048*8, shuffle=False, pin_memory=True)

model = Net().to(device)
# 加载预训练模型
model.load_state_dict(torch.load(f'./model/model_max_{seed}.pth', map_location=device, weights_only=True), strict=False)

# Training Loop
num_epochs = 60
criterion = nn.SmoothL1Loss()
optimizer_liquid = optim.AdamW(model.parameters(), lr = 2e-5, weight_decay=1e-2)
optimizer_gas = optim.AdamW(model.parameters(), lr = 2e-5, weight_decay=1e-2)
schedule_liquid = optim.lr_scheduler.ReduceLROnPlateau(optimizer_liquid, 'min', factor = 0.5, patience = 4, min_lr = 5e-7)
schedule_gas = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gas, 'min', factor = 0.5, patience = 4, min_lr = 1e-6)
train_losses_liquid = []
train_losses_gas = []
val_losses_liquid = []
val_losses_gas = []
# 向output.txt文件中写入训练结果
print(f'Seed: {seed}')
R = torch.tensor(8.314, requires_grad=False).to(device)
min_val_loss = float('inf')

class RobustLogLoss(nn.Module):
    def __init__(self, eps=1e-4, threshold=1e-5, lossfuction=nn.SmoothL1Loss()):
        super(RobustLogLoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.loss_fn = lossfuction

    def forward(self, pred, target):
        significant_mask = target > self.threshold
        new_pred = torch.asinh(pred / self.eps)
        new_target = torch.asinh(target / self.eps)
        if significant_mask.any():
            main_loss = self.loss_fn(
                new_pred[significant_mask],
                new_target[significant_mask]
            )
        else:
            main_loss = torch.tensor(0.0, device=pred.device)
        # 增加pred为负的惩罚，避免模型预测出负值
        penalty = torch.relu(-pred).sum()
        return main_loss, penalty

eps = torch.tensor(1e-4, dtype=torch.double, requires_grad=False, device=device)
threshold = torch.tensor(1e-6, dtype=torch.double, requires_grad=False, device=device)
Z_loss_function = RobustLogLoss(eps=eps, threshold=threshold, lossfuction=criterion).to(device).double()
Z_loss_function = torch.compile(Z_loss_function)
switch_epoch = 15
for epoch in range(num_epochs):
    model.train()
    total_train_loss_liquid = 0
    total_limit_Z_loss_liquid = 0
    total_limit_kappa_loss_liquid = 0
    total_train_loss_gas = 0
    total_limit_Z_loss_gas = 0
    total_limit_kappa_loss_gas = 0
    # 训练模型
    for param in model.mole.parameters():
        param.requires_grad = True
    for param in model.temp.parameters():
        param.requires_grad = True
    for param in model.pres.parameters():
        param.requires_grad = True
    for features, TC, PC, T, P, Z_ref, labels in train_loader_liquid:
        features, TC, PC, T, P, Z_ref, labels = features.to(device, non_blocking=True), TC.to(device, non_blocking=True), PC.to(device, non_blocking=True), T.to(device, non_blocking=True), P.to(device, non_blocking=True), Z_ref.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer_liquid.zero_grad()
        P = P.detach()
        P.requires_grad_(True)
        T_norm = T / TC
        P_norm = torch.log(P) - torch.log(PC)
        output = model.forward_liquid(features, T_norm, P_norm)
        loss1 = criterion(output, labels)
        dOut_dP = torch.autograd.grad(output.sum(), P_norm, create_graph=True)[0]
        Z = 1 + dOut_dP
        loss2, loss2_penalty = Z_loss_function(Z, Z_ref)
        ka = torch.autograd.grad(Z.sum(), P, create_graph=True)[0]
        
        loss2 += loss2_penalty
        loss3 = torch.nn.functional.relu(ka - Z / P).sum()

        loss2_penalty_ = loss2_penalty.clone().detach()
        loss3_ = loss3.clone().detach()
        loss2_ = loss2.clone().detach()
        loss1_ = loss1.clone().detach()
        if loss2_penalty_ > 1e-6:
            loss2_penalty *= torch.clamp(loss2_ / loss2_penalty_, max=50)
        if loss2_ > 1e-6:
            loss2 *= 2 * torch.clamp(loss1_ / loss2_, max=1)
        if loss3_ > 1e-7:
            # 调整loss3使之与loss1在同一数量级，并clip上限倍数为1，避免过大
            loss3 *= torch.clamp((loss1_ + loss2_) / loss3_, max=10)
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer_liquid.step()
        total_train_loss_liquid += loss1_.item()
        total_limit_Z_loss_liquid +=  loss2_.item()
        total_limit_kappa_loss_liquid += loss3_.item()

    # 关闭mole，temp，pres层的梯度计算，避免在训练gas时更新这三层的参数
    for param in model.mole.parameters():
        param.requires_grad = False
    for param in model.temp.parameters():
        param.requires_grad = False
    for param in model.pres.parameters():
        param.requires_grad = False
    for features, TC, PC, T, P, Z_ref, labels in train_loader_gas:
        features, TC, PC, T, P, Z_ref, labels = features.to(device, non_blocking=True), TC.to(device, non_blocking=True), PC.to(device, non_blocking=True), T.to(device, non_blocking=True), P.to(device, non_blocking=True), Z_ref.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer_gas.zero_grad()
        P = P.detach()
        P.requires_grad_(True)
        T_norm = T / TC
        P_norm = torch.log(P) - torch.log(PC)
        output = model.forward_gas(features, T_norm, P_norm)
        loss1 = criterion(output, labels)
        dOut_dP = torch.autograd.grad(output.sum(), P_norm, create_graph=True)[0]
        Z = 1 + dOut_dP
        loss2, loss2_penalty = Z_loss_function(Z, Z_ref)
        ka = torch.autograd.grad(Z.sum(), P, create_graph=True)[0]
        loss3 = torch.nn.functional.relu(ka - Z / P).sum()
        loss2_penalty_ = loss2_penalty.clone().detach()
        loss1_ = loss1.clone().detach()
        if loss2_penalty_ > 1e-6:
            loss2_penalty *= torch.clamp(loss1_ / loss2_penalty_, max=50)
        loss2 += loss2_penalty
        loss3_ = loss3.clone().detach()
        loss2_ = loss2.clone().detach()
        if loss2_ > 1e-6:
            loss2 *= torch.clamp(loss1_ / loss2_, max=1)
        if loss3_ > 1e-6:
            loss3 *= torch.clamp((loss1_ + loss2_) / loss3_, max=10)
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_gas.step()
        total_train_loss_gas += loss1_.item()
        total_limit_Z_loss_gas += loss2_.item()
        total_limit_kappa_loss_gas += loss3_.item()
        
    # 计算训练集的损失
    avg_train_loss_liquid = total_train_loss_liquid / len(train_loader_liquid)
    avg_limit_Z_loss_liquid = total_limit_Z_loss_liquid / len(train_loader_liquid)
    avg_limit_kappa_loss_liquid = total_limit_kappa_loss_liquid / len(train_loader_liquid)
    avg_train_loss_gas = total_train_loss_gas / len(train_loader_gas)
    avg_limit_Z_loss_gas = total_limit_Z_loss_gas / len(train_loader_gas)
    avg_limit_kappa_loss_gas = total_limit_kappa_loss_gas / len(train_loader_gas)
    # 调整学习率
    schedule_liquid.step(avg_train_loss_liquid)
    train_losses_liquid.append(avg_train_loss_liquid)
    schedule_gas.step(avg_train_loss_gas)
    train_losses_gas.append(avg_train_loss_gas)

    model.eval()
    total_val_loss_liquid = 0
    total_val_loss_gas = 0
    # 计算验证集的效果
    with torch.no_grad():
        for features, TC, PC, T, P, Z_ref, labels in val_loader_liquid:
            features, TC, PC, T, P, Z_ref, labels = features.to(device, non_blocking=True), TC.to(device, non_blocking=True), PC.to(device, non_blocking=True), T.to(device, non_blocking=True), P.to(device, non_blocking=True), Z_ref.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            T_norm = T / TC
            P_norm = torch.log(P) - torch.log(PC)
            output = model.forward_liquid(features, T_norm, P_norm)
            loss = criterion(output, labels)
            total_val_loss_liquid += loss.item()
        for features, TC, PC, T, P, Z_ref, labels in val_loader_gas:
            features, TC, PC, T, P, Z_ref, labels = features.to(device, non_blocking=True), TC.to(device, non_blocking=True), PC.to(device, non_blocking=True), T.to(device, non_blocking=True), P.to(device, non_blocking=True), Z_ref.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            T_norm = T / TC
            P_norm = torch.log(P) - torch.log(PC)
            output = model.forward_gas(features, T_norm, P_norm)
            loss = criterion(output, labels)
            total_val_loss_gas += loss.item()
    # 计算验证集的损失
    avg_val_loss_liquid = total_val_loss_liquid / len(val_loader_liquid)
    avg_val_loss_gas = total_val_loss_gas / len(val_loader_gas)
    if epoch >= switch_epoch and avg_limit_Z_loss_liquid < min_val_loss:
        min_val_loss = avg_limit_Z_loss_liquid
        model_save_path = f'./model/model_{outputname}_{seed}.pth'
        model = model.to('cpu')
        torch.save(model.state_dict(), model_save_path)
        model = model.to(device)
        print(f'Model saved')
    val_losses_liquid.append(avg_val_loss_liquid)
    val_losses_gas.append(avg_val_loss_gas)
    # 输出训练Loss
    print(f'Epoch {epoch+1}/{num_epochs}(liq), Train: {avg_train_loss_liquid}, Limit Z: {avg_limit_Z_loss_liquid}, Limit Kappa: {avg_limit_kappa_loss_liquid}, Val Loss: {avg_val_loss_liquid}, lr: {optimizer_liquid.param_groups[0]["lr"]}')
    print(f'Epoch {epoch+1}/{num_epochs}(gas), Train: {avg_train_loss_gas}, Limit Z: {avg_limit_Z_loss_gas}, Limit Kappa: {avg_limit_kappa_loss_gas}, Val Loss: {avg_val_loss_gas}, lr: {optimizer_gas.param_groups[0]["lr"]}')
# 绘制训练和验证Loss曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses_liquid, label='Train Loss (Liquid)')
plt.plot(val_losses_liquid, label='Validation Loss (Liquid)')
plt.plot(train_losses_gas, label='Train Loss (Gas)')
plt.plot(val_losses_gas, label='Validation Loss (Gas)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
# 保存Loss曲线
plt.savefig(f'./img/{seed}_{outputname}.jpg')
tee.close()