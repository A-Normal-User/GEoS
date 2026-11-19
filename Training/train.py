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
outputname = 'max'
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
train_dataset = torch.utils.data.TensorDataset(X_train[:, 0:feature_count].detach().clone(), 
                                               X_train[:, feature_count:feature_count+1].detach().clone(), 
                                               X_train[:, feature_count+1:feature_count+2].detach().clone(), 
                                               X_train[:, feature_count+2:feature_count+3].detach().clone(), 
                                               X_train[:, feature_count+3:feature_count+4].detach().clone(), 
                                               X_train[:, feature_count+4:feature_count+5].detach().clone(),
                                               X_train[:, feature_count+5:feature_count+6].detach().clone(), y_train.detach().clone())
val_dataset = torch.utils.data.TensorDataset(X_val[:, 0:feature_count].detach().clone(), 
                                             X_val[:, feature_count:feature_count+1].detach().clone(), 
                                             X_val[:, feature_count+1:feature_count+2].detach().clone(), 
                                             X_val[:, feature_count+2:feature_count+3].detach().clone(), 
                                             X_val[:, feature_count+3:feature_count+4].detach().clone(), 
                                             X_val[:, feature_count+4:feature_count+5].detach().clone(), 
                                             X_val[:, feature_count+5:feature_count+6].detach().clone(), y_val.detach().clone())

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048*8, shuffle=False)

model = Net().to(device)
# 初始化model中的参数
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Training Loop
num_epochs = 260
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr = 0.002, weight_decay=1e-3)
schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 6, min_lr = 1e-6)
train_losses = []
val_losses = []
# 向output.txt文件中写入训练结果
print(f'Seed: {seed}')
R = torch.tensor(8.314, requires_grad=False).to(device)
min_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_limit_Z_ref = 0
    total_limit_Z_loss = 0
    total_limit_kappa_loss = 0
    # 训练模型
    for features, TC, PC, T, P, Z_ref, mask, labels in train_loader:
        features, TC, PC, T, P, Z_ref, mask, labels = features.to(device), TC.to(device), PC.to(device), T.to(device), P.to(device), Z_ref.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        P = P.detach()
        P.requires_grad_(True)
        P_norm = torch.log(P) - torch.log(PC)
        output = model(features, T / TC, P_norm, mask)
        loss1 = criterion(output, labels)
        dOut_dP = torch.autograd.grad(output.sum(), P_norm, create_graph=True)[0]
        Z = 1 + dOut_dP
        ka = P * torch.autograd.grad(Z.sum(), P, create_graph=True)[0]
        loss2 = torch.nn.functional.relu(-Z).sum()
        loss3 = torch.nn.functional.relu(ka - Z).sum()
        loss4 = criterion(Z, Z_ref)
        
        loss4_ = loss4.clone().detach()
        loss3_ = loss3.clone().detach()
        loss2_ = loss2.clone().detach()
        loss1_ = loss1.clone().detach()
        # print(f'loss1: {loss1_}, loss2: {loss2_}, loss3: {loss3_}, loss4: {loss4_}')
        if loss2_ > 1e-6:
            loss2 *= torch.clamp(loss1_ / loss2_, max=2)
        if loss3_ > 1e-6:
            loss3 *= torch.clamp(loss1_ / loss3_, max=0.05)
        if loss4_ > 1e-6:
            loss4 *= torch.clamp(loss1_ / loss4_, max=4)

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()
        total_train_loss += loss1_.item()
        total_limit_Z_loss += loss2_.item()
        total_limit_Z_ref += loss4_.item()
        total_limit_kappa_loss += loss3_.item()
    # 计算训练集的损失
    avg_train_loss = total_train_loss / len(train_loader)
    avg_limit_Z_loss = total_limit_Z_loss / len(train_loader)
    avg_limit_Z_ref = total_limit_Z_ref / len(train_loader)
    avg_limit_kappa_loss = total_limit_kappa_loss / len(train_loader)
    # 调整学习率
    schedule.step(avg_train_loss)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    # total_val_limit_loss = 0
    # 计算验证集的效果
    with torch.no_grad():
        for features, TC, PC, T, P, Z_ref, mask, labels in val_loader:
            features, TC, PC, T, P, Z_ref, mask, labels = features.to(device), TC.to(device), PC.to(device), T.to(device), P.to(device), Z_ref.to(device), mask.to(device), labels.to(device)
            # P.requires_grad_(True)
            T_norm = T / TC
            P_norm = torch.log(P) - torch.log(PC)
            output = model(features, T_norm, P_norm, mask)
            loss = criterion(output, labels)
            # Z = 1 + P * torch.autograd.grad(output.sum(), P, create_graph=True)[0]
            # loss2 = torch.nn.functional.relu(-Z).mean()
            # total_val_limit_loss += loss2.item()
            total_val_loss += loss.item()
    # 计算验证集的损失
    avg_val_loss = total_val_loss / len(val_loader)
    # total_val_limit_loss = total_val_limit_loss / len(val_loader)
    if epoch > 20 and avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        model_save_path = f'./model/model_{outputname}_{seed}.pth'
        model = model.to('cpu')
        torch.save(model.state_dict(), model_save_path)
        model = model.to(device)
        print(f'Model saved')
    val_losses.append(avg_val_loss)
    # 输出训练Loss
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Limit Z Loss: {avg_limit_Z_loss}, Limit Z Ref Loss: {avg_limit_Z_ref}, Limit Kappa Loss: {avg_limit_kappa_loss}, Validation Loss: {avg_val_loss}, lr: {optimizer.param_groups[0]["lr"]}')
# 绘制训练和验证Loss曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
# 保存Loss曲线
plt.savefig(f'./img/{seed}_{outputname}.jpg')
tee.close()