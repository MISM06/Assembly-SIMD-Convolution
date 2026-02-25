import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# ۱. تعریف مدل CNN پیشرفته‌تر اما بهینه برای پیاده‌سازی در C
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # لایه کانولوشن: 4 تا کرنل 3x3 می‌سازیم. (بدون پدینگ)
        # عکس ورودی 28x28 است. خروجی این لایه میشه 4 تا عکس 26x26
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=0, bias=False)
        
        # لایه ادغام (Max Pooling): با پنجره 2x2
        # خروجی مرحله قبل (26x26) رو نصف می‌کنه و تبدیل میشه به 4 تا عکس 13x13
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # لایه تصمیم‌گیری نهایی (Fully Connected)
        # 4 تا عکس 13x13 داریم. 4 * 13 * 13 = 676 ویژگی مستقل
        # این 676 ویژگی رو وصل می‌کنیم به 10 خروجی (اعداد 0 تا 9)
        self.fc = nn.Linear(4 * 13 * 13, 10)

    # نحوه حرکت داده‌ها در شبکه (Forward Propagation)
    def forward(self, x):
        x = self.conv1(x)          # اعمال 4 کرنل روی عکس
        x = torch.relu(x)          # حذف مقادیر منفی (تابع فعال‌ساز)
        x = self.pool(x)           # کوچک کردن عکس‌ها (Max Pooling)
        x = x.view(x.size(0), -1)  # تبدیل 4 عکس دو بعدی به یک آرایه یک‌بعدی 676 تایی
        x = self.fc(x)             # امتیازدهی به کلاس‌های 0 تا 9
        return x

# ۲. آماده‌سازی دیتاست MNIST (اعداد دست‌نویس)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ۳. آموزش مدل (1 یا 2 اپوک کافیه برای گرفتن دقت خوب)
print("Training Model Started...")
for epoch in range(2): # دو بار کل دیتاست رو مرور می‌کنه تا دقیق‌تر بشه
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Finished!")
print("Training Finished!")

# ۴. استخراج وزن‌ها و ذخیره برای استفاده در زبان C
# تبدیل مقادیر تنسور به آرایه‌های یک‌بعدی فلت (Flatten)
conv_kernels = model.conv1.weight.data.numpy().flatten() # 4 * 1 * 3 * 3 = 36 تا عدد
fc_weight = model.fc.weight.data.numpy()                 # 10 ردیف در 676 ستون
fc_bias = model.fc.bias.data.numpy().flatten()           # 10 تا عدد

print("Exporting weights to weights.h ...")
with open("weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    
    # ذخیره 4 کرنل کانولوشن (کلا 36 تا عدد)
    f.write("// 4 Kernels of size 3x3\n")
    f.write("float conv_kernels[36] = {")
    f.write(", ".join([f"{w:.6f}" for w in conv_kernels]))
    f.write("};\n\n")
    
    # ذخیره بایاس‌های لایه آخر
    f.write("// Biases for the 10 output classes\n")
    f.write("float fc_bias[10] = {")
    f.write(", ".join([f"{b:.6f}" for b in fc_bias]))
    f.write("};\n\n")
    
    # ذخیره ماتریس وزن‌های لایه Fully Connected (سایز 10 در 676)
    f.write("// Weights for Fully Connected layer (10 x 676)\n")
    f.write("float fc_weight[10][676] = {\n")
    for row in fc_weight:
        f.write("  {" + ", ".join([f"{w:.6f}" for w in row]) + "},\n")
    f.write("};\n\n")
    
    f.write("#endif\n")

print("Weights exported to weights.h successfully!")