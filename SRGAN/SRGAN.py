####### SRGAN: từ ảnh có độ phân giải thấp, tạo ra ảnh có độ phân giải cao.
####### https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch
import numpy as np
import pandas as pd
import os, math, sys
import glob, itertools # Thao tác với tập tin; các phép toán hoán vị tổ hợp,...
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

random.seed(42)
import warnings
warnings.filterwarnings("ignore")

##### Settings
# load pretrained models
load_pretrained_models = True
# name of the dataset
dataset_path = "../input/celeba-dataset/img_align_celeba/img_align_celeba"

# number of epochs of training
n_epochs = 2
# size of the batches
batch_size = 16
# adam: learning rate
lr = 0.00008
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of second order momentum of gradient
b2 = 0.999
# epoch from which to start lr decay
decay_epoch = 100

# # number of cpu threads to use during batch generation
n_cpu = 8

# high res. image height
hr_height = 256
# high res. image width
hr_width = 256
# number of image channels
channels = 3

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

cuda = torch.cuda.is_available()
hr_shape = (hr_height, hr_width)



###### Tạo Class để tiền xử lý dữ liệu: chuyển ảnh thành tensor, normalize,..
# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
class ImageDataset(Dataset):
    def __init__(self, files, hr_shape):
        # files: danh sách các đường dẫn dẫn tới ảnh
        hr_height, hr_width = hr_shape
        # Chuyển đổi cho hình ảnh có độ phân giải thấp
        # Image.BICUBIC: Đây là phương pháp nội suy bậc 3 được sử dụng để thay đổi kích thước hình ảnh
            # Nearest Neighbor (Gần nhất) (Image.NEAREST):
            #     Ưu điểm: Nhanh nhất, bảo toàn các giá trị pixel chính xác, không làm mờ ảnh.
            #     Nhược điểm: Hình ảnh có thể trông răng cưa và không mịn màng khi phóng to.
            # Bilinear (Bậc hai) (Image.BILINEAR):
            #     Ưu điểm: Tương đối nhanh, cho kết quả mịn hơn so với Nearest Neighbor.
            #     Nhược điểm: Có thể làm mờ hình ảnh, đặc biệt khi thay đổi kích thước lớn.
            # Bicubic (Bậc ba) (Image.BICUBIC):
            #     Ưu điểm: Thường tạo ra hình ảnh mịn màng hơn, duy trì chi tiết tốt hơn so với Bilinear.
            #     Nhược điểm: Chậm hơn Nearest Neighbor và Bilinear, có thể gây ra hiệu ứng quầng sáng (halo effects) ở các biên.
            # Lanczos (Image.LANCZOS):
            #     Ưu điểm: Thường cho chất lượng cao nhất, giữ lại nhiều chi tiết, đặc biệt là với hình ảnh giảm kích thước.
            #     Nhược điểm: Chậm nhất trong các phương pháp thông dụng, có thể gây ra hiệu ứng aliasing.
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        
        # Chuyển đổi cho hình ảnh có độ phân giải cao
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files = files
    
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}
    
    def __len__(self):
        return len(self.files)
    
    
## Get train/test Dataloaders
# sử dụng sorted(glob.glob(dataset_path + "/*.*"): để tạo danh sách các link to image theo thứ tự tăng dần
# glob.glob(dataset_path + "/*.*"):
    # glob là một thư viện trong Python dùng để tìm tất cả các đường dẫn đến các tệp tin phù hợp với một mẫu nhất định.
    # dataset_path + "/*.*" tạo một mẫu tìm kiếm tất cả các tệp tin trong thư mục dataset_path với bất kỳ phần mở rộng nào (*.*).
    # glob.glob(...) trả về danh sách các đường dẫn đến các tệp tin hình ảnh trong thư mục.
# sorted(...):
    # Sắp xếp danh sách các đường dẫn đến hình ảnh. Đây là một bước không bắt buộc nhưng giúp đảm bảo rằng dữ liệu được chia theo một thứ tự nhất quán.
train_paths, test_paths = train_test_split(
    sorted(glob.glob(dataset_path + "/*.*")), 
    test_size=0.02, 
    random_state=42
)


# train_dataloader sẽ chứa các batch dữ liệu với các ảnh đã được xử lý bởi lớp ImageDataset. 
    # Cụ thể, mỗi batch sẽ là một dictionary với các khóa "lr" và "hr", mỗi khóa chứa tensor 
    # của các ảnh
# Kích thước Batch: Nếu batch_size là 16, thì mỗi batch trong train_dataloader sẽ có kích thước:
#     batch["lr"].size(): (16, 3, hr_height // 4, hr_width // 4)
#     batch["hr"].size(): (16, 3, hr_height, hr_width)
# Khi DataLoader của PyTorch lấy dữ liệu từ class ImageDataset, nó sẽ tự gọi phương thức __getitem__ để lấy các phần tử dữ liệu và chia thành các batch
train_dataloader = DataLoader(ImageDataset(train_paths, hr_shape=hr_shape), batch_size=batch_size, shuffle=True, num_workers=n_cpu)
test_dataloader = DataLoader(ImageDataset(test_paths, hr_shape=hr_shape), batch_size=int(batch_size*0.75), shuffle=True, num_workers=n_cpu)


## Defind model class
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True) # use vgg19 model
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]) # Lấy 18 lớp đầu tiên của phần mạng tích chập của VGG19.

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# input là ảnh có độ phân giải thấp chứ không phải latent vector
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            # dùng kernel_size = 9 x 9 để trích xuất được nhiều đặc trưng hơn 
            # ảnh có input_size sau khi qua lớp này thì vẫn giữ nguyên input_size
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )


        ## Residual blocks (k times)
        # Đảm bảo số lượng residual block vừa đủ để giúp mô hình tránh vanishing gradient, nếu nhiều có có thể dẫn đến
            # mô hình học quá kỹ các chi tiết dữ liệu -> overfitting
        # Trường hợp này dùng append được vì nó chỉ appen mỗi 1 phần tử -> do đó kết quả sẽ là array 1d
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks) # Xâu các khổi res block lại với nhau

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64, 0.8)
        )


        ## Upsampling layers (2 times)
        # Sử dụng += thay vì appen vì:
            # appen:
                # [
                #     [nn.Conv2d(...), nn.BatchNorm2d(...), nn.PixelShuffle(...), nn.PReLU()],
                #     [nn.Conv2d(...), nn.BatchNorm2d(...), nn.PixelShuffle(...), nn.PReLU()]
                # ] 
            # +=:
                # [
                #     nn.Conv2d(...),
                #     nn.BatchNorm2d(...),
                #     nn.PixelShuffle(...),
                #     nn.PReLU(),
                #     nn.Conv2d(...),
                #     nn.BatchNorm2d(...),
                #     nn.PixelShuffle(...),
                #     nn.PReLU()
                # ]
        # Sequential: yêu cầu một danh sách phẳng của các module do đó dùng appen là lỗi
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                # nn.Conv2d(64, 64, 3, 1, 1),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                # input của nn.PixelShuffle: (N, C, H, W) với C = số kênh đầu vào.
                # output của nn.PixelShuffle: (N, C/(r^2), Hr, Wr), với r là upscale_factor.
                # nn.PixelShuffle: khi muốn tái tạo hình ảnh có chất lượng cao, sắc nét và chi tiết hơn
                # nn.Upsample: khi muốn đơn giản hóa mô hình và không cần quá chú trọng đến chất lượng hình ảnh tái tạo.
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), 
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2) # skip connection, out1 là input của khối residual block, out2 là output của residual block
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 16), int(in_width / 16)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels # 3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            # extend <=> +=
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        # => img: [batch_size, in_channels, in_height, in_width] -> [batch_size, 512, in_height / 16, in_width / 16] 
        
        # Không sử dụng fc layer mà sử dụng Conv2d để biến ảnh thành [batch_size, 1, in_height / 16, in_width / 16] để tính toán loss bằng các so từng pixel
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
    

#### TRAIN SRGAN
# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(channels, *hr_shape))

# feature_extractor: được sử dụng chủ yếu để tính toán perceptual loss
# FeatureExtractor, thường là các mạng như VGG19, giúp trích xuất các đặc trưng của ảnh. 
    # Sau đó, perceptual loss được tính toán dựa trên các đặc trưng này. Perceptual loss đo 
    # lường sự khác biệt giữa các đặc trưng của ảnh gốc và ảnh sinh ra thay vì sự khác biệt pixel-by-pixel
    # bằng cách sử dụng nn.MSELoss() hoặc nn.L1Loss
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
# Đặt mô hình feature extractor vào chế độ inference (chế độ đánh giá). Điều này có nghĩa 
    # là mô hình sẽ không cập nhật trọng số và sẽ không tính toán các giá trị như dropout hay 
    # batch normalization, giúp mô hình chạy nhanh hơn và chính xác hơn trong quá trình đánh giá.
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss() # loss bằng cách so pixel
# L1 loss giống MSELoss nhưng không sử dụng bình phương mà sử dụng absolute -> khiến cho giá trị bất thường không còn đáng kể như L2Loss (MSE)
criterion_content = torch.nn.L1Loss() # loss bằng cách so pixel

# Load pretrained models
if load_pretrained_models:
    generator.load_state_dict(torch.load("../input/single-image-super-resolution-gan-srgan-pytorch/saved_models/generator.pth"))
    discriminator.load_state_dict(torch.load("../input/single-image-super-resolution-gan-srgan-pytorch/saved_models/discriminator.pth"))
    
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_gen_losses, train_disc_losses, train_counter = [], [], []
test_gen_losses, test_disc_losses = [], []
test_counter = [idx*len(train_dataloader.dataset) for idx in range(1, n_epochs+1)]


for epoch in range(n_epochs):

    ### Training
    gen_loss, disc_loss = 0, 0
    
    # Tạo thanh tiến trình (progress bar) trong quá trình huấn luyện, giúp theo dõi tiến độ của các epoch một cách trực quan.
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))
    for batch_idx, imgs in enumerate(tqdm_bar):
        
        # Đặt mô hình vào chế độ huấn luyện >< với chế độ kiếm tra (.val())
        generator.train()
        discriminator.train()
        
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        
        # Adversarial ground truths
        # Tạo tensor có kích thước (batch_size, 1, patch_h, patch_w) với tất cả phần tử bằng 1 -> (bằng kích thước ảnh -> so pixel)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # Tạo tensor có kích thước (batch_size, 1, patch_h, patch_w) với tất cả phần tử bằng 0 -> (bằng kích thước ảnh -> so pixel)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss 
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        gen_loss += loss_G.item()
        train_gen_losses.append(loss_G.item())
        disc_loss += loss_D.item()
        train_disc_losses.append(loss_D.item())
        
        # Lưu trữ thông tin về số lượng ví dụ đã xử lý
        train_counter.append(batch_idx*batch_size + imgs_lr.size(0) + epoch*len(train_dataloader.dataset)) 
        # hiển thị thông tin tổn thất bên cạnh thanh tiến trình
        tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))

    # Testing
    gen_loss, disc_loss = 0, 0
    tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))
    for batch_idx, imgs in enumerate(tqdm_bar):
        generator.eval(); discriminator.eval()
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        ### Eval Generator
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        ### Eval Discriminator
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        gen_loss += loss_G.item()
        disc_loss += loss_D.item()
        tqdm_bar.set_postfix(gen_loss=gen_loss/(batch_idx+1), disc_loss=disc_loss/(batch_idx+1))
        
        # Save image grid with upsampled inputs and SRGAN outputs
        if random.uniform(0,1)<0.1:
            # Thay đổi kích thước của ảnh với độ phóng đại 4 lần để bằng kích thước với imgs_hr
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            
            # Hiển thị 3 ảnh trên 1 hàng
            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1) # Ghép 3 ảnh lại
            
            save_image(img_grid, f"images/{batch_idx}.png", normalize=False)

    test_gen_losses.append(gen_loss/len(test_dataloader))
    test_disc_losses.append(disc_loss/len(test_dataloader))
    
    # Save model checkpoints
    if np.argmin(test_gen_losses) == len(test_gen_losses)-1:
        torch.save(generator.state_dict(), "saved_models/generator.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")


fig = go.Figure()
fig.add_trace(go.Scatter(x=train_counter, y=train_gen_losses, mode='lines', name='Train Generator Loss'))
fig.add_trace(go.Scatter(x=test_counter, y=test_gen_losses, marker_symbol='star-diamond', 
                         marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Generator Loss'))
fig.update_layout(
    width=1000,
    height=500,
    title="Train vs. Test Generator Loss",
    xaxis_title="Number of training examples seen",
    yaxis_title="Adversarial + Content Loss"),
fig.show()
