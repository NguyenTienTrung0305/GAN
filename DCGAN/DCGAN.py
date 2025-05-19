
# Tài liệu đọc để hiểu hơn: https://nttuan8.com/bai-2-deep-convolutional-gan-dcgan/
########### Deep convolution GAN: tạo ra một ảnh giả giống như thật từ latent vector 

# Cách thức hoạt động của DCGAN:
# Latent Space (z-vector):
    # DCGAN nhận vào một vector ngẫu nhiên từ một không gian latent (gọi là z-vector), và không gian này có 
        # thể được giả định là phân phối chuẩn (Gaussian distribution).
    # Latent vector này có kích thước cố định và thường có chiều nhỏ hơn so với ảnh đầu vào, ví dụ như một 
        # vector 100 chiều. Vector này không phải là đại diện nén của một hình ảnh cụ thể mà là một tập hợp 
        # các giá trị ngẫu nhiên (thường phân phối chuẩn), từ đó Generator của DCGAN sẽ sinh ra một hình ảnh.

# Generator trong DCGAN:
    # Generator là một mạng nơ-ron học sâu (deep neural network), có nhiệm vụ chuyển đổi latent vector (z) thành 
        # một hình ảnh giả.
    # Mạng này sử dụng các lớp up-sampling (hoặc transposed convolution) để tăng kích thước của vector latent 
        # và tạo ra hình ảnh có kích thước mong muốn (ví dụ 64x64, 128x128, v.v).
# Discriminator trong DCGAN:
    # Discriminator có nhiệm vụ phân biệt giữa các hình ảnh thực và hình ảnh giả. Nó nhận vào hình ảnh và trả 
        # về xác suất hình ảnh đó là thật (real) hay giả (fake).




# Step 1: Importing the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Defining a Transform 
# Define a basic transform 
transform = transforms.Compose([
    transforms.ToTensor(), # chuyển đổi type image to Tensor, chuẩn hóa các giá trị pixel từ khoảng [0, 255] sang [0, 1]
    
    # chuyển các giá trị pixel từ [0, 1] thành [-1, 1] bằng công thức tensor = (tensor - mean)/ std
    # mean = std = (0.5, 0.5, 0.5), 3D vì đây là 3 màu RGB
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 3: Loading the Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # 50000 ảnh 
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True) # tạo batch và shuffle data (1563 batch)

# Step 4: Defining parameters to be used in later processes
# A Generative Adversarial Network (GAN) is used with specified hyperparameters.
    # latent_dim: là số chiều của không gian tiềm ẩn mà dữ liệu được mã hóa vào 
        # (generator tạo ra dữ liệu từ một không gian tiềm ẩn ngẫu nhiên. Latent dimension ở đây là 
        # chiều của vector ngẫu nhiên mà generator nhận vào.)
    # lr is the optimizer’s learning rate.
    # The coefficients for the Adam optimizer are beta1 and beta2. To find the total number of training epochs, use num_epochs.
# Hyperparameters
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10

# Step 5: Defining a Utility Class to Build the Generator
# Mục tiêu là từ latent vector tạo ra được 1 images
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
    
        self.model = nn.Sequential(
            # latent_dim là số chiều của không gian tiềm ẩn (latent space), đây là số lượng đơn vị đầu vào của lớp tuyến tính này.
            # 128 * 8 * 8 là số lượng đơn vị đầu ra
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),  # 128 * 8 * 8 -> batch_size * 128 * 8 * 8
            
            #### Transpose Convolution với stride = 2 <=> Upsample(scale_factor = 2) + Conv2d(kernel_size = 3)
            # Tăng kích thước weight và height lên gấp 2: batch_size * 128 * 8 * 8 -> batch_size * 128 * 16 * 16
            # input của nn.Upsample: (N, C, H, W), với C = số kênh đầu vào.
            # output của nn.Upsample: (N, C, Hr, Wr), với r là scale_factor.
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # batch_size * 128 * 16 * 16
            
            # chuẩn hóa output của lớp trước
            # Với momentum=0.78, lớp Batch Normalization sẽ dựa nhiều hơn vào giá trị trung bình 
                # và độ lệch chuẩn của batch hiện tại (78%) so với giá trị của các batch trước đó (22%).
            # Nếu momentum quá cao -> không ổn định do phụ thuộc nhiều vào sự biến động của batch hiện tại
            # Nếu momentum quá thấp -> giá trị trung bình và độ lệch chuẩn di chuyển sẽ thay đổi rất chậm, 
                # có thể không phản ánh kịp thời sự thay đổi của dữ liệu
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1), # batch_size * 3 * 32 * 32
            nn.Tanh() # scale pixel [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        return img


# Step 6: Defining a Utility Class to Build the Discriminator
# Mục tiêu phân biệt real images với fake_images được tạo gởi Generator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # batch_size * 3 * 32 * 32 -> batch_size * 32 * 16 * 16
            # LeakyReLU giúp cải thiện khả năng phân biệt ảnh bằng cách tránh việc các output toàn bằng 0 (do LeakyReLU cho 1 lượng nhỏ giá trị âm đi qua) 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # batch_size * 64 * 8 * 8 
            # nn.ZeroPad2d((0, 1, 0, 1)) trong PyTorch là một lớp padding được sử dụng để thêm các pixel 
            # có giá trị bằng 0 xung quanh ảnh đầu vào. ( để đảm bảo tránh mất thông tin ở các cạnh)
                # 0 (Padding bên trái): Không thêm padding ở cạnh trái của ảnh.
                # 1 (Padding bên phải): Thêm 1 pixel padding ở cạnh phải của ảnh.
                # 0 (Padding phía trên): Không thêm padding ở cạnh trên của ảnh.
                # 1 (Padding phía dưới): Thêm 1 pixel padding ở cạnh dưới của ảnh.
            nn.ZeroPad2d((0, 1, 0, 1)), # batch_size * 64 * 9 * 9 
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # batch_size * 128 * 5 * 5 
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # batch_size * 256 * 5 * 5 
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# Step 7: Building the Generative Adversarial Network
# Define the generator and discriminator
# Initialize generator and discriminator
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Loss function
adversarial_loss = nn.BCELoss()
# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2)) # Binary cross entropy
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))


# SAVE MODEL
def save_model(path, file):
    torch.save(generator.state_dict(), '%s%s_generator.pth' %(path, file))
    torch.save(discriminator.state_dict(), '%s%s_discriminator.pth' %(path, file))
    


# Step 8: Training the Generative Adversarial Network
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # get images (have datatype is tensor so dont need convert to tensor)
        real_images = batch[0]
        
        # Gán nhãn cho images
        # size(0): lấy size của first dimension (batch_size)
        valid = torch.ones(real_images.size(0), 1) # Tạo tensor có size (batchsize, 1) với tất cả phần tử là 1
        fake = torch.zeros(real_images.size(0), 1) # Tạo tensor có size (batchsize, 1) với tất cả phần tử là 0
        
        # Configure input
        real_images = real_images
        
        # Sample noise as generator input: [batchsize, latent_dim]
        # Noise cho Generator thường được sinh ra từ normal distribution hoặc uniform distribution.
        z = torch.rand(real_images.size(0), latent_dim)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Generate a batch of images
        fake_images = generator(z)
        
        # Measure discriminator's ability to classify real and fake images
        real_loss = adversarial_loss(discriminator(real_images), valid) # mục tiêu discriminator(real_images) dần về 1
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake) # # mục tiêu discriminator(fake_images) dần về 0
        # target: min d_loss
        d_loss = (real_loss + fake_loss) / 2
        
        # Backward pass and optimize
        d_loss.backward()
        optimizer_D.step()
        
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Generate a batch of images
        gen_images = generator(z)
        g_loss = adversarial_loss(discriminator(gen_images), valid) # mục tiêu discriminator(gen_images) càng gần 1 -> ảnh càng giống nhất -> g_loss càng nhỏ
        g_loss.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Progress Monitoring
        # ---------------------
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
            
    
    # Save generated images for every epoch
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim)
            generated = generator(z).detach().cpu() # di chuyển ảnh về cpu vì plt không làm việc với gpu
            grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
            
            # (1, 2, 0): Chuyển đổi tensor từ định dạng (channels, height, width) sang (height, width, channels) vì plt chỉ nhận data có size là (height, width, channels) .
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()
        
        save_model("GANs/DCGAN/Model", "model")
        
            
