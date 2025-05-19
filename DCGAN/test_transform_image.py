from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import numpy as np

image_path = 'GANs/DCGAN/ID2201.jpg'
original_image = Image.open(image_path)
# array_img = np.array(original_image, np.uint8)
# print(array_img)

new_size = (256, 256)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(new_size, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Áp dụng transform for original image
transformed_image = transform(original_image)


# Chuyển đổi tensor trở lại dạng hình ảnh để hiển thị
def tensor_to_pil(tensor_image):
    """Chuyển đổi tensor thành hình ảnh PIL để hiển thị."""
    tensor_image = tensor_image.clone()
    # Normalize: Tensor = (Tensor + mean) / std => do đó khôi phục tensor gốc ta làm ngược lại
    # view() được sử dụng để thay đổi hình dạng (shape) của một tensor mà không thay đổi dữ liệu của nó: (3,) 
        # -> (3,1,1) để áp dụng boardcasting (RGB, height, width)
    tensor_image = tensor_image * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    # print(tensor_image)
    
    # convert tensor to image
    return transforms.ToPILImage()(tensor_image)

# Hiển thị hình ảnh gốc và hình ảnh đã biến đổi
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Hình ảnh gốc
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Hình ảnh đã biến đổi
transformed_image_pil = tensor_to_pil(transformed_image)
axes[1].imshow(transformed_image_pil)
axes[1].set_title("Transformed Image")
axes[1].axis('off')

plt.show()