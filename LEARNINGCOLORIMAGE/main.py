import numpy as np
import scipy.ndimage as ndimage
from skimage import color
import os
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import cv2

def extract_patches(high_res_images, low_res_images, r, m):
    high_res_patches_Y, high_res_patches_I, high_res_patches_Q = [], [], []
    low_res_patches_Y, low_res_patches_I, low_res_patches_Q = [], [], []

    for h_img, l_img in zip(high_res_images, low_res_images):
        h_yiq = color.rgb2yiq(h_img)
        l_yiq = color.rgb2yiq(l_img)
        h_height, h_width, _ = h_yiq.shape
        l_height, l_width, _ = l_yiq.shape

        for y in range(0, h_height - r + 1, r):
            for x in range(0, h_width - r + 1, r):
                l_y = y // r
                l_x = x // r
                if l_y + m // 2 >= l_height or l_y - m // 2 < 0 or \
                   l_x + m // 2 >= l_width or l_x - m // 2 < 0:
                    continue

                high_res_patches_Y.append(h_yiq[y:y+r, x:x+r, 0].flatten())
                high_res_patches_I.append(h_yiq[y:y+r, x:x+r, 1].flatten())
                high_res_patches_Q.append(h_yiq[y:y+r, x:x+r, 2].flatten())

                half_m = m // 2
                low_res_patches_Y.append(l_yiq[l_y-half_m:l_y+half_m+1, l_x-half_m:l_x+half_m+1, 0].flatten())
                low_res_patches_I.append(l_yiq[l_y-half_m:l_y+half_m+1, l_x-half_m:l_x+half_m+1, 1].flatten())
                low_res_patches_Q.append(l_yiq[l_y-half_m:l_y+half_m+1, l_x-half_m:l_x+half_m+1, 2].flatten())

    return (np.array(high_res_patches_Y), np.array(high_res_patches_I),
            np.array(high_res_patches_Q), np.array(low_res_patches_Y),
            np.array(low_res_patches_I), np.array(low_res_patches_Q))

def variational_learning(X, Y, a_alpha0):
    N, D = X.shape
    _, Q = Y.shape

    a_alpha = a_alpha0 + 0.5
    b_alpha = np.ones((D, Q)) * 1e-6
    a_beta = 1e-6 + N * D / 2
    b_beta = 1e-6
    beta = a_beta / b_beta
    alpha = np.ones((D, Q))
    M = np.zeros((D, Q))
    Sigma = [np.eye(Q) for _ in range(D)]

    YY = Y.T @ Y  

    prev_M = M.copy()
    delta = 1.0
    max_iter = 100
    iter_count = 0

    while delta > 1e-6 and iter_count < max_iter:
        for d in range(D):
            S_inv = np.diag(alpha[d, :]) + beta * YY
            Sigma[d] = np.linalg.inv(S_inv)
            M[d, :] = beta * Sigma[d] @ Y.T @ X[:, d]

        # Optimized trace computation
        Sigma_sum = np.sum(Sigma, axis=0)  # (121, 121)
        trace_term = 0
        for i in range(N):  # Duyệt qua các hàng của Y
            y_row = Y[i, :]  # (121,)
            trace_term += y_row @ (Sigma_sum @ y_row)  # Scalar

        b_beta = 1e-6 + 0.5 * (np.sum((X.T - M @ Y.T) ** 2) + trace_term)
        beta = a_beta / b_beta

        for d in range(D):
            for q in range(Q):
                b_alpha[d, q] = 1e-6 + 0.5 * (M[d, q]**2 + Sigma[d][q, q])
        alpha = a_alpha / b_alpha

        alpha[alpha > np.exp(20)] = np.inf
        M[alpha > np.inf] = 0

        delta = np.linalg.norm(M - prev_M, 'fro') / (np.linalg.norm(M, 'fro') + 1e-10)
        prev_M = M.copy()
        iter_count += 1

    print(f"Variational Learning Converged after {iter_count} iterations with delta={delta:.8f}")
    return M

def expand_image(low_res_img, M_Y, M_I, M_Q, r, m):
    low_res_yiq = color.rgb2yiq(low_res_img)
    height, width, _ = low_res_yiq.shape
    expanded_height, expanded_width = height * r, width * r

    expanded_Y = np.zeros((expanded_height, expanded_width))
    expanded_I = np.zeros((expanded_height, expanded_width))
    expanded_Q = np.zeros((expanded_height, expanded_width))

    pad_width = m // 2
    padded_Y = np.pad(low_res_yiq[:,:,0], pad_width, mode='edge')
    padded_I = np.pad(low_res_yiq[:,:,1], pad_width, mode='edge')
    padded_Q = np.pad(low_res_yiq[:,:,2], pad_width, mode='edge')

    for y in tqdm(range(height)):
        for x in range(width):
            y_patch_Y = padded_Y[y:y+m, x:x+m].flatten()
            y_patch_I = padded_I[y:y+m, x:x+m].flatten()
            y_patch_Q = padded_Q[y:y+m, x:x+m].flatten()

            h_y, h_x = y * r, x * r
            expanded_Y[h_y:h_y+r, h_x:h_x+r] = (M_Y @ y_patch_Y).reshape(r, r)
            expanded_I[h_y:h_y+r, h_x:h_x+r] = (M_I @ y_patch_I).reshape(r, r)
            expanded_Q[h_y:h_y+r, h_x:h_x+r] = (M_Q @ y_patch_Q).reshape(r, r)

    expanded_yiq = np.stack([expanded_Y, expanded_I, expanded_Q], axis=2)
    expanded_rgb = color.yiq2rgb(expanded_yiq)
    return np.clip(expanded_rgb, 0, 1)

def downsample_image(img, factor=2):
    low_res_size = (img.shape[1] // factor, img.shape[0] // factor) 
    low_res_img = cv2.resize(img, low_res_size, interpolation=cv2.INTER_CUBIC)
    return low_res_img

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

def load_sipi_images(directory, file_numbers):

    images = []
    for filename  in file_numbers:
        filepath  = os.path.join(directory, filename)
        if os.path.exists(filepath):
            img = plt.imread(filepath) / 255.0  # Normalize
            images.append(img)
        else:
            print(f"File {filepath} not found!")
    return images

if __name__ == "__main__":
    r = 2
    m = 11
    a_alpha0 = 20

    sipi_dir = "D:/Code/Python/XLA/data" 
    train_files = [f"4.1.0{i}.tiff" if i < 10 else f"4.1.{i}.tiff" for i in range(1, 8)]
    test_files = [f"4.2.0{i}.tiff" if i < 10 else f"4.2.{i}.tiff" for i in range(1, 8) if i not in [2]]

    train_high_res = load_sipi_images(sipi_dir, train_files)
    train_low_res = [downsample_image(img, factor=r) for img in train_high_res]
    test_high_res = load_sipi_images(sipi_dir, test_files)
    test_low_res = [downsample_image(img, factor=r) for img in test_high_res]


    # Extract patches
    X_Y, X_I, X_Q, Y_Y, Y_I, Y_Q = extract_patches(train_high_res, train_low_res, r, m)

    # Learn filters
    M_Y = variational_learning(X_Y, Y_Y, a_alpha0)
    M_I = variational_learning(X_I, Y_I, a_alpha0)
    M_Q = variational_learning(X_Q, Y_Q, a_alpha0)
    
    np.save("D:/Code/Python/XLA/matrix/filter_Y.npy", M_Y)
    np.save("D:/Code/Python/XLA/matrix/filter_I.npy", M_I)
    np.save("D:/Code/Python/XLA/matrix/filter_Q.npy", M_Q)

    #Expand and evaluate (first test image)
    expanded_img = expand_image(test_low_res[2], M_Y, M_I, M_Q, r, m)
    expanded_img_uint8 = (expanded_img * 255).astype(np.uint8)
    imageio.imwrite("D:/Code/Python/XLA/res/expanded_image.png", expanded_img_uint8)
    
    test_low_res_uint8 = (test_low_res[2] * 255).astype(np.uint8) 
    imageio.imwrite("D:/Code/Python/XLA/res/low_res_image.png", test_low_res_uint8)
    # Calculate PSNR
    psnr = calculate_psnr(test_high_res[2], expanded_img)

    print(f"PSNR: {psnr:.2f} dB")
