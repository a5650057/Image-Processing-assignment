import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_images(images, titles, cmap='gray', save_prefix=None):
    """顯示多張影像的輔助函式，並可選擇儲存影像"""
    plt.figure(figsize=(12,8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, (len(images)+1)//2, i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        # 儲存圖片
        if save_prefix is not None:
            # 將標題中的空格去除，以便成為檔名的一部分
            filename = f"{save_prefix}_{title.replace(' ','_')}.jpg"
            # 若為浮點影像需要正規化
            save_img = img
            if save_img.dtype != np.uint8:
                save_img_norm = cv2.normalize(save_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                save_img_norm = save_img
            cv2.imwrite(filename, save_img_norm)

    plt.tight_layout()
    plt.show()

def gaussian_kernel(size=3, sigma=1):
    """產生 Gaussian Kernel"""
    k = (size-1)//2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g = g / g.sum()
    return g

def gaussian_lowpass_filter(shape, cutoff):
    """產生 Gaussian 低通濾波器 (頻域使用)"""
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[:rows, :cols]
    dist2 = (x-ccol)**2 + (y-crow)**2
    mask = np.exp(-dist2/(2*(cutoff**2)))
    mask_2ch = np.zeros((rows, cols, 2), np.float32)
    mask_2ch[:,:,0] = mask
    mask_2ch[:,:,1] = mask
    return mask_2ch

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("請輸入功能號碼(1, 2, 3, 4 或 5)")
        sys.exit(1)

    func_num = sys.argv[1]

    # 若無存在，建立一個輸出資料夾
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if func_num == '1':
        # 功能1: 移除 Figure1 雜訊
        img1 = cv2.imread('Figure1.jpg', cv2.IMREAD_GRAYSCALE)
        avg_filtered = cv2.blur(img1, (3,3))
        median_filtered = cv2.medianBlur(img1, 3)

        # 顯示及儲存結果
        show_images(
            [img1, avg_filtered, median_filtered],
            ['Original Figure1', 'Average Filtered', 'Median Filtered'],
            save_prefix=os.path.join(output_dir, 'func1')
        )

    elif func_num == '2':
        # 功能2: 對 Figure2 進行銳化
        img2 = cv2.imread('Figure2.jpg', cv2.IMREAD_GRAYSCALE)
        # (a) Sobel
        sobel_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        sobel_mag = np.uint8(np.clip(sobel_mag,0,255))
        sharpened_sobel = cv2.addWeighted(img2, 1.0, sobel_mag, 0.5, 0)

        # (b) Fourier 高通銳化
        dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = img2.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols, 2), np.float32)
        r = 10  # 半徑
        y, x = np.ogrid[:rows, :cols]
        dist2 = (x - ccol)**2 + (y - crow)**2

        # 將距離中心小於半徑r的點視為低頻區域
        mask[dist2 <= r**2] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        sharpened_fourier = cv2.addWeighted(img2, 1.0, img_back, 0.5, 0)

        # 顯示及儲存結果
        show_images(
            [img2, sharpened_sobel, sharpened_fourier],
            ['Original Figure2', 'Sobel Sharpened', 'Fourier Sharpened (Added Back)'],
            save_prefix=os.path.join(output_dir, 'func2')
        )
        

    elif func_num == '3':
        # 功能3: 設計3x3高斯濾波器，對 Figure1 進行低通濾波
        img1 = cv2.imread('Figure1.jpg', cv2.IMREAD_GRAYSCALE)
        gaussian_3x3 = gaussian_kernel(10, sigma=1)
        gaussian_filtered = cv2.filter2D(img1, -1, gaussian_3x3)

        show_images(
            [img1, gaussian_filtered],
            ['Original Figure1', 'Gaussian (3x3) Filtered'],
            save_prefix=os.path.join(output_dir, 'func3')
        )

    elif func_num == '4':
        # 功能4: 使用與第3點對應的 Fourier 低通濾波器對 Figure1 進行平滑
        img1 = cv2.imread('Figure1.jpg', cv2.IMREAD_GRAYSCALE)
        dft1 = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft1_shift = np.fft.fftshift(dft1)

        lpf_mask = gaussian_lowpass_filter(img1.shape, cutoff=30)
        f_low = dft1_shift * lpf_mask
        f_ishift = np.fft.ifftshift(f_low)
        img_lowback = cv2.idft(f_ishift)
        img_lowback = cv2.magnitude(img_lowback[:,:,0], img_lowback[:,:,1])
        img_lowback = cv2.normalize(img_lowback, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        show_images(
            [img1, img_lowback],
            ['Original Figure1', 'Fourier Low-pass Filtered'],
            save_prefix=os.path.join(output_dir, 'func4')
        )
    else:
        print("無效的功能號碼！請輸入1、2、3、4。")
