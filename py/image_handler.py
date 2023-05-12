import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_images(images,titles=None, num_cols=None,scale=3, normalize=False):

    '''
    一个窗口中绘制多张图像:
    Args:
        images:可以为一张图像(不要放在列表中),也可以为一个图像列表
        titles:图像对应标题、
        num_cols:每行最多显示多少张图像scale:用于调整图窗大小
        normalize:显示灰度图时是否进行灰度归一化
    '''
        #加了下面2行后可以显示中文标题
    # plt.rcParams[ 'font.sans-serif']=[ 'simHei ']
    # plt.rcParams[ 'axes.unicode_minus'] = False
    #单张图片显示
    if not isinstance(images, list):
        if not isinstance(scale, tuple):
            scale = (scale, scale * 1.5)

        plt.figure(figsize=(scale[1], scale[0]))
        img = images
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为RGR通道,需要转换一下
            B, G, R = cv.split(img)
            img = cv.merge([R, G , B])
            plt.imshow(img)
        elif len(img.shape) == 2:
            #pYplLot显示灰度需要加一个参数if
            if normalize:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img, cmap='gray' ,vmin = 0,vmax = 255)
        else:
            raise TypeError("Invalid shape "+
                            str(img.shape)+ "of image data" )
        if titles is not None:
            plt.title(titles, y=-0.15)
        plt.axis("off")
        plt.show()
        return

    # 多张图片显示

    if not isinstance(scale, tuple):
        scale = (scale, scale)

    num_imgs = len(images)
    if num_cols is None:
        num_cols = int(np.ceil((np.sqrt(num_imgs))))
    num_rows = (num_imgs - 1) // num_cols + 1

    idx = list(range(num_imgs))
    _, figs = plt.subplots(num_rows, num_cols,
                           figsize=(scale[1] * num_cols, scale[0] * num_rows))

    for f, i, img in zip(figs.flat, idx, images):
        if len(img.shape) == 3:
        # opencv库中函数生成的图像为BGR通道,零要猜换一下
            B,G,R=cv.split(img)
            img = cv.merge([R,G,B])
            f.imshow(img)

        elif len(img.shape) == 2:
            # pYPLot显示灰度需要加一个参数

            if normalize:
                f.imshow(img, cmap='gray')
            else:
                f.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                        str(img.shape) + "of image data")

        if titles is not None:
            f.set_title(titles[i], y=-0.15)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        # 将不显示图像的fig移除,不然会显示多余的窗口
    if len(figs.shape) == 1:
        figs = figs.reshape(-1, figs.shape[0])
    for i in range(num_rows * num_cols - num_imgs):
        figs[num_rows - 1, num_imgs % num_cols + i].remove()
    plt.show()

