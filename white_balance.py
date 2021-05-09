# Goal: 調整樹脂圖片白平衡至Lab（79, 0, 0）
import sys
import cv2
import numpy as np

# initiation
wbt = [196, 196, 196] # L*a*b* = (79, 0, 0)
target_region = [0.8, 1, 0.27, 0.37] # length begin, length end, width begin, width end of gray card

def crop_gray_region(img):
        h, w, d = img.shape
        # initial point of cropping an image
        x = int(np.floor(h*target_region[0]))
        y = int(np.floor(w*target_region[2]))
        # height and width of cropped image
        h_crop = int(np.floor(h*(target_region[1]-target_region[0])))
        w_crop = int(np.floor(w*(target_region[3]-target_region[2])))
        # crop the image
        img_crop = img[x:x+h_crop, y:y+w_crop]

        #cv2.imshow("gray card", img_crop)
        #cv2.waitKey(0)

        return img_crop
    
def compute(img):
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Bmean.append(np.mean(img[:,:,0]))
    per_image_Gmean.append(np.mean(img[:,:,1]))
    per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return B_mean, G_mean, R_mean

# white balance algorithm
def image_white_balance(img):
    def detection(img):
        """偏色值"""
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b, M_a, M_b = 0, 0, 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
        D = np.sqrt((np.square(d_a) + np.square(d_b)))
 
        for i in range(m):
            for j in range(n):
                M_a = np.abs(a[i][j] - d_a - 128) + M_a
                M_b = np.abs(b[i][j] - d_b - 128) + M_b
 
        M_a, M_b = M_a / (m * n), M_b / (m * n)
        M = np.sqrt((np.square(M_a) + np.square(M_b)))
        k = D / M
        print('偏色值:%f' % k)
        return
    
    b, g, r = cv2.split(img)
    print('Image scale: ', img.shape)
    m, n = b.shape
    #detection(img)
 
    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
 
    I_r_2 = (r.astype(np.float32) ** 2).astype(np.float32)
    I_b_2 = (b.astype(np.float32) ** 2).astype(np.float32)
    sum_I_r_2 = I_r_2.sum()
    sum_I_b_2 = I_b_2.sum()
    sum_I_g = g.sum()
    sum_I_r = r.sum()
    sum_I_b = b.sum()
 
    max_I_r = r.max()
    max_I_g = g.max()
    max_I_b = b.max()
    max_I_r_2 = I_r_2.max()
    max_I_b_2 = I_b_2.max()
 
    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    # print('ub, vb, ur, vr ', u_b, v_b, u_r, v_r)
 
    b_point = u_b * (b.astype(np.float32) ** 2) + v_b * b.astype(np.float32)
    r_point = u_r * (r.astype(np.float32) ** 2) + v_r * r.astype(np.float32)
 
    b_point[b_point > 255] = 255
    b_point[b_point < 0] = 0
    b = b_point.astype(np.uint8)
 
    r_point[r_point > 255] = 255
    r_point[r_point < 0] = 0
    r = r_point.astype(np.uint8)
    
    img_crop4 = crop_gray_region(cv2.merge([b, g, r]))
    B, G, R= compute(img_crop4)
    # post-processing
    b_shift = np.array(np.round(196-B))
    g_shift = np.array(np.round(196-G))
    r_shift = np.array(np.round(196-R))
    b_sh = b+ b_shift
    g_sh = g+ g_shift
    r_sh = r+ r_shift

    b_sh[b_sh > 255] = 255
    b_sh[b_sh < 0] = 0
    g_sh[g_sh > 255] = 255
    g_sh[g_sh < 0] = 0
    r_sh[r_sh > 255] = 255
    r_sh[r_sh < 0] = 0
    b_sh = b_sh.astype(np.uint8)
    g_sh = g_sh.astype(np.uint8)
    r_sh = r_sh.astype(np.uint8)
    img_wbt = cv2.merge([b_sh, g_sh, r_sh])
 
    return img_wbt

# color test for gray card
def color_test(img):  
    def rgb2lab(inputColor):
        num = 0
        RGB = [0, 0, 0]

        for value in inputColor :
            value = float(value) / 255

            if value > 0.04045 :
                value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
            else :
                value = value / 12.92

            RGB[num] = value * 100
            num = num + 1

        XYZ = [0, 0, 0,]

        X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
        Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
        Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
        XYZ[ 0 ] = round( X, 4 )
        XYZ[ 1 ] = round( Y, 4 )
        XYZ[ 2 ] = round( Z, 4 )

        XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
        XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
        XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

        num = 0
        for value in XYZ :

            if value > 0.008856 :
                value = value ** ( 0.3333333333333333 )
            else :
                value = ( 7.787 * value ) + ( 16 / 116 )

            XYZ[num] = value
            num = num + 1

        Lab = [0, 0, 0]

        L = ( 116 * XYZ[ 1 ] ) - 16
        a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
        b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

        Lab [ 0 ] = round( L, 4 )
        Lab [ 1 ] = round( a, 4 )
        Lab [ 2 ] = round( b, 4 )

        return Lab
    img_crop = crop_gray_region(img)
    B, G, R= compute(img_crop)
    print(B, G ,R)
    Lab = rgb2lab([R,G,B])
    print(Lab)
    return Lab

if __name__ == "__main__":
    img = cv2.imread('black.jpg')
    img_wbt = image_white_balance(img)
    cv2.imwrite('image_black0.jpg', img_wbt)
    # color_test(img_wbt) # 196.18949650498132 196.12884074895652 195.51157742402316 # [79.1578, -0.1977, -0.114]