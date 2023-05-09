# import cv2
# import numpy as np

# global img
# img =cv2.imread('./test_img.jpg')
# Y, X= img.shape[:2]
# print(X, Y)
# list_pst=[]

# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     global img
#     list_xy = []
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         list_xy.append(x)
#         list_xy.append(y)
#         print(list_xy)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (255, 255, 0), thickness=1)
#         cv2.imshow("original_img", img)
#         list_pst.append(list_xy)
#         if(len(list_pst)==4):
#             # 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
#             pts1 = np.float32(list_pst)
#             pts2 = np.float32([[0, 0], [X, 0], [0, Y], [X, Y]])

#             # 生成透视变换矩阵；进行透视变换
#             M = cv2.getPerspectiveTransform(pts1, pts2)
#             calibrated_img = cv2.warpPerspective(img, M, (int(img.shape[1]),int(img.shape[0])))

#             calibrated_img = calibrated_img[:1750,:]
#             print(calibrated_img.shape)
#             print(M)
#             img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))) 
#             cv2.imshow("original_img",img)
#             cv2.imshow("calibrated_img",cv2.resize(calibrated_img, (int(calibrated_img.shape[1]/4), int(calibrated_img.shape[0]/4))))


#             # cv2.imshow("result", dst)


# cv2.namedWindow("original_img")
# cv2.setMouseCallback("original_img", on_EVENT_LBUTTONDOWN)
# cv2.imshow("original_img", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()




import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_location(img):
    '''
    function: 记录标定物体的坐标位置。get location of standard blind path block, the upper edge and lower edge must parallel to standard image.
    param {img}
    return {None}
    '''    
    img[:, :, ::-1] # 是将BGR转化为RGB
    plt.plot()
    plt.imshow(img[:, :, ::-1])
    plt.title('img')
    plt.show()

def get_calibration_params(img, location_corners, block_num=1):
    '''
    function: 将记录的标定物体的坐标位置输入，进行逆透视变换。calibrate standard object and get transform matrix
    param {img, location_corners}
    return {img, transform_matrix}
    '''
    # 标定
    # jpg中正方形标定物的四个角点(左上、右上、左下、右下),与变换后矩阵位置
    standard_loc = np.float32(location_corners)
    for corner in standard_loc:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 5, (0,0,255), -1) # 画出标定点坐标
    
    l_bot = standard_loc[0]
    r_bot = standard_loc[1]
    l_top = standard_loc[2]
    r_top = standard_loc[3]
    # 标定物最长边
    standard_edge = (r_top[0] - l_top[0])
    center_x = l_top[0] + (r_top[0] - l_top[0])/2
    center_y = l_top[1] + (r_top[1] - l_bot[1])/2
    
    left = center_x - standard_edge/2
    right = center_x + standard_edge/2
    bot = center_y - standard_edge*block_num/2
    top = center_y + standard_edge*block_num/2
    img_loc = np.float32([[left, bot],[right, bot],[left,top],[right,top]])
    print(img_loc)

    # 生成透视变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(standard_loc, img_loc)
    
    return img, transform_matrix
    
def calibration(img, location_corners, block_num=1):
    '''
    function: calibrate the img by transform matrix.
    param {img, location_corners, block_num}
    return {None}
    '''
    img, transform_matrix = get_calibration_params(img, location_corners, block_num)
    # 逆透视变换
    calibrated_img = cv2.warpPerspective(img, transform_matrix, (int(img.shape[1]),int(img.shape[0]*block_num)))
    print((int(img.shape[1]),int(img.shape[0]*block_num)))
    # 裁剪图片到合适尺寸（该值需要自行根据自己的图像进行测试）
    # get_location(calibrated_img) # 获取尺寸
    calibrated_img = calibrated_img[:1750,:]
    print(calibrated_img.shape)
    print(transform_matrix)
    # write_matrix(transform_matrix)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))) 
    cv2.imshow("original_img",img)
    cv2.imshow("calibrated_img",cv2.resize(calibrated_img, (int(calibrated_img.shape[1]/4), int(calibrated_img.shape[0]/4))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_matrix(matrix, save_dir="./"):
    import os
    np.save(os.path.join(save_dir,"transform_matrix.npy"), matrix)

if __name__ == '__main__':
    # 需要哪个注释掉另一个函数即可
    img = cv2.imread('./image.jpg')
    # get_location(img)    # 利用matplotlib记录标定物四点坐标
    calibration(img, location_corners=[[311, 545], [827, 552], [212, 957], [956, 955]], block_num=3)
    # [[472, 88],[647, 89],[318, 679],[759, 679]]
    


