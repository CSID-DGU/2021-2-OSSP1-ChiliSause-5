import cv2

def EdgeDetect():

    imageFile = "test_data/img5.png"

    img = cv2.imread(imageFile)

    # res1 = cv2.Canny(img, 50, 200)
    # res2 = cv2.Canny(img, 100, 200)
    res3 = cv2.Canny(img, 150, 200)

    res3 = 255- res3
    cv2.imwrite('experiment/img5_1.png', res3)
    # cv2.imshow("ORIGIN",img)
    # cv2.imshow("res1",res1)
    # cv2.imshow("res2",res2)
    # cv2.imshow("res3",res3)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

EdgeDetect()
