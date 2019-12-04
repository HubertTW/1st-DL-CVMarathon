# 1st-DL-CVMarathon


## day01
b,g,r=cv2.split(img)

## day02
hue(圓環)/saturation(半徑)/brightess(深淺)
* HSB(HSV):
saturation:白色到選擇的hue(0~100)
lightness:100=任何可能的顏色

* HSL(Lightness):
saturation:灰色到選擇的hue
lightness:
100=白色
* LAB
L = Lighness 
以0~100 決定明亮度
數值由⼩小到⼤大，由⿊黑到⽩白
A 以-128~127 
代表顏⾊色對立的維度
數值由⼩小到⼤大，由綠到紅
B 以-128~127 
代表顏⾊色對立的維度
數值由⼩小到⼤大，由藍藍到黃

cv2.cvtColor(img,cv2.BGR2HSV)

## day03
* 調整飽和度
```
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
change_percentage = 0.2

# 針對飽和度的值做改變，超過界線 0~1 的都會 bound
# 在 HSV color space 減少飽和度
img_hsv_down=img_hsv.astype('float32')
img_hsv_down[::1]=img_hsv_down[::1]/255-change_percentage
img_hsv_down[img_hsv_down<0]=0
img_hsv_down=img_hsv_down*255
img_hsv_down=img_hsv_down.astype('uint8')
```
* 直方圖均衡
cv2.equalizeHist(img)
* g(x)=alpha*f(x)+beta
cv2.convertScaleAbs(img,alpha=,beta=)
alpha(1.0~3.0):對比度
beta(0~100):明亮度

## day04
* flip:cv2.flip(img,0)#垂直翻轉
* scale:cv2.resize(img,new_img,fx,fy,interpolation=cv2.INTER_LINER)
default:Bilinear Interpolation
建議縮⼩用 INTER_AREA
建議放⼤用 INTER_CUBIC (slow)或INTER_LINEAR
* Translation Transformation:cv2.warpAffine(img,Matrix,(col,row))


## day06
* Affine Transformation
  1.  共線不變性
  2.  比例不變性
matrix=cv2.getAffineTransform(point1,point2)
cv2.warpAffine(img,matrix,(cols,rows))

## day07 
* 齊次座標系
* transformation matrix
![](https://i.imgur.com/9362HJW.png)

## day08
* guassian filter
  avg. filter is more blurred than guassian
  increasing sigma leads to more blurred
* sobel filter
![](https://i.imgur.com/gygvHNE.jpg)
http://silverwind1982.pixnet.net/blog/post/243360385

