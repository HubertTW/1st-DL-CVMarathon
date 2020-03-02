# Computer Vision 50 Marathon

## day01
slice image into three channels
```
b,g,r=cv2.split(img)
```

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
```
cv2.cvtColor(img,cv2.BGR2HSV)
```

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
* flip:

```
cv2.flip(img,0)#垂直翻轉
```
* scale:
 ```
cv2.resize(img,new_img,fx,fy,interpolation=cv2.INTER_LINER)
```
default:Bilinear Interpolation
建議縮⼩用 INTER_AREA
建議放⼤用 INTER_CUBIC (slow)或INTER_LINEAR
* Translation Transformation:
```
cv2.warpAffine(img,Matrix,(col,row))
```

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

ref:
1. [空間濾波](http://ccy.dd.ncu.edu.tw/~chen/course/vision/ch5/ch5.htm)

## day09
Scale Invariant Feature Transform(SIFT)
![](https://i.imgur.com/3Z4kjIp.png)
ref:
1. [scale space&image resolution](https://gwansiu.com/2017/08/24/Scale-Space-Resolution/)
粗圖像:大多低頻信息
細圖像:高低頻信息皆有
粗圖像包含於細圖像
一張圖片的scale space為
3. 

## day10 
Feature Matching
* L2 norm:計算兩點距離 若小於threshold則視為相同
* scale space
 
## day11
CNN:
* kernel size:Kernel ⼤小與其Receptive field有關，Receptive field 直觀來說就是Kernel 提取資訊的尺度，現在普遍流⾏的⽅式是選⽤不同⼤小的 Kernel對圖像做卷積後，再把輸出的Feature maps合併或平均。並且都為odd(3x3,5x5...)能保有中心點
* FC缺點:
    1. 會攤平圖片計算將失去空間資訊
    2. 參數量大 計算慢


* kernel的channel個數會與input image的channel數相對應
* Cov輸出的相片的channel數=kernel的個數
* 每個filter的channel會對應到input的channel
* to get numbers of parameters:
1. cnn:(filter的長x寬xfilter channel個數+1)*filter個數
2. FC:input個數*神經元個數
[CNN計算參數量](https://www.brilliantcode.net/1646/convolutional-neural-networks-3-calculate-number-of-parameters/)

## day12

* padding='valid'
  outputsize=(N-F)/Strides-1
  N:Image size
  F:Filter size
  無條件捨去
* padding='same'
  outputsize=N/Strides
  無條件進位

ref:
1. [卷積神經網路(Convolutional neural network, CNN):卷積計算中的步伐(stride)和填充(padding)](https://medium.com/@chih.sheng.huang821/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-%E5%8D%B7%E7%A9%8D%E8%A8%88%E7%AE%97%E4%B8%AD%E7%9A%84%E6%AD%A5%E4%BC%90-stride-%E5%92%8C%E5%A1%AB%E5%85%85-padding-94449e638e82)

## day13
pooling:
* maxpooling:
  1. 保留重要特徵,屬於一種subsampling
  2. 將低overfitting
  3. 降低運算量
  ![](https://i.imgur.com/csl4iDb.jpg)
  ![](https://i.imgur.com/CE2YK8j.jpg)
 ![](https://i.imgur.com/kIZtL9d.png)


* averagepooling:較為平滑
![](https://i.imgur.com/7oJQpKa.jpg)

ref:
1. [CNN詳解](https://kknews.cc/zh-tw/tech/vvx2qeq.html)


## day14
Batch Normalization:
基於我們希望讓input的數值範圍不要過大,一般我們會作feature scaling(讓input數值在range(0,1)),在神經網路中我們會對每一層的input都作feature scaling,但這樣每次都要重新計算其標準差與平均數很費時,因此我們使用batch normalization
z=gamma*z+beta

pros:
1. solve gradient vanish
2. normalization
3. 使分佈更穩定 加速收斂
4. because of less covariate shift,we can use larger learning rate
5. learning is less infected by normalization

ref:
1. [batch_normalization](http://violin-tao.blogspot.com/2018/02/ml-batch-normalization.html)

## day15
how to arrange cov/maxpooling/bn/activation function?
paper:
https://arxiv.org/pdf/1603.05027.pdf

cifar10範例:
https://keras.io/examples/cifar10_cnn/

* global averaging pooling(GAP)
1. 降低參數量
2. 

## day16
* ImageDataGenerator()
two steps of standardlization:
featurewise_center=以每張feature map 為單位 將平均值設為0
featurewise_std_normalization=以每張feature map為單位 除以標準差

zca_whitening:Boolean，透過ZCA取出重要特徵
rotation_range：整數值，控制隨機旋轉⾓角度
width_shift_range：「浮點、整數、⼀一維數」，圖像寬度上隨機偏移值
height_shift_range：「浮點、整數、⼀一維數」，圖像⾼高度上隨機偏移值
shear_range：浮點數，裁切範圍
zoom_range：浮點數或範圍，隨機縮放比例例
horizontal_flip:Boolean，隨機⽔水平翻轉
vertical_flip:Boolean，隨機垂直翻轉
rescale:數值，縮放比例例
dtype：輸出資料型態

## day17
![](https://i.imgur.com/Bb9bzN2.png)


---

![](https://i.imgur.com/trbgtet.png)

* [梯度消失與梯度爆炸](https://blog.csdn.net/qq_25737169/article/details/78847691)

* Floating point operation per second(FLOPS):計算模型每秒操作浮點數的次數
## day18
VGG16&VGG19

## day19
inception module:
我們通常會面臨到要如何選擇kernel_size的問題,而有一個想法是我們就讓模型自己train多個kernel來知道何者較佳,但如此一來會增大參數量因此我們就在這些kernel前先放一個1*1的kernel來減少FLOPS
* 為何使用1*1 kernel 
  
![](https://i.imgur.com/JfdcQuc.png)


ref: 
1. [inception module](https://medium.com/@chih.sheng.huang821/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-1-1%E5%8D%B7%E7%A9%8D%E8%A8%88%E7%AE%97%E5%9C%A8%E5%81%9A%E4%BB%80%E9%BA%BC-7d7ebfe34b8)


## day20 
* gradient vanishing:當我們在作BP的時候會發現到需要將每一層的sigmoid function的梯度(微分)作相乘,而sigmoid function的微分為[0,0.25]這會使得當要訓練比較前面的weight(因為是使用bp所以是從ouput開始計算較前面的weight會作更長的運算)的侯後會因相乘太多項的梯度而得其值趨近於零使weight難以更新.
  sol:
     1. Relu:微分值為1
     2. residual network
     3. batch normalization
![](https://i.imgur.com/FD7KI60.png)
![](https://i.imgur.com/vOwz3Vp.png)
* gradient exploding

* degradition:with the network depth increasing,the accuracu gets saturated

how to solve above problems?
the ResNet is solution
h(x)=x+f(x)
f(x):residual
if the 18 layers is the optimal sol but we don't know. at first we train 34 layers and the 16 layers are redundant so we design that the input is same as ouput in the 16 layers called "identity mapping".

ref: 
1. [反傳遞簡單理解](https://www.brilliantcode.net/1326/backpropagation-1-gradient-descent-chain-rule/)
2. [ResNet簡說](https://www.itread01.com/content/1544962578.html)


## day21 transfer learning

to conserve time,we'd like to use trained model to train a new model in similar task or domain.
the data used in trained model is source data
the data used in new model is target data 
however,the target data usually is usually little.
there are some transfer learning method can be adopted:
1. fine tuning:
training a model by source data, then fine-tune the model by target data 
being care for overfitting
2. consevative learning:
at first,we use source data to train A model and use target data to train B model,we hope their ouput and parameters are as close as possible they could.Therefore,we set some regularizations to achive that goal.
3. multitask learing:
4. domain-adversarial training
5. zero shot learing  

ref:
1. [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)  




## day23 object dection 
* one stage:
  directly doing classification && regression
* two stage:
  find the region proposal(many bounding box contain the possible object) first then do the classification
  evolve with speed:
  1. R-CNN
     selective search-->proposal-->cnn-->svm-->regresor
  3. Fast R-CNN
     selective search-->proposal-->cnn(feed whole img)-->softmax&&regressor
  5. Faster R-CNN
     RPN-->proposal
* region proposal:
  two main algorithms for finding region proposal
  1. selective search
     sliding window:use the box(many sizes) to scan all of img and feed into conv to detect the object.However,it spends lots of computational cost. 
  2. RPN 
 
 ref:
 1. [seclective search實作&解析](https://blog.gtwang.org/programming/selective-search-for-object-detection/)

## day24 YOLOV1
![](https://i.imgur.com/VrF6SZm.png)
* YOLOV1
![](https://i.imgur.com/4hy35DK.png)

img-->CNN-->use **defaut anchor box** to make every pixel generate "2" bounding box-->classification/regressor/confidence 
* Single Shot Multibox Dectector (SSD)
![](https://i.imgur.com/BVnsg0G.png)
* Retina Net 


## day25  Intersection-over-union ( IOU ) 


## day26 RPN

![](https://i.imgur.com/Tf5Y9Cd.png)

![](https://i.imgur.com/jTNzGWi.png)


1. Classifier determines the probability of a proposal having the target object.
2. Regression regresses the coordinates of the proposals 
* aspect ratio = width of image/height of image
* scale=the size of img
* k=aspect ratio x scale=the number of anchor per pixel
* total anchor=WxHxK


ps:the'深度'means the number of feature map(=kernel number)

ref:
1. [RPN](https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9)
2. [RPN中文](https://medium.com/@chekuei.liang/%E7%B0%A1%E4%BB%8B-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-493b3acbf436)

## day27 Bounding Box Loss
![](https://i.imgur.com/SdD2hrM.png)


---

![](https://i.imgur.com/AEcnEVq.png)


---

![](https://i.imgur.com/2mA3B23.png)


---

![](https://i.imgur.com/94R8B5f.png)

* why log?
* why L1-smooth?
  L2 is over sensitive to outiler
  L1 is too slow to converge

ref:
1. [BBOX loss中文論文](https://blog.csdn.net/u011534057/article/details/51235964) 


## day28 Non-Maximum Suppression (NMS)
1. pick the bbox with the highest confidence score
2. calculate the IOU of the bbox and other bboxes
3. remove bboxes are larger than threshold 
4. put the bbox with the highest confidence score
on the img and retain remaining bboxes 
5. repeat 1~4 untill all bboxes scroe are 0

ref:
1. [NMS黃志勝詳解](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-non-maximum-suppression-nms-aa70c45adffa) 


## day29 Single Shot Detector(SSD)



ref:
1. [anchor box number 計算](https://medium.com/@kweisamx0322/ssd-single-shot-multibox-detector-%E8%A9%B3%E8%A7%A3-d091bd0370f9) 


 

## day32 YOLO Network
![](https://i.imgur.com/JReeAIP.png)
output tensor=s x s(Bx5+C)=7x7x30
S=7
B=2
C=2

## day33 YOLO NMS
![](https://i.imgur.com/TbAyuYJ.jpg)

confidence:the possible of the bbox containing object 

## day34 YOLO Loss function
![](https://i.imgur.com/7W7Fo28.png)
one of the bbox-->containing obj?-->what obj?
* no object has smaller 信心度誤差係數(0.5) which indicates less punishment so it leads to less loss.


## day36 yolov1
* yolov1 structure
![](https://i.imgur.com/zV3YmJb.png)
* yolo adopted a few 1*1 filters like GoolgeNet to reduce the number of parameters

* (bonus)how to download the img from the web in python?
   1. wget
   2. request(recommend)
      [stackoverflow_tutorial](https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests)


## day38 YOLO Evolution
* v1 drwback:
  1. poor result on near object because of one cell only generating two bbox and the bad loss fuction
* v2 (darknet-19)
  ![](https://i.imgur.com/Ci3fRsw.png)
  1. batch norm
  2. High Resolution Classifie:feature map:7*7-->13*13
  4. Convolutional With Anchor Boxes:S x S x (B x (5 + C))
  5. Dimension Cluster
  * [詳解](https://www.twblogs.net/a/5bafd96b2b7177781a0f6394)
* v3(darknet-53)
  1. multi-scale prediction:13*13/26*26/52*52
  2. backbone:risidual block
  3. logistic loss
  ![](https://i.imgur.com/fMoLKh0.png)

ref:
1. [yolo發展](https://zhuanlan.zhihu.com/p/41438057)
2. [YOLOv2--論文學習筆記（算法詳解）](https://www.twblogs.net/a/5bafd96b2b7177781a0f6394)
3. [【目标检测简史】进击的YOLOv3，目标检测网络的巅峰之作](https://zhuanlan.zhihu.com/p/35394369)

## day39 YOLOV3
1. [darknet using colab](https://github.com/kriyeng/yolo-on-colab-notebook/blob/master/Tutorial_DarknetToColab.ipynb)
2. [kera yolov3](https://github.com/qqwweee/keras-yolo3)


## day40 tiny-yolov3
![](https://i.imgur.com/NXmomp7.png)
* there's a trade-off between the speed & accuracy
```
#step1:download the weight
tiny_yolo_model_path = "model_data/yolov3-tiny.h5"
tiny_yolo_anchor_file = "model_data/tiny_yolo_anchors.txt"
if not os.path.exists(tiny_yolo_model_path):
  print("yolov3-tiny weights doesn't exist, downloading...")
  os.system("wget https://pjreddie.com/media/files/yolov3-tiny.weights")
  print("Converting yolov3-tiny.weights to yolov3-tiny.h5...")
  os.system("python convert.py yolov3-tiny.cfg yolov3-tiny.weights %s" % tiny_yolo_model_path)
  if os.path.exists(tiny_yolo_model_path):
    print("Done!")
  else:
    print("Strange, model doesn't exist, pleace check")

#step2:construct model
tiny_yolo=YOLO(**config_dict)

#step3:run the model
start=time.time()
result=tiny_yolo.pure_detect_image(image)
end=time.time()
print('it takes:%.4f'%(end-start))
print('fps:%.4fs'%(1/(end-start)))

```

## day41 yolo training
1. [建立自己的YOLO辨識模型 – 以柑橘辨識為例](https://chtseng.wordpress.com/2018/09/01/%E5%BB%BA%E7%AB%8B%E8%87%AA%E5%B7%B1%E7%9A%84yolo%E8%BE%A8%E8%AD%98%E6%A8%A1%E5%9E%8B-%E4%BB%A5%E6%9F%91%E6%A9%98%E8%BE%A8%E8%AD%98%E7%82%BA%E4%BE%8B/)
2. [目标检测数据集PASCAL VOC简介](https://arleyzhang.github.io/articles/1dc20586/)
3. [yolov3程式碼解說](https://zhuanlan.zhihu.com/p/42011577)

## day44
1. [image augmentation package](https://github.com/aleju/imgaug)
2. [Achieving top 5 in Kaggle's facial keypoints detection using FCN](https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html)

## day45 
1.[【论文学习】人脸识别 —— Deep Face Recognition: A Survey. 新人必看入门总结](https://blog.csdn.net/DL_wly/article/details/93902260)
2.[人脸识别：Deep Face Recognition论文阅读 ](https://yongyuan.name/blog/deep-face-recognition-note.html)

## day46 MobileNet
* three methods to reduce lighten models: 
    1. model prunning
    2. quantization
    3. architecture design 
* mobilenet
    1. depthwise separable conv:
       reduce the multiplication times
       a. depthwise 
        ![](https://i.imgur.com/ZHBrBJ7.png)
       b. pointwise
        

        ![](https://i.imgur.com/lCtKFMI.png)
        ![](https://i.imgur.com/HOPit2l.png)
    2. hyperparameter
       * Width multiplier (α)
       * Resolution multiplier(β)    
         

ref:
1.[A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
2.[黃志勝解說saparable conv](https://medium.com/@chih.sheng.huang821/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-mobilenet-depthwise-separable-convolution-f1ed016b3467)

## day47 MobileNetv2
1. liner bottleneck
   with the decrease of no. of channels and ReLU,the information loss will get higher. To prevent this problem we adopted liner bottleneck
2. inverted residual block
   increase the no. of channels to reduce the information loss 
   ![](https://i.imgur.com/sZEFFOZ.png)

* mobilenetv2 structure:
![](https://i.imgur.com/0qRZlgl.png)
