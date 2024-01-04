#  一、图像分割 Image Segementation

> **什么是图像分割：**
>
> 1. 图像分割将图片分成若干 ROI 区域（Regions of interest）
> 2. 是自动图像分析系统的第一步
> 3. 完整的图像分割将图片分成有限个不重叠的区域，并且所有区域的并集为原图像
>
> **图像分割算法的类别：**
>
> 1. Thresholding, region labelling and growing algorithms
>
> 	(connected components, region growing, watershed)
>
> 2. Statistical Segmentation
>
> 	(k-means, mean shift)
>
> 3. Graph based methods
>
> 	(Merging algorithms, splitting algorithms, split/merge)



## 1. 图像分割效果评估

- **ROC 分析**

	> ROC 曲线可以表现二分类图像分割
	>
	> 混淆矩阵
	>
	> $\begin{bmatrix} True\ positive & False\ negative\\ False\ positive & True\ negative\end{bmatrix}$
	>
	> 真正率 $TPR=TP/P=TP/(TP+FN)$
	>
	> 假正率 $FPR=FP/N=FP/(FP+TN)$

	曲线上每个点反映着对同一信号刺激的感受，横轴为假正率（特异度）， 纵轴为真正率（灵敏度）

	曲线的面积为AUC值，越大越好

	![ROC曲线](https://ask.qcloudimg.com/http-save/yehe-7969553/hirzm1wjyv.png)

	

## 2. 阈值分割 Thresholding

- $B(x,y)=\left\{\begin{matrix} 1,I(x,y)\ge T\\ 0,I(x,y)<T\end{matrix}\right.$



## 3. Connected Components

- 是一种DFS的实现

```python
function L = ConnectedComponents(B)
[X,Y] = size(B);
L = zeros(X,Y);
n=1;
For each (x,y) in B 
    if(B(x,y) & L(x,y)==0)
        label(x,y,n,B,L);
        n=n+1
    end
end

function label(x_start, y_start, n, B, L) 
% Recursively give label n to this pixel 
% and all its foreground neighbours. 
L(x_start,y_start)=n; 
For each (x,y) in N(x_start, y_start) 
	if(L(x,y)==0 & B(x, y)) 
    	label(x,y,n,B,L);
	end
end
```



## 4. Region Growing

- 用BFS的思想，递归实现邻居的添加

```python
function B = RegionGrow(I, seed)
	[X,Y] = size(I);
	visited = zeros(X,Y); 
	visited(seed) = 1;
	boundary = emptyQ; 
	boundary.enQ(seed); 
	while(~boundary.empty())
		nextPoint = boundary.deQ(); 
		if(include(nextPoint))
			visited(nextPoint) = 2;
			Foreach (x,y) in N(nextPoint) 
				if(visited(x,y) == 0)
					boundary.enQ(x,y); 
					visited(x,y) = 1;
				end
			end
		end
	end
```



## 5. 分水岭算法 Watershed

![分水岭算法](https://pic2.zhimg.com/80/v2-ea8ce5a64e744c8c56c492b03a039265_1440w.webp)

- **基本思想**：假设图像是一个坑坑洼洼的平面，从底端注水，当顶部水洼融合的边界，则为最佳分割
- **缺点**：会过度分割
- **步骤**
	1. 把梯度图像中的所有像素按照灰度值进行分类，并设定一个测地距离阈值。
	2. 找到灰度值最小的像素点（默认标记为灰度值最低点），让threshold从最小值开始增长，这些点为起始点。
	3. 水平面在增长的过程中，会碰到周围的邻域像素，测量这些像素到起始点（灰度值最低点）的测地距离，如果小于设定阈值，则将这些像素淹没，否则在这些像素上设置大坝，这样就对这些邻域像素进行了分类。



## 6. K-Means 图像分割

- **优点**：
	- 实现简单，效率高
	- 根据距离函数聚合到局部最小值
- **缺点**：
	- 需要设定K值
	- 对初始条件敏感
	- 对outliers敏感
	- 只能作用于球状簇



## 7. Mean Shift Algorithm

<img src="https://img-blog.csdn.net/20160511165844638" alt="Mean Shift Algorithm" style="zoom:50%;" />

- 该算法在求解空间中寻找局部密度的最大值
	1. 选择一个搜索窗口（宽度和位置）
	2. 计算窗口内数据的均值
	3. 更新窗口的重心为均值位置
	4. 重复上述过程直到聚合
	5. 窗口移动过程中，一定范围内的点都为分类结果
- **优点**：
	- 不需要假设一个球形簇
	- 只需要一个参数（窗口大小）
	- Finds variable number of modes
	- 对outlier有鲁棒性
- **缺点**：
	- 输出依赖于窗口大小
	- 计算消耗大
	- Does not scale well with dimension of feature space



## 8. Greedy Merging Algorithm

- **步骤**
	1. 初始化，将每个像素分为各自的类别
	2. 为每一条边赋权重（可以根据像素差值）
	3. 根据权重为所有边排序
	4. 根据排序结果连接边
	5. 当达到某个阈值时停止
- **改进算法**
	1. 1-3步相同
	2. 按照顺序对每两对点C1、C2计算簇内最大代价边Int(C)，和两个簇的最小连接代价Diff(C1, C2)
	3. 若满足以下条件，则合并
		$Dif(C_1,C_2)<min(Int(C_1)+\tau(C_1),Int(C_2)+\tau(C_2)),\ \tau(C)=k/|C|$

- **特点**
	- 非常快
	- 对噪声敏感
	- 贪婪算法导致太大的区域



## 9. Splitting Algorithm

https://blog.csdn.net/qq_43349296/article/details/122222469

- 特点
	- SOTA效果（？）
	- 需要空间复杂度和时间复杂度
	- 区分相同的区域有偏置
	- 对于有纹理的背景划分有困难



## 10. Split and Merge Algorithm

https://blog.csdn.net/webzhuce/article/details/81431512



## 总结

1. **Thresholding**
	- 依赖一个全局阈值
	- 生成分离的区域
2. **Region growing**
	- Supervised，需要至少一个seed点
	- 依赖一个阈值
	- 只产生一个区域
3. **分水岭算法 Watershed**
	- Supervised / Unsupervised
	- 不需要阈值参数
	- 将图片分成多个区域
	- 效果依赖于区域数量和seed的位置
4. **K-means 图像分割**
	- Unsupervised
	- 可以作用于高维数据
	- 可能产生不聚合的区域
	- 必须设定簇的数量
5. **Mean Shift**
	- Unsupervised
	- 可以作用于高维数据
	- 可能产生不聚合的区域
	- 只需要确定窗口大小

---





# 二、图像变换 Image Transformation

## 1. 灰度图像变换

> 通过直方图可以看到像素值的分布情况，通过映射将像素值改变
>
> $I_2(x,y)=f(I_1(x,y))$

- **线性变换：对比度拉伸**

	$f(x)=\alpha x+\beta$

	需要保证数值在0~255内

- **非线性变换：伽马值纠正**

	该方法用于调整相机和人眼的敏感度区别

	$f(x)=Ax^\gamma,\ A=255^{1-\gamma}$，其中A的作用可以让像素值范围是保持不变的
	
- **直方图均衡化**

  - 计算直方图的累积函数
  - $h(v)=round(\frac{cdf(v)-cdf_{min}}{(M*N-cdf_{min})}*(L-1))$
    其中cdf为累积函数，$cdf_{min}$为累积函数的最小值，M*N是像素个数，L是像素值范围一般为256


## 2. 几何变换

> 用于改变图像特征的位置，同样通过映射将新的位置映射到旧的位置

- **仿射变换 Affine Transformation**

	$\begin{pmatrix}x' \\ y'\end{pmatrix}=\begin{pmatrix} a & b\\ c & d\end{pmatrix}\begin{pmatrix}x' \\ y'\end{pmatrix}+\begin{pmatrix}t_x \\ t_y\end{pmatrix}$

	- **平移变换**

		$\begin{pmatrix} a & b\\ c & d\end{pmatrix}=\begin{pmatrix} 1 & 0\\ 0 & 1\end{pmatrix},\begin{pmatrix} t_x\\t_y\end{pmatrix}$

	- **缩放变换**

		$\begin{pmatrix} a & b\\ c & d\end{pmatrix}=\begin{pmatrix} 2 & 0\\ 0 & 2\end{pmatrix},\begin{pmatrix} t_x\\t_y\end{pmatrix}=\begin{pmatrix} -w/2\\-h/2\end{pmatrix}$

	- **拉伸变换**

		$\begin{pmatrix} a & b\\ c & d\end{pmatrix}=\begin{pmatrix} 0.5 & 0\\ 0 & 1\end{pmatrix},\begin{pmatrix} t_x\\t_y\end{pmatrix}=\begin{pmatrix} -w/4\\0\end{pmatrix}$

	- **旋转变换**

		$\begin{pmatrix} a & b\\ c & d\end{pmatrix}=\begin{pmatrix} cos\ \theta & sin\ \theta\\ -sin\ \theta & cos\ \theta\end{pmatrix},\begin{pmatrix} t_x\\t_y\end{pmatrix}=\begin{pmatrix} -(W(cos\ \theta-1)+Hsin\ \theta)/2\\(Wsin\ \theta-H(cos\ \theta-1)/2)\end{pmatrix}$

	- **错切变换 Shear**

		$\begin{pmatrix} a & b\\ c & d\end{pmatrix}=\begin{pmatrix} 1 & 0\\ s & 1\end{pmatrix},\begin{pmatrix} t_x\\t_y\end{pmatrix}=\begin{pmatrix} 0\\-sH/2\end{pmatrix}$

	> **齐次坐标**
	>
	> $\begin{pmatrix}x' \\y' \\1\end{pmatrix}=\begin{pmatrix} a & b & t_x\\ c & d & t_y\\0  & 0 &1\end{pmatrix}=\begin{pmatrix}x \\y \\1\end{pmatrix}$
	>
	> **差值**：通常通过变换后坐标并不是整数，需要进行差值以达到更好的效果
	>
	> - **最近邻居插值** Nearest Neighbour Interpolation
	>
	> - **双线性插值** Bilinear Interpolation
	>
	> 	$I_1(x',y')=\Delta x\Delta y\ I_1(x_2,y_2)+\Delta x(1-\Delta y)\ I_1(x_1,y_1)+(1-\Delta x)\Delta y\ I_1(x_1,y_2)+(1-\Delta x)(1-\Delta y)I_1(x_1,y_1)$
	>
	> 	<img src="https://img-blog.csdnimg.cn/20181231194058746.png" alt="Bilinear Interpolation" style="zoom:50%;" />

- **多项式扭曲变换 Polynomial Warps** 

	$\begin{pmatrix} x_1' & y_1'\\ x_1' & y_1'\\ \vdots & \vdots\\ x_1' & y_1'\end{pmatrix}=\begin{pmatrix} 1 & x_1 & y_1 & x_1^2 & x_1y_1 & y_1^2\\ 1 & x_2 & y_2 & x_2^2 & x_2y_2 & y_2^2\\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\ 1 & x_m & y_m & x_m^2 & x_my_m & y_m^2\end{pmatrix}\begin{pmatrix} a_0 &b_0  \\ a_1  &b_1  \\ a_2  &b_2  \\ a_3  &b_3  \\ a_4  &b_4  \\ a_5  &b_5 \end{pmatrix}$

>  **图像相似度**
>
> $S(I_1,I_2)=-(\sum{(I_1(x)-I_2(x))^2})$

---





# 三、图像滤波 Image Filtering

> **目标**
>
> 1. Low-level data (e.g., pixel) processing operations
> 2. Smoothing and noise reduction
> 3. Feature extraction and enhancement

---

## 1. 线性滤波 Linear filtering

> ***什么是线性操作***
>
> $L(I_1+I_2)=L(I_1)+L(I_2)$
> $L(aI)=aL(I)$
> $L(aI_1+bI_2)=aL(I_1)+bL(I_2)$



## 2. 卷积滤波 Convolutional Filters

https://zhuanlan.zhihu.com/p/33194385

- **互相关**

	$I'(x,y)=\sum_{i=-a}^{a}\sum_{j=-b}^{b}K(i,j)I(x+i,y+j)$

	kernel和图片点乘时从上到下，从左到右

- **卷积**

	$I'(x,y)=\sum_{i=-a}^{a}\sum_{j=-b}^{b}K(i,j)I(x-i,y-j)$

	与互相关不同，相当于kernel做了中心对称

	> 注意:此卷积和CNN中的卷积的概念有所不同

	|                           不变滤波                           |                           平移滤波                           |                           模糊滤波                           |                           锐化滤波                           |
	| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
	| $\begin{bmatrix} 0 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & 0\end{bmatrix}$ | $\begin{bmatrix} 0 & 0 & 0\\ 1 & 0 & 0\\ 0 & 0 & 0\end{bmatrix}$ | $\frac{1}{9} \begin{bmatrix} 1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1\end{bmatrix}$ | $\begin{bmatrix} 0 & 0 & 0\\ 0 & 2 & 0\\ 0 & 0 & 0\end{bmatrix}-\frac{1}{9} \begin{bmatrix} 1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1\end{bmatrix}$ |

- **边缘处理方法**
	- clip filter: 将边缘部分裁切
	- wrap around: 将相对一侧的图像复制过来
	- copy edge: 按照边缘的像素复制延伸
	- reflect cross edge: 按照边缘的像素镜像
	
- **高斯核**

  $G=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$
  > 均值 Mean: $\mu=\frac{\sum{x_i}}{N}$
  >
  > 方差 Variance: $\sigma^2=\sum{(x_i-\mu)^2/n}$
  >
  > 标准差 Standard Deviation: $\sigma=\sqrt{Variance}$
  
  - **可分离卷积核**
  
  	$g(x,y)=g(x)g(y)$
  
  - **高斯模糊核**
  
  	取决于$\sigma$和窗口大小
  
  - **高斯模糊核**的**优点**
  
  	- 旋转对称 Rotationally symmetric
  	- 模式单一,邻居的影响单调递减 Has a single lobe/mode - Neighbour's influence decreases monotonically
  	- 不会污染高频信息 Still one lobe in frequency domain - No corruption from high frequencies
  	- 与$$\sigma$$关系简单 Simple relationship to σ
  	- 实现简单 Easy to implement efficiently
  
- **卷积与微分的关系**

  - **一阶导**

    $f'(x)=\frac{f(x+1)-f(x-1)}{2}$

    - **Prewitt 算子**

    	$\begin{bmatrix} -1 & 0 & 1\\ -1 & 0 & 1\\ -1 & 0 & 1\end{bmatrix}$

    - **Sobel 算子**

    	$\begin{bmatrix} -1 & 0 & 1\\ -2 & 0 & 2\\ -1 & 0 & 1\end{bmatrix}$

  - **二阶导**

    $f''(x)=\frac{f(x+dx)-2f(x)+f(x-dx)}{dx^2}$

    - **Laplace 算子**

    	$\begin{bmatrix} 0 & -1 & 0\\ -1 & 4 & -1\\ 0 & -1 & 0\end{bmatrix}$ 或  $\begin{bmatrix} -1 & -1 & 1\\ -1 & 8 & 1\\ -1 & -1 & 1\end{bmatrix}$

- **图像锐化 Image sharpening/enhancing**

  增强高频信息来强化边缘

  $I'=I+\alpha(k*I)$, 其中 **k** 是一个高频通过的卷积核(如**Laplace**算子), **alpha**是0-1之间的常量

  $I'=I+\alpha(I-K*I)$, 其中 **k** 是一个反锐化mask

  $I'=2I-K*I$, 其中 **k** 是一个反锐化mask



## 3. 傅里叶Domain

> - 傅里叶变换是将图像转变为不同频率成分的加权和
>
> - 很多图像处理方法在傅里叶Domain中都非常简单，尤其是卷积操作
> - 离散傅里叶变换的时间复杂度为 O(N^4)
> - 快速傅里叶变换的时间复杂度为 O(N*2 logN)
> - 特点：
> 	- 所有自然图像的幅度谱都相似，着重于低频信息，在高频信息衰弱
> 	- 大多数信息在phase谱中
> - 傅里叶变换中的卷积操作
> 	- 直接相乘即可

https://zhuanlan.zhihu.com/p/163651606

- 理想低频通过滤波

	$K(u,v)=\left\{\begin{matrix} 0,\ u^2+v^2>r^2\\1,\ otherwise\end{matrix}\right.$

- 高斯-低频通过滤波

	$K(u,v)=exp(-\frac{u^2+v^2}{2\sigma^2})$

- 理想高频通过滤波

	$K(u,v)=\left\{\begin{matrix} 0,\ u^2+v^2<r^2\\1,\ otherwise\end{matrix}\right.$

- 高斯-高频通过滤波

	$K(u,v)=1-exp(-\frac{u^2+v^2}{2\sigma^2})$

- 截断Band-Step 滤波

	$K(u,v)=\frac{1}{1+(\Omega r/\sqrt{u^2+v^2-r_0^2})^{2n}}$
	
- 截通Band-Pass 滤波

	$K(u,v)=1-\frac{1}{1+(\Omega r/\sqrt{u^2+v^2-r_0^2})^{2n}}$

---





# 四、滤波与边缘检测 Filtering and Edge Detection

> - **走样 Aliasing**：当对连续信号采样时，会发生这类现象
> 	- 当采样频率不够高时，无法获取图像的全部信息时，发生这类现象
> 	- 图像在不同尺度下含有不同的结构（傅里叶domain中的不同频域）
> - 如何避免走样
> 	- 采样频率 > 2倍图片最高频率，这个频率被称为**Nyquist Rate**
> 	- 超采样，对于离散信号需要插值
> 	- 通过预滤波降低图像的最大频率

## 1. **高斯低频通过预滤波**

图片降采样：对图像每隔一行、一列去除，但会产生走样。通过高斯滤波，再降采样



## 2. **高斯金字塔**

![Image Pyramid](https://pic1.zhimg.com/70/v2-ae448acdeb025c613df668dd0921012e_1440w.awebp?source=172ae18b&biz_tag=Post)

- 用高斯滤波+降采样，重复k次得到一个图片金字塔



## 3. 拉普拉斯金字塔

<img src="https://pic4.zhimg.com/80/v2-4cace9022cf90a13cc5948b1b6c5c5a3_1440w.webp" alt="Laplace Pyramid" style="zoom:67%;" />

## 4. 边缘检测

- 关键步骤
	- 图片模糊化
	- 检测边缘点
	- 极大抑制

- **简单边缘检测：梯度核**

	- Prewitt 核

		$k_x=\begin{bmatrix} -1 & 0 & 1\\ -1 & 0 & 1\\ -1 & 0 & 1\end{bmatrix},k_y=\begin{bmatrix} -1 & -1 & -1\\ 0 & 0 & 0\\ 1 & 1 & 1\end{bmatrix}$

	- Sobel 核

		$k_x=\begin{bmatrix} -1 & 0 & 1\\ -2 & 0 & 2\\ -1 & 0 & 1\end{bmatrix},k_y=\begin{bmatrix} -1 & -2 & -1\\ 0 & 0 & 0\\ 1 & 2 & 1\end{bmatrix}$

	- Robert’s Cross Operator

		$k_1=\begin{bmatrix} 0 & 1\\ -1 & 0\end{bmatrix},k_2=\begin{bmatrix} 1 & 0\\ 0 & -1\end{bmatrix}$

- **Canny边缘检测**

	- 将噪声缩减和边缘强化结合
	- 边缘定位：非极大值抑制 / Hysteresis thresholding

- **模型拟合**

	- 通过拟合一个平面来定位边缘
	- 用方向、位置、像素值等参数创建平面
	- 在每个小窗口中找到L2最小拟合
	- 当达到阈值时停止拟合

- **总结**

	- 一阶导方式
		- 快速、简单、易于理解
		- 对噪声敏感、丢失交点、需要选择大量阈值
	- 二阶导方式
		- 快速、需要少量阈值
		- 对噪声非常敏感
	- 模型拟合
		- 慢速
		- 对噪声不敏感
	- 如何检验边缘检测器的表现
		- 假边缘的概率
		- 丢失边缘的概率
		- 边缘角度的损失
		- 和真实值的MSE
