# 一、图像滤波 Image Filtering

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

  
