---
title: 'Understanding Black-box predictions via Influence Functions'
date: 2024-4-22
permalink: /posts/2024/03/Understanding Black-box predictions via Influence Functions
tags:
  - Intrpretable Model
  - Influence Function
---

# *Understanding Black-box predictions via Influence Functions* - 2017 ICML best paper

* **Keywords**: interpretable ML, robust statistics
* **References**
  * 张贤达 矩阵分析与应用 (实值标量函数的求导, 下个链接同)
  * https://zhuanlan.zhihu.com/p/273729929 
  * https://drive.google.com/drive/folders/1uuDKDY0ZtCJFtrzdTdfrWPUgCIh_mXhK?usp=sharing (后续会进行上传，先占坑)


---

# 1. Introduction

**Why did the system make this prediction?**

**What would happen if we did not have this training point, or if the values of this training point were changed slightly?**

简单来说，通过对一个点对应的loss进行scale或者增加一定的扰动，看对结果有什么影响

-----

# 2. Approach

prediction problem from input space $\mathcal{X}$ to output space $\mathcal{Y}$.

training points: $z_1, \cdots, z_n$, where $z_i=(x_i,y_i)\in \mathcal{X} \times \mathcal{Y}$

$L(z,\theta)$ be the loss

$\frac{1}{n}\sum^n_{i=1}L(z_i, \theta)$ be the empirical risk

$\hat{\theta}\stackrel{\mathrm{def}}{=}\arg\min_{\theta\in\Theta}\frac1n\sum_{i=1}^nL(z_i,\theta)$​

## a). Study the change in model parameters by removing that point

$\begin{aligned}\hat{\theta}_{-z}&\stackrel{\mathrm{def}}{=}\arg\min_{\theta\in\Theta}\sum_{z_i\neq z}L(z_i,\theta)\end{aligned}$: remove the point $z$

parameter change: $\hat{\theta}_{-z}-\hat{\theta}$

通过对loss function增加一个微小up-weight, 重新得到一个新的参数$\begin{aligned}\hat{\theta}_{\epsilon,z}&\stackrel{\text{def}}{=}\arg\min_{\theta\in\Theta}\frac{1}{n}\sum_{i=1}^{n}L(z_{i},\theta)+\epsilon L(z,\theta)\end{aligned}$

根据influence function的定义: $\mathcal{I}_{\mathrm{up,params}}(z)\overset{\mathrm{def}}{\operatorname*{=}}\left.\frac{d\hat{\theta}_{\epsilon,z}}{d\epsilon}\right|_{\epsilon=0}$

Following Appendix A: 

$\mathcal{I}_{\mathrm{up,params}}(z)\overset{\mathrm{def}}{\operatorname*{=}}\left.\frac{d\hat{\theta}_{\epsilon,z}}{d\epsilon}\right|_{\epsilon=0}=-H_{\hat{\theta}}^{-1}\nabla_\theta L(z,\hat{\theta})$, where $H_{\hat{\theta}}\stackrel{\text{def}}{=}\frac{1}{n}\sum_{i=1}^{n}\nabla_{\theta}^{2}L(z_{i},\hat{\theta})$

上式中的影响函数指的是对一个点的loss增加$\epsilon$对参数的影响, **而去除一个点相当于对该点增加$-\frac{1}{n}$**， $\hat{\theta}_{-z}-\hat{\theta}\approx-\frac1n\mathcal{I}_{\text{up,params}}(z)$

通过up-weight $z$ 最终可以得到对测试集中的点$z_{test}$的影响：

$\begin{aligned}
\mathcal{I}_\text{up,loss}{ ( z , z _ {\text{test}} )}& \overset{\mathrm{def}}{=}\left.\frac{dL(z_{\text{test}} , \theta _ { \epsilon , z })}{d\epsilon}\right|_{\epsilon=0}  \\
&=\nabla_\theta L(z_{\mathrm{test}},\hat{\theta})^\top\frac{d\hat{\theta}_{\epsilon,z}}{d\epsilon}\Big|_{\epsilon=0} \\
&=-\nabla_\theta L(z_{\mathrm{test}},\hat{\theta})^\top H_{\hat{\theta}}^{-1}\nabla_\theta L(z,\hat{\theta})
\end{aligned}$

## b) Perturb a training point

更一般的情况：

$z_{\delta}\overset{\mathrm{def}}{\operatorname*{=}}(x+\delta,y)$

$\hat{\theta}_{z_{\delta},-z}\stackrel{\mathrm{def}}{=}\arg\min_{\theta\in\Theta}\frac1n\sum_{i=1}^nL(z_i,\theta)+\epsilon L(z_\delta,\theta)-\epsilon L(z,\theta)$, 近似于将原先$z$的质量移动$\epsilon$到$z_{\delta}$​

$\begin{aligned}
\left.\frac{d\theta_{\epsilon,z_\delta,-z}}{d\epsilon}\right|_{\epsilon=0}& =\mathcal{I}_{\mathfrak{u}\text{p},\text{params}} ( z _ \delta ) - \mathcal{I}_{\mathfrak{u}\text{p},\text{params}} ( z )=-H_{\hat{\theta}}^{-1}\left(\nabla_\theta L(z_\delta,\hat{\theta})-\nabla_\theta L(z,\hat{\theta})\right)
\end{aligned}$

$\left.\frac{d\hat{\theta}_{\epsilon,z_\delta,-z}}{d\epsilon}\right|_{\epsilon=0}\approx-H_{\hat{\theta}}^{-1}[\nabla_x\nabla_\theta L(z,\hat{\theta})]\delta$

扰动之后对于parameter的影响就可以写为: $\hat{\theta}_{z_\delta,-z}-\hat{\theta}\approx-\frac1nH_{\hat{\theta}}^{-1}[\nabla_x\nabla_\theta L(z,\hat{\theta})]\delta$

同理，使用chain rule,对于train point的扰动会造成$z_{test}$点的影响为: $\begin{aligned}
\mathcal{I}_\text{pert,loss}{ ( z , z _ {\text{test}} )}& \stackrel{\mathrm{def}}{=}\left.\nabla_\delta L(z_{\text{test}} , \hat { \theta }_{z_\delta,-z})\right|_{\delta=0}=-\nabla_\theta L(z_{\mathsf{test}},\hat{\theta})^\top H_{\hat{\theta}}^{-1}\nabla_x\nabla_\theta L(z,\hat{\theta})
\end{aligned}$

==如何理解上式？==

将$\delta$设置为$\mathcal{I}_\text{pert,loss}{ ( z , z _ {\text{test}} )}^T$会得到对$z$进行扰动对于$z_{test}$最大影响的方向，而这个理论可以用于

**文章在MNIST上训练了一个logistic regression模型，用于区分1和7, 并且比较了在$\mathcal{I}_{\mathrm{up,loss}}$中海森矩阵和训练损失对于模型整体的影响 (Green: 7 as the test image, red: 1) **

![image-20240330162411879](ddrEricNo1.github.io/images/assets/2024_03_30/image-20240330162411879.png)

Fig 1: overestimate the influence of many points

Fig 2: green points helpful (remove will increase test loss), red points harmful (remove decrease test loss)

Fig 3: fail to capture influence, deviates from diagonal

==Loss term==: give points with high training loss more influence

==Hessian==: resistance of other training points to the removal of $z$

-----

# 3. 算法优化

Hessian of empirical risk: $O(np^2+p^3)$, with $n$ training points and $\theta \in R^p$

通过Hessian-vector products (HVP)优化掉计算海森和另一个向量乘积的步骤

两种可行方法: 

* **Conjugate gradients**
* **Stochastic estimation**

-----

# 4. 假设条件的relaxation

2中的证明基于loss function二阶可导并严格为凸函数，但实际中可能SGD算法并没有收敛到最优就提前结束，函数非凸，或是loss function不可导，本节对上述限制依次松弛并empirically prove influence function依然有效

## a) 下图可以证明influence function和leave-one-out retrain几乎等价，共轭梯度法和stochastic estimation等效，算法可以做用于非凸和非收敛到极值的场景

![image-20240330170751213](/home/ddr/Desktop/组会/assets/image-20240330170751213-1711789675846-1.png)

Influence function避免了取出一个点之后重新训练一个新的模型 (leave-one-out retraining), Fig 1 比较了重训练模型 $L(z_{\mathsf{test}},\hat{\theta}_{-z})-L(z_{test}, \hat{\theta})$ 与influence function $-\frac1n\mathcal{I}_{\text{up,loss}} ( z , z _ {\text{test}} )$, 使用的是共轭梯度法，Fig 2 使用的是stochastic estimation,

Fig 3 是在非凸函数上的结果, 其中假设$\tilde{\theta}$为SGD提前结束时候的模型参数，$H_{\tilde{\theta}}$就不是PD, 通过引入damping term对loss around $\tilde{\theta}$ 做凸的二阶近似：$\tilde{L}(z, \theta)=L(z, \tilde{\theta})+\nabla L(z,\tilde{\theta})^\top(\theta-\tilde{\theta})+\frac12(\theta-\tilde{\theta})^\top(H_{\tilde{\theta}}+\lambda I)(\theta-\tilde{\theta})$​

## b) **Loss function不可导**

作者训练了一个[SVM模型](https://cs231n.github.io/linear-classify/)同样用于分类MNIST中的1和7, 而SVM需要minimize Hinge(s) = max(0, 1 - s)，在0处不可导,使用平滑过的hinge $\begin{aligned}\text{Smooth}&\text{Hinge}(s,t)=t\log(1+\exp(\frac{1-s}t))\end{aligned}$ (Fig 1)

![image-20240330173000336](/home/ddr/Desktop/组会/assets/image-20240330173000336.png)



-----

# 5. Influence function使用场景

* Understanding model behavior (模型可解释性方向) - explain how model relies and extrapolate train data 

![image-20240330174033347](/home/ddr/Desktop/组会/assets/image-20240330174033347-1711791635260-3.png)

* adversarial training samples (模型对抗攻击方向) - 通过对train samples进行一定的改动 (indistinguishable by humans), 改变模型预测输出
* debugging domain  (domain shift)
* fixing mislabeled examples (连续学习) - flag the training points that exert the most influence on the model
