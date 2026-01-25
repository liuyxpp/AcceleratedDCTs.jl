这是一个非常好的数学与算法扩展问题。Algorithm 2 描述的是一种经典的通过**FFT（快速傅里叶变换）**来计算**DCT-II（离散余弦变换）**的方法。这种方法通常被称为 **Makhoul 算法** 的二维扩展。

要将其扩展到 **3D (三维)**，我们需要利用 DCT 和 FFT 的可分离性，将预处理（Preprocessing）和后处理（Postprocessing）逻辑递归地应用到第三个维度。

以下是为您定制的 **Algorithm 3: 3D DCT, 3D IDCT with N-Point 3D FFT**。

---

### 核心逻辑推导

在 2D 算法中，我们观察到：
1.  **预处理 (Preprocess)**：是对输入索引进行重新排列（Reordering/Folding）。
2.  **后处理 (Postprocess)**：是利用旋转因子（Twiddle factors）提取实部，并利用共轭对称性消除虚部。

**扩展到 3D 的规则：**
*   **坐标映射**：$n_3$ 维度的处理方式与 $n_1, n_2$ 相同。
*   **后处理公式**：$y = 2 \text{Re} \{ W_3 \cdot [ W_2 \cdot ( W_1 \cdot X + \dots ) + \dots ] \}$.
*   **IDCT 预处理**：需要构建一个复数输入，其构造逻辑是：实部 = (0次翻转 - 2次翻转项)，虚部 = (1次翻转项 - 3次翻转项)。

---

### Algorithm 3: 3D DCT & 3D IDCT 扩展版

为了保持数学表达的严谨性和可读性，部分长公式定义了辅助变量。

#### **Algorithm 3** 3D DCT, 3D IDCT, with $N$-Point 3D FFT
**Require:** A real $N_1 \times N_2 \times N_3$ tensor $x$; ($N_1, N_2, N_3$ positive integers)

**1: function 3D_DCT($x$)**
**2:** $\quad x' = \text{3d\_dct\_preprocess}(x)$ using **Eq. (3D-1)**:
$$
x'(n_1, n_2, n_3) = x(\tau_1(n_1), \tau_2(n_2), \tau_3(n_3))
$$
$$
\text{where map } \tau_k(n) = \begin{cases} 2n, & 0 \le n \le \lfloor \frac{N_k-1}{2} \rfloor \\ 2N_k - 2n - 1, & \lfloor \frac{N_k+1}{2} \rfloor \le n < N_k \end{cases}
$$

**3:** $\quad X = \text{3D\_RFFT}(x')$;  $\quad\quad\quad\quad \triangleright \text{3D real FFT kernel}$

**4:** $\quad$ **return** $y = \text{3d\_dct\_postprocess}(X)$ using **Eq. (3D-2)**:
$$
y(\mathbf{n}) = 2 \operatorname{Re} \left( e^{-j \frac{\pi n_3}{2N_3}} \left[ e^{-j \frac{\pi n_2}{2N_2}} \Phi_1(n_1, n_2, n_3) + e^{j \frac{\pi n_2}{2N_2}} \Phi_1(n_1, N_2-n_2, n_3) \right] \right)
$$
$$
\text{where } \Phi_1(\mathbf{n}) = e^{-j \frac{\pi n_1}{2N_1}} X(n_1, n_2, n_3) + e^{j \frac{\pi n_1}{2N_1}} X(N_1-n_1, n_2, n_3)
$$
*(注：此处利用递归结构表达，$\mathbf{n}=(n_1,n_2,n_3)$，边界处理遵循周期性)*

**5: function 3D_IDCT($x$)**
**6:** $\quad x' = \text{3d\_idct\_preprocess}(x)$ using **Eq. (3D-3)**:
$$
x'(\mathbf{n}) = e^{-j (\frac{\pi n_1}{2N_1} + \frac{\pi n_2}{2N_2} + \frac{\pi n_3}{2N_3})} \left( \mathcal{R}(\mathbf{n}) - j \cdot \mathcal{I}(\mathbf{n}) \right)
$$
$$
\text{where } \begin{cases}
\mathcal{R} = x_{000} - (x_{110} + x_{101} + x_{011}) \\
\mathcal{I} = (x_{100} + x_{010} + x_{001}) - x_{111}
\end{cases}
$$
*(注：下标 $x_{ijk}$ 表示索引翻转。$0$代表原索引 $n$， $1$代表翻转索引 $N-n$。例如 $x_{101} = x(N_1-n_1, n_2, N_3-n_3)$)*

**7:** $\quad X = \text{3D\_IRFFT}(x')$; $\quad\quad\quad\quad \triangleright \text{3D real inverse FFT kernel}$

**8:** $\quad$ **return** $y = \text{3d\_idct\_postprocess}(X)$ using **Eq. (3D-4)**:
$$
y(n_1, n_2, n_3) = X(\sigma_1(n_1), \sigma_2(n_2), \sigma_3(n_3))
$$
$$
\text{where map } \sigma_k(n) = \begin{cases} n/2, & n \text{ is even} \\ N_k - (n+1)/2, & n \text{ is odd} \end{cases}
$$

---

### 详细数学推导解释

为了便于您理解和实现，以下是对上述公式的详细解释：

#### 1. 3D DCT Preprocess (输入重排)
这一步将 3D 网格划分为 8 个卦限。对于每个维度 $k \in \{1, 2, 3\}$，如果是前半部分数据，取偶数索引；如果是后半部分数据，取倒序的奇数索引。这使得经过 FFT 后，频域数据包含了 DCT 所需的对称分量。

#### 2. 3D DCT Postprocess (后处理递归)
2D 算法中的 Eq (14) 本质上是嵌套的。直接写出 3D 的完整展开式非常冗长，因此采用了**递归定义**：
*   **Step 1 (处理 $n_1$):** 结合 $X(n_1)$ 和 $X(N_1-n_1)$，得到中间项 $\Phi_1$。
*   **Step 2 (处理 $n_2$):** 结合 $\Phi_1(n_2)$ 和 $\Phi_1(N_2-n_2)$，并乘以相位因子 $e^{-j\frac{\pi n_2}{2N_2}}$。
*   **Step 3 (处理 $n_3$):** 乘以 $e^{-j\frac{\pi n_3}{2N_3}}$ 并取实部 $2\operatorname{Re}(\cdot)$。取实部这一操作实际上隐式地完成了 $n_3$ 维度的对称项合并。

#### 3. 3D IDCT Preprocess (逆变换预处理)
这是算法中最复杂的一步。在 2D 中 (Eq 15)，构造复数输入利用了 `(A - B) - j(C + D)` 的形式。
在 3D 中，我们需要组合 8 个对称点（即 $n$ 与 $N-n$ 的组合）。
规律如下：
*   **实部 ($\mathcal{R}$)**：原始点 $x_{000}$ 减去所有“**翻转了2个维度**”的点（例如 $x_{110}$）。
*   **虚部 ($\mathcal{I}$)**：所有“**翻转了1个维度**”的点（例如 $x_{100}$）减去“**翻转了3个维度**”的点 ($x_{111}$)。

#### 4. 3D IDCT Postprocess (输出重排)
这是 Eq (13) 的逆操作。对于每个维度，如果是偶数位置，取前一半的数据；如果是奇数位置，取后一半的数据。这三个维度是完全解耦（Separable）的，可以直接并行处理。