# Harris Corner Detection

## See https://hackmd.io/@mingyang-tu/BkRebukpo

## Procedure

1. Compute the $x$ and $y$ derivatives of image $I$.
    
    $$
    \begin{align*} I_x &= I \otimes \begin{bmatrix} -1 & 0 & 1 \end{bmatrix} \\ I_y &= I \otimes \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix} \end{align*}
    $$
    
2. Compute the $2 \times 2$ matrix $M$.
    
    $$
    M = \begin{bmatrix} A & C \\ C & B \end{bmatrix}
    $$
    
    $$
    \begin{align*} A &= I_x^2 \otimes w \\ B &= I_y^2 \otimes w \\ C &= (I_xI_y) \otimes w \end{align*}
    $$
    
    $w$ is a smooth circular window, for example a Gaussian:
    
    $$
    w(u, v) = \exp(-\frac{u^2+v^2}{2 \sigma^2})
    $$
    
3. Compute $R$.
    
    $$
    R = Det(M) - k \cdot Tr^2(M)
    $$
    
    $$
    \begin{align*} Tr(M) &= A + B \\ Det(M) &= AB - C^2 \end{align*}
    $$
    
4. Set the threshold on $R$ and find the local maximum.