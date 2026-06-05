from scipy.linalg import solve_sylvester
import warnings
import numpy as np
from numpy.linalg import svd

warnings.filterwarnings('ignore')


def update_weight_matrix_adaptive(S, gamma=0.5, eps=1e-8, epoch=0, total_epochs=100):
    """
    自适应权重更新：随着训练进行逐渐增强权重效果
    """
    # 动态调整gamma：开始时较小，逐渐增大
    adaptive_gamma = gamma * (1 + epoch / total_epochs)

    row_norms = np.linalg.norm(S, axis=1, ord=2)
    row_norms = np.maximum(row_norms, eps)

    weights = adaptive_gamma / (2 * row_norms + eps)
    W = np.diag(weights)

    return W


def update_weight_matrix_enhanced_adaptive(S, gamma=0.3, eps=1e-8, epoch=0, total_epochs=100,
                                           warmup_epochs=10, max_gamma_ratio=1.5):
    """
    增强的自适应权重更新：包含预热期和更平滑的增长
    """
    if epoch < warmup_epochs:
        # 预热期：使用较小的gamma值
        adaptive_gamma = gamma * 0.5 * (epoch / warmup_epochs)
    else:
        # 正常期：平滑增长
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        adaptive_gamma = gamma * (1 + (max_gamma_ratio - 1) * progress)

    row_norms = np.linalg.norm(S, axis=1, ord=2)
    row_norms = np.maximum(row_norms, eps)

    weights = adaptive_gamma / (2 * row_norms + eps)
    # 限制权重范围以增强稳定性
    weights = np.clip(weights, 1e-6, 10.0)

    W = np.diag(weights)
    return W


def optimize_weighted_representation_enhanced(S, W, lambda_w=0.03, max_iter=5, tolerance=1e-6,
                                              damping=0.1):
    """
    增强的加权表示优化：添加阻尼项提高稳定性
    """
    S_new = S.copy()

    for iteration in range(max_iter):
        S_old = S_new.copy()

        # 计算加权L2范数
        weighted_S = W @ S_new
        row_norms = np.linalg.norm(weighted_S, axis=1, ord=2)
        row_norms = np.maximum(row_norms, 1e-8)

        # 计算权重矩阵
        weights = 1.0 / (2 * row_norms)
        weight_matrix = np.diag(weights)

        # 添加阻尼项的更新
        A = W.T @ weight_matrix @ W + (lambda_w + damping) * np.eye(S_new.shape[0])
        B = W.T @ weight_matrix @ W @ S_old + damping * S_old

        try:
            S_new = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            S_new = np.linalg.pinv(A) @ B

        # 检查收敛
        if np.linalg.norm(S_new - S_old, 'fro') < tolerance:
            break

    return S_new


def soft_threshold(s, tau):
    """软阈值函数（按元素收缩）"""
    return np.sign(s) * np.maximum(np.abs(s) - tau, 0)


def tensorPreFusion(x, rho, sX, is_weight=False, mode=1):
    # 张量重构
    X = x.reshape(sX)

    # 根据模式调整维度
    if mode == 1:
        Y = np.transpose(X, (0, 2, 1))  # 横向切片（假设X2Yi的逻辑）
    elif mode == 3:
        Y = np.moveaxis(X, 0, 1)  # 顶部切片（类似shiftdim）
    else:
        Y = X.copy()

    # FFT变换（沿第三轴）
    Y_hat = np.fft.fft(Y, axis=2)

    n3 = Y_hat.shape[2]
    end_value = n3 // 2 + 1 if n3 % 2 == 0 else (n3 + 1) // 2
    objV = 0.0

    # 计算权重常数C（如果启用）
    C = np.sqrt(sX[1] * sX[2]) if is_weight else None

    for i in range(end_value):
        slice_freq = Y_hat[:, :, i]
        U, S_diag, Vh = svd(slice_freq, full_matrices=False)
        S = np.diag(S_diag)

        if is_weight:
            weights = C / (S_diag + np.finfo(float).eps)
            tau = rho * weights
            S_shrink = soft_threshold(S, tau)
        else:
            S_shrink = np.maximum(S - rho, 0)

        objV += np.sum(S_shrink)
        Y_hat[:, :, i] = U @ S_shrink @ Vh

        # 对称处理（复数共轭）
        if i > 0 and i < end_value:
            Y_hat[:, :, n3 - i] = np.conj(U) @ S_shrink @ np.conj(Vh).T
            objV += np.sum(S_shrink)

    # 逆FFT
    Y = np.fft.ifft(Y_hat, axis=2).real  # 假设结果为实数

    # 恢复维度
    if mode == 1:
        X = np.transpose(Y, (0, 2, 1))
    elif mode == 3:
        X = np.moveaxis(Y, 1, 0)
    else:
        X = Y

    return X.flatten(), objV


def compute_l21_norm(matrix):
    """计算L2,1范数：每一行的L2范数之和"""
    return np.sum(np.linalg.norm(matrix, axis=1, ord=2))


def Efficient_multi_view_clustering_with_weights(H, B, dataname, alpha=1, beta=1,
                                                 gamma=0.25, lambda_w=0.025,
                                                 eps=1e-5, threshold=1e-5, epochs=100):
    global S
    V = len(H)

    H_3d = np.stack(H, axis=2)

    H_reshaped = np.transpose(H_3d, (0, 2, 1))
    sX = H_reshaped.shape

    # 处理参数
    shrink_rho = 0.1
    shrink_mode = 1
    is_weight = False

    # 调用处理函数
    H_shrink, _ = tensorPreFusion(
        H_reshaped.flatten(),
        rho=shrink_rho,
        sX=sX,
        is_weight=is_weight,
        mode=shrink_mode
    )

    # 恢复形状和维度
    H_shrink_3d = H_shrink.reshape(sX)
    H_processed = np.transpose(H_shrink_3d, (0, 2, 1))
    H_processed_list = [H_processed[:, :, v] for v in range(H_processed.shape[2])]
    H = H_processed_list

    Im = np.eye(B[0].shape[0])
    omiga = [1 / V] * V

    # 初始化权重矩阵
    W = np.eye(B[0].shape[0]) * 1e-6  # 从很小的权重开始

    loss_last = 1e16

    for epoch in range(epochs):
        BBt = []
        BHt = []
        for v in range(V):
            BBt_v = B[v].dot(B[v].T)
            BHt_v = B[v].dot(H[v].T)
            BBt.append(BBt_v)
            BHt.append(BHt_v)

        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v] + alpha * Im)
            tmp2 += omiga_v_square * BHt[v]

        # 更新S
        S_temp = np.linalg.inv(tmp1 + beta * Im).dot(tmp2 * (1 + alpha))

        if epoch >= 5:
            S = optimize_weighted_representation_enhanced(S_temp, W, lambda_w, max_iter=3)
        else:
            S = S_temp

        # 计算损失
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * (loss_SE + alpha * loss_BG)

        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        weighted_S = W @ S
        loss_weighted_l21 = compute_l21_norm(weighted_S)
        loss_total = loss_view + beta * loss_L2 + lambda_w * loss_weighted_l21

        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total

        # 更新B
        SSt = S.dot(S.T)
        for v in range(V):
            HtH_v = H[v].T.dot(H[v])
            SH_v = S.dot(H[v])
            B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v * (1 + alpha))

        if dataname == "ACM":
            W = update_weight_matrix_adaptive(S, gamma, eps, epoch, epochs)
        elif dataname == "DBLP":
            W = update_weight_matrix_adaptive(S, gamma, eps, epoch, epochs)
        elif dataname == "YELP":
            W = update_weight_matrix_adaptive(S, gamma, eps, epoch, epochs)
        else:
            W = update_weight_matrix_enhanced_adaptive(S, gamma, eps, epoch, epochs)

        # 更新omiga
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            Const_loss[v] += SE + alpha * BG

        Total = [1 / CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)

    return S, omiga, B
