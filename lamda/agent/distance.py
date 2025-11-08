import numpy as np
from pgmpy.factors.discrete import DiscreteFactor


def factor_to_dict(factor: DiscreteFactor):
    """
    使用向量化方法将 DiscreteFactor 对象归一化后转换为字典。
    字典的键为每个状态组合（以元组形式，每个元素为 (变量, 状态)），值为该状态组合对应的概率。
    注意：该方法一次性生成所有状态组合，适用于状态数不太多的情况。
    """
    factor_copy = factor.copy()
    factor_copy.normalize(inplace=True)

    scope = factor_copy.scope()
    card = factor_copy.cardinality  # 列表或数组
    n_vars = len(scope)
    # 生成所有状态索引组合的网格，形状 (n_vars, total_combinations)
    grids = np.indices(card).reshape(n_vars, -1)
    # 对于每个变量，使用向量化方式将状态索引映射为真实状态名
    state_arrays = []
    for i, var in enumerate(scope):
        ori_state_names = np.array(factor_copy.state_names[var])
        try:
            state_names = np.char.mod("%g", ori_state_names)  # 转换为字符串
        except:
            state_names = ori_state_names.astype(str)
        # grids[i] 为所有组合中该变量的状态索引
        state_arrays.append(state_names[grids[i]])
    # 将各变量状态数组合并成一个二维数组，每行对应一个状态组合
    # keys_array 的形状为 (total_combinations, n_vars)
    keys_array = np.stack(state_arrays, axis=1)
    keys = ['|'.join(r) for r in keys_array]
    probs = factor_copy.values.ravel()
    return dict(zip(keys, probs))


def factor_to_array(factor1, factor2):
    d1 = factor_to_dict(factor1)
    d2 = factor_to_dict(factor2)
    all_keys = set(d1) | set(d2)  # 统一 key 顺序（可选）
    p = np.array([d1.get(k, 0.0) for k in all_keys], dtype=np.float64)
    q = np.array([d2.get(k, 0.0) for k in all_keys], dtype=np.float64)
    return p, q


def total_variation_distance(factor1, factor2):
    p, q = factor_to_array(factor1, factor2)
    return 0.5 * np.sum(np.abs(p - q))


def kl_divergence(factor1, factor2, epsilon=1e-12):
    p, q = factor_to_array(factor1, factor2)
    mask = p > 0
    q_safe = np.maximum(q[mask], epsilon)
    return np.sum(p[mask] * np.log(p[mask] / q_safe))


def js_divergence(factor1, factor2, epsilon=1e-12):
    p, q = factor_to_array(factor1, factor2)
    M = 0.5 * (p + q)
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    M_safe = np.maximum(M, epsilon)
    kl1 = np.sum(p * np.log(p_safe / M_safe), where=p > 0)
    kl2 = np.sum(q * np.log(q_safe / M_safe), where=q > 0)
    return 0.5 * (kl1 + kl2)


def brier_score(factor1, factor2):
    """
    Calculate the multi-class Brier score between two probability distributions.
    For joint probability distributions, we compare probability vectors directly.
    
    Parameters:
    -----------
    factor1 : DiscreteFactor
        The first probability distribution (typically ground truth)
    factor2 : DiscreteFactor
        The second probability distribution (typically predicted)
        
    Returns:
    --------
    float
        The Brier score between the two distributions
    """
    p, q = factor_to_array(factor1, factor2)
    # Mean squared error between the probability vectors
    return np.mean((p - q) ** 2)
