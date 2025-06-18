import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm, ttest_rel, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from time import time
import argparse
import os
from tqdm import tqdm
import pickle
import warnings
import math
from tabulate import tabulate

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 设置随机种子保证可复现性
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

class KernelRegression:
    """核回归估计器 (Nadaraya-Watson估计器)"""
    
    def __init__(self, kernel="gaussian", bandwidth=None):
        """
        初始化核回归
        Args:
            kernel: 核函数类型 ("gaussian", "epanechnikov", "tricube")
            bandwidth: 带宽参数 (h)
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        
        # 如果没有提供带宽，使用Silverman法则估计
        if self.bandwidth is None:
            n, d = self.X_train.shape
            # Silverman's rule of thumb
            h = 1.06 * np.std(self.X_train, axis=0) * n**(-1/(d+4))
            self.bandwidth = np.mean(h)  # 使用平均带宽
        return self
    
    def _kernel_function(self, u):
        if self.kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
        elif self.kernel == "tricube":
            return np.where(np.abs(u) <= 1, (70/81) * (1 - np.abs(u)**3)**3, 0)
        else:
            raise ValueError(f"未知核函数: {self.kernel}")
    
    def predict(self, X):
        X = np.asarray(X)
        predictions = []
        
        for x in X:
            # 计算与所有训练样本的距离
            distances = np.linalg.norm(self.X_train - x, axis=1)
            
            # 标准化距离
            u = distances / self.bandwidth
            
            # 计算核权重
            weights = self._kernel_function(u)
            
            # 防止除零错误
            weight_sum = np.sum(weights)
            if weight_sum < 1e-8:
                # 没有足够近邻，使用全局均值
                predictions.append(np.mean(self.y_train))
            else:
                # Nadaraya-Watson估计
                predictions.append(np.sum(weights * self.y_train) / weight_sum)
        
        return np.array(predictions)

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 设置随机种子保证可复现性
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150

class DataGenerator:
    """数据生成工具类 (对应论文第4节)"""
    
    @staticmethod
    def generate_pairwise_ranking_data(n_labeled, n_unlabeled, p=5, 
                                      cov_structure="equicorrelated", rho=0.5, 
                                      noise_level=1.0, discrete_levels=5, seed=None):
        """
        生成成对排序数据 (论文第4.1节)
        Args:
            n_labeled: 标签样本量 (n)
            n_unlabeled: 无标签样本量 (m)
            p: 特征维度 (p)
            cov_structure: 协方差结构 ("equicorrelated", "toeplitz", "identity")
            rho: 相关性参数 (ρ)
            noise_level: 噪声水平 (σ)
            discrete_levels: 离散等级数 (K)
            seed: 随机种子
        Returns:
            D_labeled: 标签数据集 [X, y]
            D_unlabeled: 无标签数据集 [X]
            theta_true: 真实参数 (θ^*)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 设置协方差矩阵 Σ (论文第4.1节)
        if cov_structure == "equicorrelated":
            cov = np.eye(p) * (1 - rho) + rho
        elif cov_structure == "toeplitz":
            cov = toeplitz([rho**i for i in range(p)])
        else:  # identity
            cov = np.eye(p)
        
        # 生成特征 X ~ N(0, Σ)
        total_samples = n_labeled + n_unlabeled
        X = np.random.multivariate_normal(np.zeros(p), cov, total_samples)
        
        # 真实参数 (单位向量) θ^* (论文第4.1节)
        theta_true = np.ones(p) / np.sqrt(p)
        
        # 生成标签数据 (论文第4.1节)
        epsilon = np.random.normal(0, noise_level, n_labeled)  # ε ~ N(0, σ^2)
        Z_latent = X[:n_labeled] @ theta_true + epsilon  # 潜在变量 Z
        
        # 离散化为多个等级 (论文第4.1节)
        if discrete_levels > 1:
            quantiles = np.linspace(0, 1, discrete_levels + 1)[1:-1]
            thresholds = np.quantile(Z_latent, quantiles)
            Y = np.digitize(Z_latent, [-np.inf] + list(thresholds) + [np.inf])
        else:
            Y = Z_latent  # 连续情况
        
        # 构建数据集
        D_labeled = [np.concatenate([x, [y]]) for x, y in zip(X[:n_labeled], Y)]
        D_unlabeled = X[n_labeled:]  # 仅特征
        
        # 计算数据集统计信息
        stats = {
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "p": p,
            "cov_structure": cov_structure,
            "rho": rho,
            "noise_level": noise_level,
            "discrete_levels": discrete_levels,
            "Y_distribution": np.bincount(Y) / len(Y) if discrete_levels > 1 else None,
            "theta_true": theta_true.copy()
        }
        
        return D_labeled, D_unlabeled, theta_true, stats
    
    @staticmethod
    def generate_survival_data(n_labeled, n_unlabeled, p=4, 
                               cov_structure="toeplitz", rho=0.5, 
                               censor_rate=0.25, error_dist="gumbel", seed=None):
        """
        生成生存分析数据 (论文第4.2节)
        Args:
            n_labeled: 标签样本量 (n)
            n_unlabeled: 无标签样本量 (m)
            p: 特征维度 (p)
            cov_structure: 协方差结构 ("equicorrelated", "toeplitz", "identity")
            rho: 相关性参数 (ρ)
            censor_rate: 截尾率 (c)
            error_dist: 误差分布 ("gumbel", "normal", "logistic")
            seed: 随机种子
        Returns:
            D_labeled: 标签数据集 [Y, Delta, X]
            D_unlabeled: 无标签数据集 [X]
            theta_true: 真实参数 (θ^*)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 设置协方差矩阵 Σ (论文第4.2节)
        if cov_structure == "equicorrelated":
            cov = np.eye(p) * (1 - rho) + rho
        elif cov_structure == "toeplitz":
            cov = toeplitz([rho**i for i in range(p)])
        else:  # identity
            cov = np.eye(p)
        
        # 生成特征 X ~ N(0, Σ)
        total_samples = n_labeled + n_unlabeled
        X = np.random.multivariate_normal(np.zeros(p), cov, total_samples)
        
        # 真实参数 θ^* (论文第4.2节)
        theta_true = np.ones(p)
        
        # 生成失效时间 (论文第4.2节)
        if error_dist == "gumbel":
            epsilon = np.random.gumbel(size=total_samples)  # ε ~ Gumbel(0,1)
        elif error_dist == "normal":
            epsilon = np.random.normal(size=total_samples)  # ε ~ N(0,1)
        else:  # logistic
            epsilon = np.random.logistic(size=total_samples)  # ε ~ Logistic(0,1)
            
        log_T = X @ theta_true + epsilon  # log T_i = X_i^T θ^* + ε_i
        T = np.exp(log_T)
        
        # 生成截尾时间 (论文第4.2节)
        tau = np.quantile(T, 1 - censor_rate)
        C = np.random.uniform(0, tau, total_samples)
        
        # 观测数据 (Y_i, Δ_i)
        Y_obs = np.minimum(T, C)
        Delta = (T <= C).astype(int)
        
        # 构建数据集
        D_labeled = []
        for i in range(n_labeled):
            # 每个样本: [Y_obs, Delta, X1, X2, ..., Xp]
            sample = np.concatenate([[Y_obs[i], Delta[i]], X[i]])
            D_labeled.append(sample)
            
        D_unlabeled = X[n_labeled:]  # 仅特征
        
        # 计算数据集统计信息
        stats = {
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "p": p,
            "cov_structure": cov_structure,
            "rho": rho,
            "censor_rate": censor_rate,
            "error_dist": error_dist,
            "actual_censor_rate": 1 - Delta.mean(),
            "theta_true": theta_true.copy()
        }
        
        return D_labeled, D_unlabeled, theta_true, stats


class KernelFunctions:
    """核函数定义 (对应论文第2节)"""
    
    @staticmethod
    def pairwise_ranking_kernel(theta, zi, zj):
        """
        成对排序核函数
        l(θ; Z_i, Z_j) = log(1 + exp(-sign(Y_i-Y_j)θ^T(X_i-X_j)))
        """
        xi, yi = zi[:-1], zi[-1]
        xj, yj = zj[:-1], zj[-1]
        sign_diff = np.sign(yi - yj)
        
        X_diff = xi - xj
        inner_prod = X_diff @ theta
        exponent = -1*sign_diff * inner_prod
        
        if exponent < -30:  # exp(-30) ≈ 1e-14，可以忽略
            return np.log(1)
        elif exponent > 30:  
            return exponent
        else:
            return np.log(1 + np.exp(exponent))
    
    @staticmethod
    def smoothed_gehan_kernel(theta, zi, zj, Sigma, n_samples):
        """
        smoothed Gehan kernel (论文第4.2节, 基于Brown-Wang平滑)
        l(θ; Z_i, Z_j) = Δ_i * [(e_j(θ)-e_i(θ))Φ((e_j-e_i)/r_ij) + r_ijϕ((e_j-e_i)/r_ij)]
        """
        Yi, Deltai, Xi = zi[0], zi[1], zi[2:]
        Yj, Deltaj, Xj = zj[0], zj[1], zj[2:]
        ei = np.log(Yi) - Xi @ theta  # e_i(θ) = log Y_i - X_i^T θ
        ej = np.log(Yj) - Xj @ theta  # e_j(θ) = log Y_j - X_j^T θ
        u = ej - ei
        X_diff = Xi - Xj
        
        # 计算平滑参数 r_ij (Brown & Wang 2007)
        r_sq = (X_diff @ Sigma @ X_diff) / n_samples
        r = np.sqrt(max(r_sq, 1e-6))  # 避免除零
        
        # Brown-Wang smooth
        z = -u / r
        smooth_term = norm.cdf(z) + (u/r) * norm.pdf(z)
        return Deltai * u * smooth_term


class SemiSupervisedEstimator:
    """半监督U估计器 (对应论文第2.3节)"""
    
    def __init__(self, r=2, reg_model="random_forest", lambda_reg=0.01):
        """
        初始化估计器
        Args:
            r: U统计量阶数 (r)
            reg_model: 回归模型类型 ("linear", "random_forest", "mlp")
            lambda_reg: L2正则化系数 (λ)
        """
        self.r = r
        self.lambda_reg = lambda_reg
        self.reg_model_type = reg_model
        self.current_theta = None
    
    def _get_regression_model(self):
        """获取回归模型实例 (对应论文第2.3.2节)"""
        if self.reg_model_type == "linear":
            return LinearRegression()
        elif self.reg_model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.reg_model_type == "mlp":
            return MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        elif self.reg_model_type == "kernel":
            # 核回归模型 (Nadaraya-Watson估计器)
            return KernelRegression(kernel="gaussian")
        else:
            raise ValueError(f"未知回归模型: {self.reg_model_type}")
    
    def _compute_ell1(self, theta, z, D_sub, kernel_func):
        """
        估计 l₁(θ; z) (论文公式11)
        l̂₁(θ; z) = binom(k, r-1)^{-1} ∑_{1≤i₂<...<i_r≤k} l(θ; z, Z_{i₂}, ..., Z_{i_r})
        """
        k = len(D_sub)
        r_minus_1 = self.r - 1
        
        # 检查是否有足够的样本
        if k < r_minus_1:
            return 0.0
        
        # 计算组合数
        total = 0
        count = 0
        
        # 对于r=2的情况
        if r_minus_1 == 1:
            for i in range(k):
                total += kernel_func(theta, z, D_sub[i])
                count += 1
            return total / count if count > 0 else 0
        
        # 对于r>2的情况，使用随机抽样避免组合爆炸
        num_samples = min(100, math.comb(k, r_minus_1))
        for _ in range(num_samples):
            indices = np.random.choice(k, r_minus_1, replace=False)
            z_subset = [D_sub[i] for i in indices]
            total += kernel_func(theta, z, *z_subset)
            count += 1
        
        return total / count if count > 0 else 0
    
    def _estimate_psi1(self, theta, D_b, D_a, kernel_func):
        """
        估计 ψ₁(θ; x) 使用嵌套回归 (论文第2.3.2节)
        """
        # 准备训练数据
        X_train = np.array([z[:-1] for z in D_b])
        ell1_vals = [self._compute_ell1(theta, z, D_a, kernel_func) for z in D_b]
        
        # 训练回归模型
        reg_model = self._get_regression_model()
        reg_model.fit(X_train, np.array(ell1_vals))
        
        return reg_model
    
    def supervised_risk(self, theta, D_labeled, kernel_func):
        """
        计算监督风险 L_n(θ) (论文公式3)
        L_n(θ) = binom(n, r)^{-1} ∑_{1≤i₁<...<i_r≤n} l(θ; Z_{i₁},...,Z_{i_r})
        """
        n = len(D_labeled)
        total_loss = 0
        count = 0
        
        # 计算所有成对组合
        for i in range(n):
            for j in range(i+1, n):
                total_loss += kernel_func(theta, D_labeled[i], D_labeled[j])
                count += 1
        risk = total_loss / count if count > 0 else 0
        #risk += self.lambda_reg * np.linalg.norm(theta)**2
        return risk
    
    def semi_supervised_risk(self, theta, D_labeled, D_unlabeled, kernel_func):
        """
        计算半监督风险 L_S(θ) (论文公式6)
        L_S(θ) = L_n(θ) - (2/n)∑_{i=1}^n ψ̂₁(θ; X_i) + (2/(n+m))∑_{i=1}^{n+m} ψ̂₁(θ; X_i)
        """
        n = len(D_labeled)
        m = len(D_unlabeled)
        
        # 保存当前θ用于ψ1估计
        self.current_theta = theta
        
        # 1. 监督部分 L_n(θ)
        L_n = self.supervised_risk(theta, D_labeled, kernel_func)
        
        # 2. 分割数据集 (算法1)
        # 划分标签数据 (对应论文Step 1)
        n1 = n // 2
        D1_labeled = D_labeled[:n1]
        D2_labeled = D_labeled[n1:]
        
        # 进一步划分每个标签子集 (对应论文Step 2)
        # D_{Z,1}^a 大小 ⌊n/4⌋
        n1a = n1 // 2
        D1a = D1_labeled[:n1a]
        D1b = D1_labeled[n1a:]
        
        # D_{Z,2}^a 大小 ⌊n/4⌋
        n2a = len(D2_labeled) // 2
        D2a = D2_labeled[:n2a]
        D2b = D2_labeled[n2a:]
        
        # 划分无标签数据 (对应论文Step 1)
        m1 = m // 2
        D1_unlabeled = D_unlabeled[:m1]
        D2_unlabeled = D_unlabeled[m1:]

        # 3. 估计ψ1 (对应论文Step 2)
        # 使用D1a估计D1b的l1，然后在D1b上训练回归模型
        f1_model_a = self._estimate_psi1(theta, D1b, D1a, kernel_func)
        # cross-fitting
        f1_model_b = self._estimate_psi1(theta, D1a, D1b, kernel_func)

        # 使用D2a估计D2b的l1，然后在D2b上训练回归模型
        f2_model_a = self._estimate_psi1(theta, D2b, D2a, kernel_func)
        # cross-fitting
        f2_model_b = self._estimate_psi1(theta, D2a, D2b, kernel_func)

        # 4. 计算校正项
        # 准备特征数据
        X_labeled = np.array([z[:-1] for z in D_labeled])
        X_unlabeled = D_unlabeled
        
        # 预测ψ1 (对应论文Step 3)
        # 定义交叉拟合预测函数 (论文公式9)
        def cross_fit_predict(X, n, n1, m, m1):
            predictions = []
            for i in range(n+m):
                if i < n1: # xi 属于Dz1
                    avgpred = f2_model_a.predict([X[i]])[0]+f2_model_b.predict([X[i]])[0]
                    avgpred = avgpred/2
                    predictions.append(avgpred)
                elif i >=n1 and i < n: # xi 属于Dz2
                    avgpred = f1_model_a.predict([X[i]])[0]+f1_model_b.predict([X[i]])[0]
                    avgpred = avgpred/2
                    predictions.append(avgpred)
                elif i >= n and i < (n+m1): # xi 属于Dx1
                    avgpred = f2_model_a.predict([X[i]])[0]+f2_model_b.predict([X[i]])[0]
                    avgpred = avgpred/2
                    predictions.append(avgpred)
                elif i >= (n+m1) and i < (n+m): # xi 属于Dx2
                    avgpred = f1_model_a.predict([X[i]])[0]+f1_model_b.predict([X[i]])[0]
                    avgpred = avgpred/2
                    predictions.append(avgpred)

                # 确定x属于哪个分区
                #if any(np.array_equal(x, z[:-1]) for z in D1_labeled):
                #    predictions.append(f2_model.predict([x])[0])
                #elif any(np.array_equal(x, z[:-1]) for z in D2_labeled):
                #    predictions.append(f1_model.predict([x])[0])
                #else:  # 无标签数据
                #    if any(np.array_equal(x, z) for z in D1_unlabeled):
                #        predictions.append(f1_model.predict([x])[0])
                #    elif any(np.array_equal(x, z) for z in D2_unlabeled):
                #        predictions.append(f2_model.predict([x])[0])
            return np.array(predictions)
        
        # 预测所有数据
        X_all = np.vstack([X_labeled, X_unlabeled])
        f_all = cross_fit_predict(X_all, n, n1, m, m1)
        f_labeled = f_all[:n]
        f_all_mean = np.mean(f_all)
        f_labeled_mean = np.mean(f_labeled)
        f_all_sum = np.sum(f_all)
        f_labeled_sum = np.sum(f_labeled)
        # 5. 组合半监督风险 (论文公式6)
        #risk = L_n - f_labeled_mean + f_all_mean
        risk = L_n - (2/n) * f_labeled_sum + (2/(n+m)) *f_all_sum
        #print(f'Ln: {L_n},  f_labeled_mean: {f_labeled_mean},  f_all_mean : {f_all_mean}, ')
        # 添加正则化项
        #risk += self.lambda_reg * np.linalg.norm(theta)**2
        
        return risk
    
    def fit(self, D_labeled, D_unlabeled, kernel_func, init_theta=None):
        """
        拟合模型 (对应论文算法1)
        θ̂ = argmin_{θ∈Θ} L_S^{Cross}(θ)
        """
        n_features = len(D_labeled[0]) - 1 if isinstance(D_labeled[0], (list, np.ndarray)) else len(D_labeled[0]) - 2
        
        # 初始化参数
        if init_theta is None:
            init_theta = np.random.randn(n_features)
            init_theta /= np.linalg.norm(init_theta)
        
        # 定义优化目标
        def objective(theta):
            return self.semi_supervised_risk(theta, D_labeled, D_unlabeled, kernel_func)
        
        # 优化 (使用L-BFGS算法)
        result = minimize(
            objective,
            init_theta,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
        )
        
        self.theta_hat = result.x
        return self


class ExperimentRunner:
    """实验运行器 (对应论文第4节)"""
    
    @staticmethod
    def print_experiment_header(experiment_name, params):
        """打印实验标题和参数"""
        print("\n" + "="*80)
        print(f"{experiment_name} experiment")
        print("="*80)
        print("experiment settings:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print("-"*80)
    
    @staticmethod
    def print_dataset_stats(stats):
        """打印数据集统计信息"""
        print("data stats:")
        for key, value in stats.items():
            if key == "theta_true":
                print(f"  {key}: {np.round(value, 4)}")
            elif isinstance(value, np.ndarray):
                continue
            else:
                print(f"  {key}: {value}")
        print("-"*80)

    @staticmethod
    def run_pairwise_ranking_experiment(n_labeled=100, n_unlabeled=500, p=5, 
                                       reg_models=["linear", "kernel", "mlp"],
                                       trials=10, discrete_levels=5, **data_kwargs):
        """
        运行成对排序实验 (论文第4.1节)
        Returns:
            results: 结果DataFrame
            theta_data: 参数估计数据
        """
        columns = ["Trial", "Method", "RegModel", "Error", "Time", "ThetaHat", "ThetaTrue"]
        results = []
        theta_data = []
        
        # 打印实验信息
        params = {
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "p": p,
            "reg_models": reg_models,
            "trials": trials,
            "discrete_levels": discrete_levels,
            **data_kwargs
        }
        ExperimentRunner.print_experiment_header("pairwise_rank", params)
        
        for trial in tqdm(range(trials), desc="progress"):
            # 生成数据
            D_labeled, D_unlabeled, theta_true, stats = DataGenerator.generate_pairwise_ranking_data(
                n_labeled, n_unlabeled, p=p, discrete_levels=discrete_levels, seed=42+trial, **data_kwargs
            )
            

            ExperimentRunner.print_dataset_stats(stats)
            
            # 监督基准
            print('run baseline')
            start_time = time()
            sup_theta = ExperimentRunner.supervised_baseline(D_labeled, KernelFunctions.pairwise_ranking_kernel)
            sup_time = time() - start_time
            sup_error = np.linalg.norm(sup_theta - theta_true)
            print('end baseline', sup_time)
            results.append([trial, "Supervised", "None", sup_error, sup_time, sup_theta.copy(), theta_true.copy()])
            #print(results)
            theta_data.append({
                "trial": trial, "method": "Supervised", "model": "None",
                "theta_hat": sup_theta, "theta_true": theta_true
            })
            
            # 半监督方法比较
            for model_type in reg_models:
                print(model_type)
                start_time = time()
                estimator = SemiSupervisedEstimator(reg_model=model_type)
                estimator.fit(D_labeled, D_unlabeled, KernelFunctions.pairwise_ranking_kernel)
                semi_time = time() - start_time
                semi_error = np.linalg.norm(estimator.theta_hat - theta_true)
                
                results.append([trial, "Semi-Supervised", model_type, semi_error, semi_time, 
                               estimator.theta_hat.copy(), theta_true.copy()])
                #print(results)
                theta_data.append({
                    "trial": trial, "method": "Semi-Supervised", "model": model_type,
                    "theta_hat": estimator.theta_hat, "theta_true": theta_true
                })
        
        # 创建结果DataFrame
        df = pd.DataFrame(results, columns=columns)
        return df, theta_data
    
    @staticmethod
    def run_survival_analysis_experiment(n_labeled=100, n_unlabeled=500, p=4, 
                                         reg_models=["linear", "kernel",  "mlp"],
                                         trials=10, **data_kwargs):
        """
        运行生存分析实验 (论文第4.2节)
        Returns:
            results: 结果DataFrame
            theta_data: 参数估计数据
        """
        columns = ["Trial", "Method", "RegModel", "Error", "Time", "ThetaHat", "ThetaTrue"]
        results = []
        theta_data = []
        
        # 打印实验信息
        params = {
            "n_labeled": n_labeled,
            "n_unlabeled": n_unlabeled,
            "p": p,
            "reg_models": reg_models,
            "trials": trials,
            **data_kwargs
        }
        ExperimentRunner.print_experiment_header("survival_analyze", params)
        
        # 估计协方差矩阵 Σ
        cov_matrix = ExperimentRunner.estimate_covariance(p, **data_kwargs)
        
        for trial in tqdm(range(trials), desc="progress"):
            # 生成数据
            D_labeled, D_unlabeled, theta_true, stats = DataGenerator.generate_survival_data(
                n_labeled, n_unlabeled, p=p, seed=42+trial, **data_kwargs
            )
            
            if trial == 0:
                ExperimentRunner.print_dataset_stats(stats)
            
            # 定义核函数 (部分应用协方差矩阵)
            def gehan_kernel(theta, zi, zj):
                return KernelFunctions.smoothed_gehan_kernel(
                    theta, zi, zj, cov_matrix, n_labeled
                )
            
            # 监督基准
            start_time = time()
            sup_theta = ExperimentRunner.supervised_baseline(D_labeled, gehan_kernel)
            sup_time = time() - start_time
            sup_error = np.linalg.norm(sup_theta - theta_true)
            
            results.append([trial, "Supervised", "None", sup_error, sup_time, sup_theta.copy(), theta_true.copy()])
            theta_data.append({
                "trial": trial, "method": "Supervised", "model": "None",
                "theta_hat": sup_theta, "theta_true": theta_true
            })
            
            # 半监督方法比较
            for model_type in reg_models:
                start_time = time()
                estimator = SemiSupervisedEstimator(reg_model=model_type)
                estimator.fit(D_labeled, D_unlabeled, gehan_kernel)
                semi_time = time() - start_time
                semi_error = np.linalg.norm(estimator.theta_hat - theta_true)
                
                results.append([trial, "Semi-Supervised", model_type, semi_error, semi_time, 
                               estimator.theta_hat.copy(), theta_true.copy()])
                theta_data.append({
                    "trial": trial, "method": "Semi-Supervised", "model": model_type,
                    "theta_hat": estimator.theta_hat, "theta_true": theta_true
                })
        
        # 创建结果DataFrame
        df = pd.DataFrame(results, columns=columns)
        return df, theta_data
    
    @staticmethod
    def supervised_baseline(D_labeled, kernel_func):
        """监督基准方法 (论文公式4)"""
        n_features = len(D_labeled[0]) - 1 if isinstance(D_labeled[0], (list, np.ndarray)) else len(D_labeled[0]) - 2
        
        # 初始化参数
        init_theta = np.random.randn(n_features)
        init_theta /= np.linalg.norm(init_theta)
        
        # 定义优化目标
        def objective(theta):
            n = len(D_labeled)
            total_loss = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_loss += kernel_func(theta, D_labeled[i], D_labeled[j])
                    count += 1
            risk = total_loss / count if count > 0 else 0
            #正则项
            #risk += 0.01 * np.linalg.norm(theta)**2
            return risk
        
        # 优化
        result = minimize(
            objective,
            init_theta,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
        )
        
        return result.x
    
    @staticmethod
    def estimate_covariance(p, cov_structure="toeplitz", rho=0.5):
        """估计协方差矩阵 Σ (论文第4.2节)"""
        if cov_structure == "equicorrelated":
            return np.eye(p) * (1 - rho) + rho
        elif cov_structure == "toeplitz":
            return toeplitz([rho**i for i in range(p)])
        else:  # identity
            return np.eye(p)
    
    @staticmethod
    def save_theta_data(theta_data, experiment_name):
        """保存theta数据到文件"""
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{experiment_name.replace(' ', '_')}_theta_data_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(theta_data, f)
        
        print(f"theta saved to: {filename}")
        return filename


class ResultAnalyzer:
    """实验结果分析器"""
    
    @staticmethod
    def analyze_results(df, experiment_name):
        """分析并可视化结果，并生成统计显著性分析表格"""
        print(f"\n{experiment_name} result summary:")
        summary = df.groupby(['Method', 'RegModel'])['Error'].agg(['mean', 'std', 'count'])
        print(summary)
        
        # 误差分布图
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x="Method", y="Error", hue="RegModel", data=df)
        plt.title(f"{experiment_name} - Theta Error Distribution")
        plt.legend(title="Regression Method")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 时间比较
        plt.subplot(1, 2, 2)
        sns.barplot(x="Method", y="Time", hue="RegModel", data=df, estimator=np.mean)
        plt.title(f"{experiment_name} - Average Computation Time (Seconds)")
        plt.legend(title="Regression Method")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        filename = f"{experiment_name.replace(' ', '_')}_results.png"
        plt.savefig(filename, dpi=300)
        print(f"Result plot saved to: {filename}")
        plt.show()
        
        # 统计显著性检验和表格生成
        stats_df = ResultAnalyzer.significance_test(df, experiment_name)
        
        return summary, stats_df

    @staticmethod
    def significance_test(df, experiment_name):
        """统计显著性检验并生成结果表格"""
        # 提取监督方法误差
        sup_errors = df[df['Method'] == 'Supervised']['Error'].values
        
        # 对每种半监督方法进行配对t检验
        semi_methods = df[df['Method'] == 'Semi-Supervised']['RegModel'].unique()
        
        print(f"\n{experiment_name} Significance (Pairwise t-test vs Supervised):")
        results = []
        for model in semi_methods:
            semi_errors = df[(df['Method'] == 'Semi-Supervised') & 
                            (df['RegModel'] == model)]['Error'].values
            t_stat, p_value = ttest_rel(sup_errors, semi_errors)
            improvement = (np.mean(sup_errors) - np.mean(semi_errors)) / np.mean(sup_errors)
            print(f"  {model} model: t={t_stat:.4f}, p={p_value:.6f}, improvement={improvement:.2%}")
            results.append({
                "Model": model,
                "T-statistic": t_stat,
                "P-value": p_value,
                "Improvement": improvement
            })
        
        # 创建结果DataFrame
        stats_df = pd.DataFrame(results)
        
        # 保存为CSV
        csv_file = f"{experiment_name.replace(' ', '_')}_significance_results.csv"
        stats_df.to_csv(csv_file, index=False)
        print(f"Significance results saved to: {csv_file}")
        
        # 保存为LaTeX表格
        latex_file = f"{experiment_name.replace(' ', '_')}_significance_results.tex"
        stats_df.to_latex(latex_file, index=False, float_format="%.4f")
        print(f"LaTeX table saved to: {latex_file}")
        
        # 打印表格
        print("\nSignificance Analysis Table:")
        print(tabulate(stats_df, headers='keys', tablefmt='psql', floatfmt=".4f"))
        
        return stats_df
    
    @staticmethod
    def visualize_theta_comparison(theta_data, experiment_name, trial=0):
        """可视化θ估计值与真实值比较，支持多个半监督模型"""
        # 筛选数据
        trial_data = [d for d in theta_data if d['trial'] == trial]
        
        plt.figure(figsize=(14, 8))
        
        # 获取真实θ值
        theta_true = next(d['theta_true'] for d in trial_data)
        p = len(theta_true)
        
        # 创建索引
        index = np.arange(p)
        bar_width = 0.15  # 根据模型数量调整宽度
        
        # 绘制真实θ值
        plt.bar(index, theta_true, bar_width, label='True θ', color='black', alpha=0.7)
        
        # 定义颜色和模型显示名称
        method_colors = {
            "Supervised": 'red',
            "Semi-Supervised_linear": 'blue',
            "Semi-Supervised_kernel": 'green',
            "Semi-Supervised_mlp": 'orange'
        }
        model_labels = {
            "linear": "Linear",
            "kernel": "Kernel",
            "mlp": "MLP"
        }
        
        # 绘制监督方法
        theta_hat_sup = next((d['theta_hat'] for d in trial_data 
                            if d['method'] == "Supervised" and d['model'] == "None"), None)
        if theta_hat_sup is not None:
            plt.bar(index + bar_width, theta_hat_sup, bar_width, 
                    label="Supervised", alpha=0.7, color=method_colors["Supervised"])
        
        # 绘制半监督方法
        semi_models = ["linear", "kernel", "mlp"]
        for i, model in enumerate(semi_models):
            # 获取θ估计值
            theta_hat_semi = next((d['theta_hat'] for d in trial_data 
                                if d['method'] == "Semi-Supervised" and d['model'] == model), None)
            if theta_hat_semi is not None:
                plt.bar(index + (i+2)*bar_width, theta_hat_semi, bar_width, 
                        label=f"Semi-Supervised ({model_labels[model]})", 
                        alpha=0.7, color=method_colors[f"Semi-Supervised_{model}"])
        
        plt.xlabel('Theta Index')
        plt.ylabel('Theta Value')
        plt.title(f'{experiment_name} - Theta Comparison (Trial {trial})')
        plt.xticks(index + bar_width * 2, [f'θ_{i}' for i in range(p)])  # 调整x轴刻度位置
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        filename = f"{experiment_name.replace(' ', '_')}_theta_comparison_trial_{trial}.png"
        plt.savefig(filename, dpi=300)
        print(f"Theta comparison diagram saved to: {filename}")
        plt.show()
    
    @staticmethod
    def sample_size_sensitivity(experiment_func, exp_name, n_values, m_ratio=5, trials=10, **kwargs):
        """样本量敏感性分析"""
        results = []
        
        for n in tqdm(n_values, desc=f"{exp_name} sample size sensitivity"):
            m = int(n * m_ratio)
            df, _ = experiment_func(n_labeled=n, n_unlabeled=m, trials=trials, **kwargs)
            
            # 提取平均误差
            for method in ['Supervised', 'Semi-Supervised']:
                for model in df[df['Method'] == method]['RegModel'].unique():
                    if method == 'Supervised' and model != 'None':
                        continue
                    
                    subset = df[(df['Method'] == method) & (df['RegModel'] == model)]
                    mean_error = subset['Error'].mean()
                    results.append([n, m, method, model, mean_error])
        
        # 创建结果DataFrame
        sens_df = pd.DataFrame(results, columns=['n', 'm', 'Method', 'Model', 'Error'])
        
        # 可视化
        plt.figure(figsize=(12, 8))
        
        # 对每种方法-模型组合
        for (method, model), group in sens_df.groupby(['Method', 'Model']):
            if method == 'Supervised' and model != 'None':
                continue
                
            label = f"{method} ({model})" if model != 'None' else method
            plt.plot(group['n'], group['Error'], 'o-', label=label, markersize=8, linewidth=2)
        
        plt.xlabel('sample size (n)')
        plt.ylabel('average parameter error')
        plt.title(f'{exp_name} - sensitivity to sample size (m = {m_ratio}*n)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filename = f"{exp_name.replace(' ', '_')}_sensitivity.png"
        plt.savefig(filename, dpi=300)
        print(f"sensitivity diagram saved to: {filename}")
        plt.show()
        
        return sens_df


def main():
    parser = argparse.ArgumentParser(description='semi-super-U')
    parser.add_argument('--experiment', type=str, default='both', 
                        choices=['pairwise', 'survival', 'both'], 
                        help='simulation task')
    parser.add_argument('--n_labeled', type=int, default=100, 
                        help='n')
    parser.add_argument('--n_unlabeled', type=int, default=500, 
                        help='m')
    parser.add_argument('--trials', type=int, default=50, 
                        help='number of trails')
    parser.add_argument('--p', type=int, default=5, 
                        help='dimension')
    parser.add_argument('--discrete_levels', type=int, default=5, 
                        help='t')
    parser.add_argument('--censor_rate', type=float, default=0.25, 
                        help='')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # 运行成对排序实验
    if args.experiment in ['pairwise', 'both']:
        pairwise_df, pairwise_theta_data = ExperimentRunner.run_pairwise_ranking_experiment(
            n_labeled=args.n_labeled,
            n_unlabeled=args.n_unlabeled,
            p=args.p,
            trials=args.trials,
            discrete_levels=args.discrete_levels,
            reg_models=["linear", "kernel",  "mlp"]
        )
        
        # 保存θ数据
        theta_file = ExperimentRunner.save_theta_data(pairwise_theta_data, "pairwise_rank")
        
        # 分析结果
        pairwise_summary, pairwise_stats = ResultAnalyzer.analyze_results(pairwise_df, "pairwise_rank_exam")
        
        # 可视化参数比较
        ResultAnalyzer.visualize_theta_comparison(pairwise_theta_data, "pairwise_rank", trial=0)
        
        # 样本量敏感性分析
        n_values = [50, 200, 500]
        sensitivity_df = ResultAnalyzer.sample_size_sensitivity(
            ExperimentRunner.run_pairwise_ranking_experiment,
            "pairwise_rank",
            n_values,
            trials=20
        )
    
    # 运行生存分析实验
    if args.experiment in ['survival', 'both']:
        survival_df, survival_theta_data = ExperimentRunner.run_survival_analysis_experiment(
            n_labeled=args.n_labeled,
            n_unlabeled=args.n_unlabeled,
            p=args.p,
            trials=args.trials,
            reg_models=["linear", "kernel", "mlp"],
            censor_rate=args.censor_rate,
            error_dist="gumbel"
        )
        
        # 保存θ数据
        theta_file = ExperimentRunner.save_theta_data(survival_theta_data, "survival_analyze")
        
        # 分析结果
        survival_summary, survival_stats = ResultAnalyzer.analyze_results(survival_df, "survival_analyze_exam")
        
        # 可视化参数比较
        ResultAnalyzer.visualize_theta_comparison(survival_theta_data, "survival_analyze", trial=0)
        
        # 样本量敏感性分析
        n_values = [50, 100, 200, 500]
        sensitivity_df = ResultAnalyzer.sample_size_sensitivity(
            ExperimentRunner.run_survival_analysis_experiment,
            "survival_analyze",
            n_values,
            trials=5
        )
    
    print("\n All experiment finished!")


if __name__ == "__main__":
    main()