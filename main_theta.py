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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, gumbel_r
from sklearn.decomposition import PCA


def visualize_optimization_process(history, unique_id, theta_true=None):
    """
    统一的优化过程可视化函数
    Args:
        history: 包含优化历史数据的字典，包括:
            losses: 损失函数值列表
            lambda_vals: 拉格朗日乘子值列表
            theta_history: 参数历史列表
        unique_id: 唯一标识符用于文件名
        theta_true: 真实参数值(可选)
    """
    # 确保visualization目录存在
    os.makedirs("visualization", exist_ok=True)
    
    # 从历史数据中提取信息
    losses = history['losses']
    lambda_vals = history['lambda_vals']
    theta_history = np.array(history['theta_history'])
    
    # 创建图表
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Optimization Process (ID: {unique_id})", fontsize=16)
    
    # 1. 损失函数变化
    plt.subplot(2, 2, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Objective Function Value')
    plt.grid(True)
    
    # 2. 拉格朗日乘子变化
    plt.subplot(2, 2, 2)
    plt.plot(lambda_vals, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Lambda')
    plt.title('Lagrange Multiplier Evolution')
    plt.grid(True)
    
    # 3. 参数分量变化
    plt.subplot(2, 2, 3)
    for i in range(min(4, theta_history.shape[1])):  # 最多显示4个参数分量
        plt.plot(theta_history[:, i], label=f'$\\theta_{i}$')
        if theta_true is not None and i < len(theta_true):
            plt.axhline(y=theta_true[i], color=f'C{i}', linestyle='--', 
                        label=f'True $\\theta_{i}$')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Values')
    plt.legend()
    plt.grid(True)
    
    # 4. 参数空间轨迹(前两个维度)
    plt.subplot(2, 2, 4)
    if theta_history.shape[1] >= 2:
        # 绘制单位圆
        angles = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(angles), np.sin(angles), 'k--', alpha=0.3)
        
        # 绘制轨迹
        plt.plot(theta_history[:, 0], theta_history[:, 1], 'b-')
        plt.scatter(theta_history[0, 0], theta_history[0, 1], c='green', 
                  s=100, marker='o', label='Start')
        plt.scatter(theta_history[-1, 0], theta_history[-1, 1], c='red', 
                  s=100, marker='x', label='End')
        if theta_true is not None and len(theta_true) >= 2:
            plt.scatter(theta_true[0], theta_true[1], c='black', 
                      s=150, marker='*', label='Optimum')
        plt.xlabel('$\\theta_0$')
        plt.ylabel('$\\theta_1$')
        plt.title('Parameter Space Trajectory (First 2 Dimensions)')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Not enough dimensions for trajectory plot', 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # 调整布局防止重叠
    plt.subplots_adjust(
        left=0.1, 
        right=0.95, 
        bottom=0.1, 
        top=0.9, 
        wspace=0.3,  # 水平间距
        hspace=0.4    # 垂直间距
    )
    
    # 保存图像
    filename = f"visualization/optimization_process_{unique_id}.png"
    plt.savefig(filename, dpi=300)
    print(f"Optimization visualization saved to: {filename}")
    plt.close()

class DataVisualizer:
    """Data Visualization Tools with Improved Layout"""
    
    @staticmethod
    def visualize_pairwise_data(D_labeled, D_unlabeled, theta_true, args, stats):
        """Visualize pairwise ranking data with optimized layout"""
        # Extract data
        X_labeled = np.array([z[:-1] for z in D_labeled])
        y_labeled = np.array([z[-1] for z in D_labeled])
        X_unlabeled = np.array(D_unlabeled)
        
        # Create larger figure with better layout
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"Pairwise Ranking Data Visualization (n={len(D_labeled)}, m={len(D_unlabeled)}, p={len(theta_true)})", 
                    fontsize=18, y=0.98)
        
        # Define grid layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3)
        
        # 1. Covariance matrix heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        cov_matrix = np.cov(X_labeled.T)
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
        ax1.set_title(f"Covariance Matrix (ρ={stats['rho']})", fontsize=14)
        
        # 2. Feature distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(min(5, X_labeled.shape[1])):
            sns.kdeplot(X_labeled[:, i], label=f"Feature {i}", fill=True, alpha=0.3, ax=ax2)
        ax2.set_title("Feature Distributions", fontsize=14)
        ax2.set_xlabel("Feature Value")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=10)
        
        # 3. Response variable distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if stats['discrete_levels'] > 1:
            sns.countplot(x=y_labeled, ax=ax3)
            ax3.set_title(f"Discrete Response (K={stats['discrete_levels']})", fontsize=14)
        else:
            sns.histplot(y_labeled, kde=True, ax=ax3)
            ax3.set_title("Continuous Response", fontsize=14)
        
        # 4. Latent variable vs response relationship
        ax4 = fig.add_subplot(gs[1, 0])
        Z_latent = X_labeled @ theta_true
        if stats['discrete_levels'] > 1:
            sns.boxplot(x=y_labeled, y=Z_latent, ax=ax4)
            ax4.set_title("Latent Variable vs Response", fontsize=14)
        else:
            ax4.scatter(Z_latent, y_labeled, alpha=0.5)
            ax4.set_xlabel("Latent Variable (Xθ*)")
            ax4.set_ylabel("Response Variable")
            ax4.set_title("Latent Variable vs Response", fontsize=14)
        
        # 5. Principal Component Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(np.vstack([X_labeled, X_unlabeled]))
        scatter = ax5.scatter(X_pca[:len(X_labeled), 0], X_pca[:len(X_labeled), 1], 
                             c=y_labeled, alpha=0.6, cmap="viridis", label="Labeled")
        ax5.scatter(X_pca[len(X_labeled):, 0], X_pca[len(X_labeled):, 1], 
                   alpha=0.2, c="gray", label="Unlabeled")
        ax5.set_xlabel("Principal Component 1")
        ax5.set_ylabel("Principal Component 2")
        ax5.set_title("PCA: Labeled vs Unlabeled", fontsize=14)
        plt.colorbar(scatter, ax=ax5, label="Response Value")
        ax5.legend(fontsize=10)
        
        # 6. True parameter vector
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(range(len(theta_true)), theta_true, alpha=0.7)
        ax6.set_xlabel("Parameter Index")
        ax6.set_ylabel("Parameter Value")
        ax6.set_title("True Parameter Vector θ*", fontsize=14)
        ax6.grid(True, linestyle='--', alpha=0.5)
        
        # 7. Data statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create statistics table
        stats_data = [
            ["Labeled samples", len(X_labeled)],
            ["Unlabeled samples", len(X_unlabeled)],
            ["Feature dimension", X_labeled.shape[1]],
            ["Response range", f"{y_labeled.min()} - {y_labeled.max()}"],
            ["Latent variable range", f"{Z_latent.min():.2f} - {Z_latent.max():.2f}"],
            ["True parameter norm", f"{np.linalg.norm(theta_true):.4f}"],
            ["Feature means", np.array2string(np.mean(X_labeled, axis=0).round(2), 70)],
            ["Feature std devs", np.array2string(np.std(X_labeled, axis=0).round(2), 70)]
        ]
        
        table = ax7.table(
            cellText=stats_data,
            colLabels=["Statistic", "Value"],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.7]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        ax7.set_title("Data Statistics", fontsize=16, pad=20)
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        plt.savefig(f"pairwise_data_visualization_{args.unique_id}.png", dpi=300, bbox_inches='tight')
        #plt.show()
    
    @staticmethod
    def visualize_survival_data(D_labeled, D_unlabeled, theta_true, stats, unique_id):
        """Visualize survival analysis data with optimized layout"""
        # Extract data
        Y_obs = np.array([z[0] for z in D_labeled])
        Delta = np.array([z[1] for z in D_labeled])
        X_labeled = np.array([z[2:] for z in D_labeled])
        X_unlabeled = np.array(D_unlabeled)
        T = Y_obs  # Observed times
        
        # Create larger figure with better layout
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"Survival Data Visualization (n={len(D_labeled)}, m={len(D_unlabeled)}, p={len(theta_true)})", 
                    fontsize=18, y=0.98)
        
        # Define grid layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3)
        
        # 1. Covariance matrix heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        cov_matrix = np.cov(X_labeled.T)
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax1)
        ax1.set_title(f"Covariance Matrix (ρ={stats['rho']})", fontsize=14)
        
        # 2. Feature distributions
        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(min(5, X_labeled.shape[1])):
            sns.kdeplot(X_labeled[:, i], label=f"Feature {i}", fill=True, alpha=0.3, ax=ax2)
        ax2.set_title("Feature Distributions", fontsize=14)
        ax2.set_xlabel("Feature Value")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=10)
        
        # 3. Failure time distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sns.histplot(T, kde=True, ax=ax3)
        ax3.set_title(f"Failure Time Distribution (Censoring={stats['actual_censor_rate']:.2f})", fontsize=14)
        
        # 4. Censoring indicator distribution
        ax4 = fig.add_subplot(gs[1, 0])
        sns.countplot(x=Delta, ax=ax4)
        ax4.set_title("Event Indicator (Δ=1: event)", fontsize=14)
        
        # 5. Linear predictor vs failure time relationship
        ax5 = fig.add_subplot(gs[1, 1])
        linear_predictor = X_labeled @ theta_true
        scatter = ax5.scatter(linear_predictor, np.log(T), c=Delta, cmap="coolwarm", alpha=0.6)
        plt.colorbar(scatter, ax=ax5, label="Event Indicator (Δ)")
        ax5.set_xlabel("Linear Predictor (Xθ*)")
        ax5.set_ylabel("Log Failure Time (log T)")
        ax5.set_title("Linear Predictor vs Log Failure Time", fontsize=14)
        
        # 6. True parameter vector
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(range(len(theta_true)), theta_true, alpha=0.7)
        ax6.set_xlabel("Parameter Index")
        ax6.set_ylabel("Parameter Value")
        ax6.set_title("True Parameter Vector θ*", fontsize=14)
        ax6.grid(True, linestyle='--', alpha=0.5)
        
        # 7. Data statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create statistics table
        stats_data = [
            ["Labeled samples", len(X_labeled)],
            ["Unlabeled samples", len(X_unlabeled)],
            ["Feature dimension", X_labeled.shape[1]],
            ["Failure time range", f"{T.min():.2f} - {T.max():.2f}"],
            ["Actual censoring rate", f"{1 - Delta.mean():.2f} (Target: {stats['censor_rate']})"],
            ["Event rate", f"{Delta.mean():.2f}"],
            ["True parameter norm", f"{np.linalg.norm(theta_true):.4f}"],
            ["Feature means", np.array2string(np.mean(X_labeled, axis=0).round(2), 70)],
            ["Feature std devs", np.array2string(np.std(X_labeled, axis=0).round(2), 70)]
        ]
        
        table = ax7.table(
            cellText=stats_data,
            colLabels=["Statistic", "Value"],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.7]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        ax7.set_title("Data Statistics", fontsize=16, pad=20)
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        plt.savefig(f"survival_data_visualization_{unique_id}.png", dpi=300, bbox_inches='tight')
        #plt.show()

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
                                      noise_level=1.0, discrete_levels=5, seed=None, args=None):
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
            thresholds = [-1.71323908, -0.50637112,  0.52770505,  1.70297884]
            Y = np.digitize(Z_latent, [-np.inf] + list(thresholds) + [np.inf])
            print(np.unique(Y, return_counts=True))
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
        DataVisualizer.visualize_pairwise_data(D_labeled, D_unlabeled, theta_true, args, stats)
        return D_labeled, D_unlabeled, theta_true, stats
    
    @staticmethod
    def generate_survival_data(n_labeled, n_unlabeled, p=4, 
                               cov_structure="toeplitz", rho=0.5, 
                               censor_rate=0.25, error_dist="gumbel", seed=None, args = None):
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
        DataVisualizer.visualize_survival_data(D_labeled, D_unlabeled, theta_true, stats, args.unique_id)
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
            if np.isnan(np.log(1 + np.exp(exponent))):
                print('in pairwise kernel: ', f'expo: {exponent}, inerprod: {inner_prod}, xdiff:{ X_diff}, theta: {theta}, sign:{sign_diff}')
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
    
    def __init__(self, r=2, reg_model="random_forest", lambda_reg=0.01, 
                 learning_rate=0.1, max_epochs=1000, patience=50, tol=1e-6):
        """
        初始化估计器
        Args:
            r: U统计量阶数 (r)
            reg_model: 回归模型类型 ("linear", "random_forest", "mlp")
            lambda_reg: L2正则化系数 (λ)
            learning_rate: 学习率
            max_epochs: 最大迭代次数
            patience: 提前停止耐心值
            tol: 收敛容忍度
        """
        self.r = r
        self.lambda_reg = lambda_reg
        self.reg_model_type = reg_model
        self.current_theta = None
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.optimization_history = {
            'losses': [],
            'constraints': [],
            'lambda_vals': [],
            'grad_norms': [],
            'theta_history': []
        }
    
    def _get_regression_model(self):
        """获取回归模型实例 (对应论文第2.3.2节)"""
        if self.reg_model_type == "linear":
            return LinearRegression()
        elif self.reg_model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=5
            )
        elif self.reg_model_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=(50,),
                max_iter=1000,
                random_state=42,
                alpha=0.1,  # 更强的L2正则化
                early_stopping=True,  # 启用早停
                n_iter_no_change=20,  # 20次迭代无改进则停止
                learning_rate_init=0.001,  # 更小的学习率
                learning_rate='adaptive',  # 自适应学习率
                solver='adam',  # 使用Adam优化器
                activation='relu',  # 使用ReLU激活函数
                batch_size='auto',
                tol=1e-5,
                verbose=False,
                max_fun=15000  # 增加最大函数评估次数
            )
        elif self.reg_model_type == "kernel":
            return KernelRegression(kernel="gaussian", bandwidth=0.5)
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
            if np.isnan(total):
                print(f'in ell1: k: {k}, z: {z}')
                print(f'in ell1: total{total}, count: {count}, D_sub[i] :{D_sub[i]}')
            return total / count if count > 0 else 0
        
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
    
    def fit(self, D_labeled, D_unlabeled, kernel_func, init_theta=None, theta_true=None, unique_id=None):
        """
        拟合模型 (对应论文算法1) - 使用拉格朗日乘数法进行单位模长约束优化
        """
        n_features = len(D_labeled[0]) - 1 if isinstance(D_labeled[0], (list, np.ndarray)) else len(D_labeled[0]) - 2
        
        # 初始化参数和拉格朗日乘子
        if init_theta is None:
            init_theta = np.random.randn(n_features)
            init_theta /= np.linalg.norm(init_theta)
        
        theta = init_theta.copy()
        lambda_val = 0.0  # 初始拉格朗日乘子
        
        # 清空历史记录
        self.optimization_history = {
            'losses': [],
            'constraints': [],
            'lambda_vals': [],
            'grad_norms': [],
            'theta_history': [theta.copy()]
        }
        
        # 定义目标函数
        def objective(theta):
            return self.semi_supervised_risk(theta, D_labeled, D_unlabeled, kernel_func)
        
        # 有限差分法计算梯度
        def compute_gradient(theta, delta=1e-6):
            grad = np.zeros_like(theta)
            base_value = objective(theta)
            
            for i in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[i] += delta
                value_plus = objective(theta_plus)
                grad[i] = (value_plus - base_value) / delta
            
            return grad
        
        # 梯度下降优化
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(self.max_epochs):
            # 计算目标函数值和梯度
            loss = objective(theta)
            grad_f = compute_gradient(theta)
            
            # 计算约束相关项
            constraint = np.dot(theta, theta) - 1.0  # ||θ||^2 - 1
            grad_constraint = 2 * theta  # 约束的梯度
            
            # 拉格朗日函数的梯度
            grad_theta = grad_f + lambda_val * grad_constraint
            grad_lambda = constraint
            
            # 记录当前状态
            self.optimization_history['losses'].append(loss)
            self.optimization_history['constraints'].append(np.abs(constraint))
            self.optimization_history['lambda_vals'].append(lambda_val)
            self.optimization_history['grad_norms'].append(np.linalg.norm(grad_theta))
            
            # 检查收敛
            if loss < best_loss - self.tol:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 更新参数和拉格朗日乘子
            theta = theta - self.learning_rate * grad_theta
            print(epoch, theta, grad_theta)
            lambda_val = lambda_val + self.learning_rate * grad_lambda
            
            # 投影到单位球面（确保约束满足）
            theta_norm = np.linalg.norm(theta)
            if theta_norm > 1e-8:  # 避免除以零
                theta /= theta_norm
            
            self.optimization_history['theta_history'].append(theta.copy())
            
            # 每10个epoch打印一次信息
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss:.6f}, constraint={constraint:.6f}, "
                      f"lambda={lambda_val:.6f}, grad_norm={np.linalg.norm(grad_theta):.6f}")
        
        self.theta_hat = theta
        
        # 可视化优化过程
        optimization_history = {
            'losses': self.optimization_history['losses'],
            'lambda_vals': self.optimization_history['lambda_vals'],
            'theta_history': self.optimization_history['theta_history']
        }
        
        # 调用统一的优化可视化函数
        visualize_optimization_process(
            optimization_history, 
            f"semi_supervised_{unique_id}", 
            theta_true 
        )
        return self

    def visualize_optimization(self, unique_id):
        """可视化半监督优化过程"""
        # 确保visualization目录存在
        os.makedirs("visualization", exist_ok=True)
        
        # 获取历史数据
        losses = self.optimization_history['losses']
        constraints = self.optimization_history['constraints']
        lambda_vals = self.optimization_history['lambda_vals']
        grad_norms = self.optimization_history['grad_norms']
        theta_history = np.array(self.optimization_history['theta_history'])
        
        # 创建图表
        plt.figure(figsize=(18, 12))
        plt.suptitle(f"Semi-Supervised Optimization Process (ID: {unique_id})", fontsize=16)
        
        # 1. 损失函数变化
        plt.subplot(2, 2, 1)
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Semi-Supervised Risk')
        plt.grid(True)
        
        # 2. 约束违反程度
        plt.subplot(2, 2, 2)
        plt.plot(constraints, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('|Constraint|')
        plt.title('Constraint Violation')
        plt.yscale('log')
        plt.grid(True)
        
        # 3. 拉格朗日乘子变化
        plt.subplot(2, 2, 3)
        plt.plot(lambda_vals, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Lambda')
        plt.title('Lagrange Multiplier Evolution')
        plt.grid(True)
        
        # 4. 参数分量变化
        plt.subplot(2, 2, 4)
        for i in range(theta_history.shape[1]):
            plt.plot(theta_history[:, i], label=f'$\\theta_{i}$')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Values')
        plt.legend()
        plt.grid(True)
        
        # 调整布局防止重叠
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
        
        # 保存图像
        filename = f"visualization/semi_supervised_optimization_{unique_id}.png"
        plt.savefig(filename, dpi=300)
        print(f"Optimization visualization saved to: {filename}")
        plt.close()

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
                                       reg_models=["linear", "kernel", "mlp"],args = None,
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
        
        for trial in tqdm(range(4,trials+4), desc="progress"):
            # 生成数据
            D_labeled, D_unlabeled, theta_true, stats = DataGenerator.generate_pairwise_ranking_data(
                n_labeled, n_unlabeled, p=p, discrete_levels=discrete_levels, seed=42+trial, args=args, **data_kwargs
            )
            

            ExperimentRunner.print_dataset_stats(stats)
            
            # 监督基准
            print('run baseline')
            start_time = time()
            sup_theta = ExperimentRunner.supervised_baseline(D_labeled, KernelFunctions.pairwise_ranking_kernel, args=args, theta_true=theta_true)
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
                estimator = SemiSupervisedEstimator(
                    reg_model=model_type,
                    learning_rate=0.05,  # 调小学习率
                    max_epochs=1000        # 减少迭代次数（因为计算成本高）
                )
                estimator.fit(
                    D_labeled, 
                    D_unlabeled, 
                    KernelFunctions.pairwise_ranking_kernel,
                    unique_id=model_type+args.unique_id,
                    theta_true=theta_true
                )
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
                                         trials=10, args=None, **data_kwargs):
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
                n_labeled, n_unlabeled, p=p, seed=42+trial, args=args, **data_kwargs)
            
            if trial == 0:
                ExperimentRunner.print_dataset_stats(stats)
            
            # 定义核函数 (部分应用协方差矩阵)
            def gehan_kernel(theta, zi, zj):
                return KernelFunctions.smoothed_gehan_kernel(
                    theta, zi, zj, cov_matrix, n_labeled
                )
            
            # 监督基准
            start_time = time()
            sup_theta = ExperimentRunner.supervised_baseline(D_labeled, gehan_kernel, args=args)
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
                estimator.fit(D_labeled, D_unlabeled, gehan_kernel, theta_true=theta_true)
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
    def supervised_baseline(D_labeled, kernel_func, args, theta_true=None):
        """监督基准方法 - 使用拉格朗日乘数法进行单位模长约束优化"""
        n_features = len(D_labeled[0]) - 1 if isinstance(D_labeled[0], (list, np.ndarray)) else len(D_labeled[0]) - 2
        
        # 初始化参数和拉格朗日乘子
        theta = np.random.randn(n_features)
        theta /= np.linalg.norm(theta)  # 初始化为单位向量
        lambda_val = 0.0  # 初始拉格朗日乘子
        
        # 梯度下降参数
        learning_rate = 0.05
        max_epochs = 1000
        patience = 50
        tol = 1e-6
        
        # 记录优化过程
        losses = []
        constraints = []  # 约束违反程度 (||θ||^2 - 1)
        lambda_vals = []  # 拉格朗日乘子变化
        grad_norms = []   # 梯度范数
        theta_history = []  # θ历史记录
        theta_history.append(theta.copy())
        
        # 梯度下降优化
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(max_epochs):
            # 计算目标函数值和梯度
            total_loss, grad_f = ExperimentRunner.compute_objective_and_gradient(theta, D_labeled, kernel_func)
            
            # 计算约束相关项
            constraint = np.dot(theta, theta) - 1.0  # ||θ||^2 - 1
            grad_constraint = 2 * theta  # 约束的梯度
            
            # 拉格朗日函数的梯度
            grad_theta = grad_f + lambda_val * grad_constraint
            grad_lambda = constraint
            
            # 记录当前状态
            losses.append(total_loss)
            constraints.append(np.abs(constraint))
            lambda_vals.append(lambda_val)
            grad_norm = np.linalg.norm(grad_theta)
            grad_norms.append(grad_norm)
            
            # 检查收敛
            if total_loss < best_loss - tol:
                best_loss = total_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 更新参数和拉格朗日乘子
            theta = theta - learning_rate * grad_theta
            lambda_val = lambda_val + learning_rate * grad_lambda
            
            # 投影到单位球面（确保约束满足）
            theta_norm = np.linalg.norm(theta)
            if theta_norm > 1e-8:  # 避免除以零
                theta /= theta_norm
            
            theta_history.append(theta.copy())
            
            # 学习率衰减
            if epoch % 100 == 0 and epoch > 0:
                learning_rate *= 0.8
                print(f"Epoch {epoch}: loss={total_loss:.6f}, "
                      f"constraint={constraint:.6f}, lambda={lambda_val:.6f}, "
                      f"grad_norm={grad_norm:.6f}, lr={learning_rate:.6f}")
        
        optimization_history = {
            'losses': losses,
            'lambda_vals': lambda_vals,
            'theta_history': theta_history
        }
        
        # 可视化优化过程
        visualize_optimization_process(
            optimization_history, 
            f"supervised_{args.unique_id}", 
            theta_true
        )
        return theta
    @staticmethod
    def pairwise_kernel_with_grad(theta, zi, zj, kernel_func):
        """计算核函数值和梯度"""
        # 有限差分法计算梯度
        delta = 1e-6
        base_value = kernel_func(theta, zi, zj)
        
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += delta
            value_plus = kernel_func(theta_plus, zi, zj)
            
            grad[i] = (value_plus - base_value) / delta
        
        return base_value, grad
    @staticmethod
    def compute_objective_and_gradient(theta, D_labeled, kernel_func):
        """计算目标函数值和梯度"""
        n = len(D_labeled)
        total_loss = 0
        total_grad = np.zeros_like(theta)
        count = 0
        
        # 计算所有成对组合的损失和梯度
        for i in range(n):
            for j in range(i+1, n):
                # 计算损失和梯度
                loss_ij, grad_ij = ExperimentRunner.pairwise_kernel_with_grad(
                    theta, D_labeled[i], D_labeled[j], kernel_func
                )
                total_loss += loss_ij
                total_grad += grad_ij
                count += 1
        
        avg_loss = total_loss / count if count > 0 else 0
        avg_grad = total_grad / count if count > 0 else np.zeros_like(theta)
        
        return avg_loss, avg_grad
    
    @staticmethod
    def visualize_constrained_optimization(losses, constraints, lambda_vals, grad_norms, 
                                         theta_history, args, theta_true=None):
        """可视化带约束的优化过程"""
        plt.figure(figsize=(20, 15))
        
        # 损失函数变化
        plt.subplot(3, 2, 1)
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Function During Optimization')
        plt.grid(True)
        
        # 约束违反程度
        plt.subplot(3, 2, 2)
        plt.plot(constraints, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('|Constraint|')
        plt.title('Constraint Violation')
        plt.yscale('log')
        plt.grid(True)
        
        # 拉格朗日乘子变化
        plt.subplot(3, 2, 3)
        plt.plot(lambda_vals, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Lambda')
        plt.title('Lagrange Multiplier Evolution')
        plt.grid(True)
        
        # 梯度范数变化
        plt.subplot(3, 2, 4)
        plt.plot(grad_norms, 'm-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm')
        plt.yscale('log')
        plt.grid(True)
        
        # 参数分量变化
        plt.subplot(3, 2, 5)
        theta_history = np.array(theta_history)
        for i in range(theta_history.shape[1]):
            plt.plot(theta_history[:, i], label=f'$\\theta_{i}$')
            if theta_true is not None:
                plt.axhline(y=theta_true[i], color=f'C{i}', linestyle='--', 
                            label=f'True $\\theta_{i}$')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Values')
        plt.legend()
        plt.grid(True)
        
        # 参数空间轨迹
        if theta_history.shape[1] >= 2:
            plt.subplot(3, 2, 6)
            # 绘制单位圆
            angles = np.linspace(0, 2*np.pi, 100)
            plt.plot(np.cos(angles), np.sin(angles), 'k--', alpha=0.3)
            
            # 绘制轨迹
            plt.plot(theta_history[:, 0], theta_history[:, 1], 'b-')
            plt.scatter(theta_history[0, 0], theta_history[0, 1], c='green', 
                      s=100, marker='o', label='Start')
            plt.scatter(theta_history[-1, 0], theta_history[-1, 1], c='red', 
                      s=100, marker='x', label='End')
            if theta_true is not None and len(theta_true) >= 2:
                plt.scatter(theta_true[0], theta_true[1], c='black', 
                          s=150, marker='*', label='Optimum')
            plt.xlabel('$\\theta_0$')
            plt.ylabel('$\\theta_1$')
            plt.title('Parameter Space Trajectory')
            plt.axis('equal')
            plt.legend()
            plt.grid(True)
        
        plt.subplots_adjust(
            left=0.1, 
            right=0.95, 
            bottom=0.1, 
            top=0.9, 
            wspace=0.3,  # 水平间距
            hspace=0.4    # 垂直间距
        )
        plt.savefig(f'constrained_optimization_{args.unique_id}.png', dpi=300)
        #plt.show()
    
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
    def save_theta_data(theta_data, experiment_name, args):
        """保存theta数据到文件"""
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{experiment_name.replace(' ', '_')}_theta_data_{timestamp}_{args.unique_id}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(theta_data, f)
        
        print(f"theta saved to: {filename}")
        return filename


class ResultAnalyzer:
    """实验结果分析器"""
    
    @staticmethod
    def analyze_results(df, experiment_name, args):
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
        filename = f"{experiment_name.replace(' ', '_')}_results_{args.unique_id}.png"
        plt.savefig(filename, dpi=300)
        print(f"Result plot saved to: {filename}")
        #plt.show()
        
        # 统计显著性检验和表格生成
        stats_df = ResultAnalyzer.significance_test(df, experiment_name, args=args)
        
        return summary, stats_df

    @staticmethod
    def significance_test(df, experiment_name, args):
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
        csv_file = f"{experiment_name.replace(' ', '_')}_significance_results_{args.unique_id}.csv"
        stats_df.to_csv(csv_file, index=False)
        print(f"Significance results saved to: {csv_file}")
        
        # 保存为LaTeX表格
        latex_file = f"{experiment_name.replace(' ', '_')}_significance_results_{args.unique_id}.tex"
        stats_df.to_latex(latex_file, index=False, float_format="%.4f")
        print(f"LaTeX table saved to: {latex_file}")
        
        # 打印表格
        print("\nSignificance Analysis Table:")
        print(tabulate(stats_df, headers='keys', tablefmt='psql', floatfmt=".4f"))
        
        return stats_df
    
    @staticmethod
    def visualize_theta_comparison(theta_data, experiment_name, trial=0, args=None):
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
        
        filename = f"{experiment_name.replace(' ', '_')}_theta_comparison_trial_{trial}_{args.unique_id}.png"
        plt.savefig(filename, dpi=300)
        print(f"Theta comparison diagram saved to: {filename}")
        #plt.show()
    
    @staticmethod
    def sample_size_sensitivity(experiment_func, exp_name, n_values, m_ratio=5, trials=10,args=None,  **kwargs):
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
        
        filename = f"{exp_name.replace(' ', '_')}_sensitivity_{args.unique_id}.png"
        plt.savefig(filename, dpi=300)
        print(f"sensitivity diagram saved to: {filename}")
        #plt.show()
        
        return sens_df


def main():
    parser = argparse.ArgumentParser(description='semi-super-U')
    parser.add_argument('--experiment', type=str, default='pairwise', 
                        choices=['pairwise', 'survival', 'both'], 
                        help='simulation task')
    parser.add_argument('--n_labeled', type=int, default=100, 
                        help='n')
    parser.add_argument('--n_unlabeled', type=int, default=500, 
                        help='m')
    parser.add_argument('--trials', type=int, default=20, 
                        help='number of trails')
    parser.add_argument('--p', type=int, default=5, 
                        help='dimension')
    parser.add_argument('--discrete_levels', type=int, default=5, 
                        help='t')
    parser.add_argument('--censor_rate', type=float, default=0.25, 
                        help='')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='')
    parser.add_argument('--unique_id', type=str, default='n100m500p5t5pair', 
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
            reg_models=["linear", "kernel",  "mlp"],
            args = args
        )
        
        # 保存θ数据
        theta_file = ExperimentRunner.save_theta_data(pairwise_theta_data, "pairwise_rank", args=args)
        
        # 分析结果
        pairwise_summary, pairwise_stats = ResultAnalyzer.analyze_results(pairwise_df, "pairwise_rank_exam", args=args)
        
        # 可视化参数比较
        ResultAnalyzer.visualize_theta_comparison(pairwise_theta_data, "pairwise_rank", trial=0, args=args)
        
        # 样本量敏感性分析
        n_values = [50, 200, 500]
        sensitivity_df = ResultAnalyzer.sample_size_sensitivity(
            ExperimentRunner.run_pairwise_ranking_experiment,
            "pairwise_rank",
            n_values,
            trials=20,
            args=args
        )
    
    # 运行生存分析实验
    if args.experiment in ['survival', 'both']:
        survival_df, survival_theta_data = ExperimentRunner.run_survival_analysis_experiment(
            n_labeled=args.n_labeled,
            n_unlabeled=args.n_unlabeled,
            p=args.p,
            trials=args.trials,
            args = args,
            reg_models=["linear", "kernel", "mlp"],
            censor_rate=args.censor_rate,
            error_dist="gumbel"
        )
        
        # 保存θ数据
        theta_file = ExperimentRunner.save_theta_data(survival_theta_data, "survival_analyze", args=args)
        
        # 分析结果
        survival_summary, survival_stats = ResultAnalyzer.analyze_results(survival_df, "survival_analyze_exam",args=args)
        
        # 可视化参数比较
        ResultAnalyzer.visualize_theta_comparison(survival_theta_data, "survival_analyze", trial=0, args=args)
        
        # 样本量敏感性分析
        n_values = [50, 100, 200, 500]
        sensitivity_df = ResultAnalyzer.sample_size_sensitivity(
            ExperimentRunner.run_survival_analysis_experiment,
            "survival_analyze",
            n_values,
            trials=5,
            args = args
        )
    
    print("\n All experiment finished!")


if __name__ == "__main__":
    main()