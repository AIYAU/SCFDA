import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class FuzzyKNN:
    def __init__(self, n_neighbors=5, fuzziness=2):
        """
        n_neighbors: 近邻的数量
        fuzziness: 模糊因子，控制隶属度的衰减
        """
        self.n_neighbors = n_neighbors
        self.fuzziness = fuzziness

    def fit(self, X_train, y_train):
        """
        训练数据集（保存特征和标签）
        X_train: 训练样本的特征矩阵 (n_samples, n_features)
        y_train: 训练样本的标签 (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)  # 提取所有类别标签

    def predict(self, X_test):
        """
        预测测试样本的类别标签
        X_test: 测试样本的特征矩阵 (n_samples, n_features)
        返回值: 预测标签 (n_samples,)
        """
        # 初始化隶属度矩阵 (每个测试样本对每个类别的隶属度)
        membership_matrix = np.zeros((X_test.shape[0], len(self.classes_)))

        # 计算测试样本和训练样本之间的距离矩阵
        distances = euclidean_distances(X_test, self.X_train)

        # 对每个测试样本逐个处理
        for i, test_sample_dist in enumerate(distances):
            # 找到最近的 n_neighbors 个邻居及其对应的标签
            neighbor_indices = np.argsort(test_sample_dist)[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_distances = test_sample_dist[neighbor_indices]

            # 防止除零错误，对距离进行微小调整
            neighbor_distances = np.maximum(neighbor_distances, 1e-6)

            # 计算隶属度 (使用距离的倒数和模糊因子)
            weights = 1.0 / (neighbor_distances ** (2 / (self.fuzziness - 1)))

            # 归一化权重，使所有邻居的权重和为 1
            weights /= np.sum(weights)

            # 根据邻居的标签，更新隶属度矩阵
            for j, class_label in enumerate(self.classes_):
                # 确保标签匹配，并计算该类的隶属度
                mask = (neighbor_labels == class_label)

                if np.any(mask):
                    membership_matrix[i, j] = np.sum(weights[mask])
                else:
                    membership_matrix[i, j] = 0.0  # 如果该类别没有邻居，隶属度设为 0

        # 根据最大隶属度选择最终的预测标签
        predicted_labels = np.argmax(membership_matrix, axis=1)
        return self.classes_[predicted_labels]


# 示例用法
if __name__ == "__main__":
    # 生成一些示例数据
    X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 8], [7, 8]])
    y_train = np.array([0, 0, 0, 1, 1])  # 两个类别：0 和 1
    X_test = np.array([[2, 2], [6, 7]])

    # 初始化并训练模糊KNN分类器
    fuzzy_knn = FuzzyKNN(n_neighbors=3, fuzziness=2)
    fuzzy_knn.fit(X_train, y_train)

    # 预测测试样本的类别标签
    predicted_labels = fuzzy_knn.predict(X_test)
    print("预测标签：", predicted_labels)
