import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from safetensors.numpy import save_file, load_file
from pathlib import Path
from typing import Union
from PIL import Image
import json


class GMM:
    def __init__(self, n_components: int, data_dim: int):
        self.n_components = n_components
        self.data_dim = data_dim
        self.means = np.random.rand(n_components, data_dim)
        self.covs = np.array([np.eye(data_dim) for _ in range(n_components)])
        self.pi = np.ones(n_components) / n_components

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        params = load_file(path / "gmm.safetensors")
        model = cls(**config)
        model.means = params["means"]
        model.covs = params["covs"]
        model.pi = params["pi"]
        return model

    def fit(self, X: np.ndarray, max_iter: int = 100):

        kmeans = KMeans(n_clusters=self.n_components)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_
        self.covs = np.array([np.cov(X[kmeans.labels_ == i].T) for i in range(self.n_components)])
        self.pi = np.array([np.mean(kmeans.labels_ == i) for i in range(self.n_components)])

        for _ in tqdm(range(max_iter)):
            gamma = self._e_step(X)
            self._m_step(X, gamma)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        K = self.n_components
        gamma = np.zeros((N, K))
        for k in range(K):
            inv_cov = np.linalg.inv(self.covs[k])
            det = np.linalg.det(self.covs[k])
            gamma[:, k] = self.pi[k] * self._gaussian(X, self.means[k], inv_cov, det)
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma = gamma / gamma_sum

        return gamma

    def _m_step(self, X: np.ndarray, gamma: np.ndarray):
        N, D = X.shape
        K = self.n_components
        N_k = np.sum(gamma, axis=0)
        self.pi = N_k / N
        self.means = (gamma.T @ X) / N_k[:, np.newaxis]
        for k in range(K):
            X_centered = X - self.means[k]
            gamma_k = gamma[:, k].reshape(-1, 1)
            cov_k = (gamma_k * X_centered).T @ X_centered / N_k[k]
            self.covs[k] = cov_k + 1e-6 * np.eye(D)

    def _gaussian(self, X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, det: float) -> np.ndarray:
        N, D = X.shape
        diff = X - mean
        exponent = np.sum(diff @ inv_cov * diff, axis=1)
        log_prob = -0.5 * exponent - 0.5 * np.log(det) - D / 2 * np.log(2 * np.pi)  # Prevent overflow
        return np.exp(log_prob)

    def predict(self, X: np.ndarray) -> np.ndarray:
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"n_components": self.n_components, "data_dim": self.data_dim}
        params = {"means": self.means, "covs": self.covs, "pi": self.pi}
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        save_file(params, path / "gmm.safetensors")


class PCA:
    def __init__(self, dim: int):
        self.dim = dim
        self.components = None
        self.mean = None

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]):

        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        params = load_file(path / "pca.safetensors")
        model = cls(**config)
        model.components = params["components"]
        model.mean = params["mean"]
        return model

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        self.components = sorted_eigenvectors[:, :self.dim].T

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X - self.mean
        return X @ self.components.T

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        return X_pca @ self.components + self.mean

    def save_pretrained(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"dim": self.dim}
        params = {"components": self.components, "mean": self.mean}
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        save_file(params, path / "pca.safetensors")


# 5
def sample_from_gmm(gmm: GMM, pca: PCA, label: int, path: Union[str, Path]):
    np.random.seed()
    mean = gmm.means[label]
    cov = gmm.covs[label]
    try:
        sample_pca = np.random.multivariate_normal(mean, cov, size=1)
    except np.linalg.LinAlgError:
        cov += 1e-6 * np.eye(gmm.data_dim)
        sample_pca = np.random.multivariate_normal(mean, cov, size=1)
    sample_original = pca.inverse_transform(sample_pca)
    sample_original = np.clip(sample_original, 0, 1) * 255
    sample_original = sample_original.astype(np.uint8)
    sample_image = sample_original.reshape(28, 28)
    sample = Image.fromarray(sample_image, mode="L")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sample.save(path / "gmm_sample.png")