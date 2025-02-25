import pandas as pd
import yaml
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from typing import Tuple
from model import BaseModel

from utils import (
    TrainConfigR,
    TrainConfigC,
    DataLoader,
    Parameter,
    Loss,
    SGD,
    GD,
    save,
)

# landepen = 0

def data_preprocessing_regression(data_path: str, saved_to_disk: bool = False) -> Dataset:

    data_path = "E:/dev/mech/1/USTC-ML24-Fall-main/lab1/src/sgemm_train.csv"
    df = pd.read_csv(data_path)
    numerical_features = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB', 'KWI', 'VWM', 'VWN']
    binary_features = ['STRM', 'STRN', 'SA', 'SB']
    means = df[numerical_features].mean()
    stds = df[numerical_features].std()
    stds_replaced = stds.replace(0, 1)
    df[numerical_features] = (df[numerical_features] - means) / stds_replaced
    df['Run_time'] = np.log(df['Run_time'])
    dataset = Dataset.from_pandas(df)
    return dataset


def data_split_regression(dataset: Dataset, batch_size: int, shuffle: bool) -> Tuple[DataLoader]:

    dataset_split = dataset.train_test_split(test_size=0.1)
    train_valid_split = dataset_split['train'].train_test_split(test_size=0.1/0.9)

    trainset = train_valid_split['train']
    validset = train_valid_split['test']
    testset = dataset_split['test']
    combinedset = Dataset.from_pandas(pd.concat([trainset.to_pandas(), validset.to_pandas()], ignore_index=True))
    # combinedset = Dataset.from_pandas(pd.concat([combinedset.to_pandas(), testset.to_pandas()], ignore_index=True))
    train_loader = DataLoader(combinedset, batch_size, shuffle)
    test_loader = DataLoader(testset, batch_size, shuffle=False, train=False)
    return train_loader, test_loader


class LinearRegression(BaseModel):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = Parameter(np.zeros((in_features, out_features)))
        self.bias = Parameter(np.zeros((1, out_features)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = np.dot(x, self.weight) + self.bias
        return y

    def weight_square(self):
        return np.sum(self.weight ** 2)

    def weight_sum(self):
        return np.sum(self.weight)


class MSELoss(Loss):

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:

        batch_size = y_pred.shape[0]
        dL_dy_pred = (2 / batch_size) * (y_pred - y_true)
        dL_dW = x.T @ dL_dy_pred
        dL_db = np.sum(dL_dy_pred, axis=0, keepdims=True)

        grads = {'weight': dL_dW, 'bias': dL_db}
        return grads


class TrainerR:

    def __init__(self, model: BaseModel, train_loader: DataLoader, loss: Loss, optimizer: SGD, config: TrainConfigR, results_path: Path):
        self.model = model
        self.train_loader = train_loader
        self.criterion = loss
        self.opt = optimizer
        self.cfg = config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps = len(self.train_loader) * self.cfg.epochs
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):

        loss_list = []
        with tqdm(
                initial=self.step,
                total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:  # self.train_num_steps

                batch = next(iter(self.train_loader))

                X = batch[:, :-1]
                y_true = batch[:, -1].reshape(-1, 1)
                y_pred = self.model.predict(X)

                current_loss = self.criterion(y_pred, y_true)
                loss_list.append(current_loss)

                pbar.set_description(f"Loss:{current_loss:.4f}")

                grads = self.criterion.backward(x=X, y_pred=y_pred, y_true=y_true)
                # grads['weight'] += 2 * landepen * self.model.weight_sum()
                self.opt.step(grads=grads)
                self.step += 1
                pbar.update()

        plt.plot(loss_list)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(self.results_path / 'regression_loss_list.png')
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")


def eval_LinearRegression(model: LinearRegression, loader: DataLoader) -> Tuple[float, float, float]:

    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        features = batch[:, :-1]
        target = batch[:, -1].reshape(-1, 1)

        features = np.array(features)
        target = np.array(target)

        preds = model.predict(features)

        all_preds.append(preds)
        all_targets.append(target)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    pred_mean = all_preds.mean()
    target_mean = all_targets.mean()
    relative_error1 = np.abs(pred_mean - target_mean) / target_mean

    relative_error2 = np.abs(all_preds - all_targets).mean() / target_mean

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - target_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    mse = np.mean((all_targets - all_preds) ** 2)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean of predictions: {pred_mean}")
    print(f"Mean of targets: {target_mean}")
    print(f"Relative Error: {relative_error1}")
    # print(f"Relative Error 2: {relative_error2}")
    print(f"R^2 Score: {r2}")

    return pred_mean, relative_error1


def data_preprocessing_classification(data_path: str, mean: float, saved_to_disk: bool = False) -> Dataset:

    data_path = "E:/dev/mech/1/USTC-ML24-Fall-main/lab1/src/sgemm_train1.csv"
    df = pd.read_csv(data_path)
    df['label'] = (np.log(df['Run_time']) > mean).astype(int)
    df.drop(columns=['Run_time'], inplace=True)
    dataset = Dataset.from_pandas(df)

    return dataset


def data_split_classification(dataset: Dataset) -> Tuple[Dataset]:

    dataset_split = dataset.train_test_split(test_size=0.1)
    train_valid_split = dataset_split['train'].train_test_split(test_size=0.1/0.9)

    trainset = train_valid_split['train']
    validset = train_valid_split['test']
    testset = dataset_split['test']

    combinedset = Dataset.from_pandas(pd.concat([trainset.to_pandas(), validset.to_pandas()], ignore_index=True))
    combinedset = Dataset.from_pandas(pd.concat([combinedset.to_pandas(), testset.to_pandas()], ignore_index=True))

    return combinedset, testset


class LogisticRegression(BaseModel):

    def __init__(self, in_features: int):

        super().__init__()
        self.beta = Parameter(np.zeros((in_features + 1, 1)))

    def predict(self, x: np.ndarray) -> np.ndarray:

        a = np.maximum(np.dot(x, self.beta), -500)  # ?
        b = 1 / (1 + np.exp(-a))
        return b


class BCELoss(Loss):

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:

        e = 1e-12
        y_pred = np.clip(y_pred, e, 1 - e)
        bce_loss = -np.mean(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bce_loss

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:

        batch_size = np.shape(x)[0]
        dL_dy_pred = y_pred - y_true
        grad = np.dot(x.T, dL_dy_pred) / batch_size
        return {"beta": grad}


class TrainerC:

    def __init__(self, model: BaseModel, dataset: np.ndarray, loss: Loss, optimizer: GD, config: TrainConfigC, results_path: Path):
        self.model = model
        self.dataset = dataset
        self.criterion = loss
        self.opt = optimizer
        self.cfg = config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps = self.cfg.steps
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:

                x_batch = self.dataset[:, :-1]
                x_batch = np.insert(x_batch, -1, 1, axis=1)
                y_batch = self.dataset[:, -1]

                y_pred = self.model.predict(x_batch)
                loss = self.criterion(y_pred, y_batch.reshape(-1, 1))
                loss_list.append(loss)

                pbar.set_description(f"Loss: {loss:.6f}")
                gradients = self.criterion.backward(x_batch, y_pred, y_batch.reshape(-1, 1))
                self.opt.step(gradients)

                self.step += 1
                pbar.update()

        with open(self.results_path / 'loss_list.txt', 'w') as f:
            print(loss_list, file=f)
        plt.plot(loss_list)
        plt.savefig(self.results_path / 'regression_loss_list.png')
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")


def eval_LogisticRegression(model: LogisticRegression, dataset: np.ndarray) -> float:

    model.eval()
    X = dataset[:, :-1]
    y_true = dataset[:, -1]

    y_true = y_true.flatten()
    bias = np.ones((X.shape[0], 1))
    X_bias = np.hstack((X, bias))

    y_pred_probs = model(X_bias)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int).flatten()

    correct = np.sum(y_pred_labels == y_true)
    accuracy = correct / y_true.shape[0]
    return accuracy
