"""Anomaly scorers and disjoint calibration subsets."""

import numpy as np
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


class Autoencoder:
    """Tanh encoder and decoder with manual Adam, optionally with a Gaussian latent."""

    def __init__(self, seed: int, variational: bool, epochs: int = 16):
        self.rng = np.random.default_rng(seed)
        self.variational = variational
        self.epochs = epochs

    def fit(self, x: np.ndarray) -> None:
        width = x.shape[1]
        latent = max(2, min(8, width // 2))
        hidden = 24
        shapes = [(width, hidden), (hidden, latent * (1 + self.variational)),
                  (latent, hidden), (hidden, width)]
        self.params = []
        for fan_in, fan_out in shapes:
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.params.extend([self.rng.uniform(-limit, limit, (fan_in, fan_out)), np.zeros(fan_out)])
        first = [np.zeros_like(p) for p in self.params]
        second = [np.zeros_like(p) for p in self.params]
        step = 0
        for _ in range(self.epochs):
            order = self.rng.permutation(len(x))
            for start in range(0, len(x), 128):
                step += 1
                batch = x[order[start:start + 128]]
                for i, grad in enumerate(self._gradients(batch)):
                    first[i] = .9 * first[i] + .1 * grad
                    second[i] = .999 * second[i] + .001 * grad ** 2
                    self.params[i] -= .003 * (first[i] / (1 - .9 ** step)) / (
                        np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8
                    )

    def _forward(self, x: np.ndarray, training: bool):
        w1, b1, w2, b2, w3, b3, w4, b4 = self.params
        encoded = np.tanh(x @ w1 + b1)
        latent = encoded @ w2 + b2
        aux = None
        if self.variational:
            mean, raw = np.split(latent, 2, axis=1)
            logvar = 6 * (expit(raw) - .5)
            deviation = np.exp(logvar / 2)
            noise = self.rng.normal(size=mean.shape) if training else np.zeros_like(mean)
            latent = mean + noise * deviation
            aux = mean, raw, logvar, deviation, noise
        decoded = np.tanh(latent @ w3 + b3)
        return encoded, latent, decoded, decoded @ w4 + b4, aux

    def _gradients(self, x: np.ndarray) -> list[np.ndarray]:
        w1, _, w2, _, w3, _, w4, _ = self.params
        encoded, latent, decoded, prediction, aux = self._forward(x, True)
        residual = 2 * (prediction - x) / x.size
        decoder_grad = (residual @ w4.T) * (1 - decoded ** 2)
        latent_grad = decoder_grad @ w3.T
        if aux is not None:
            mean, raw, logvar, deviation, noise = aux
            beta = .02 / mean.size
            mean_grad = latent_grad + beta * mean
            logvar_grad = .5 * (latent_grad * deviation * noise + beta * np.expm1(logvar))
            sigmoid = expit(raw)
            latent_grad = np.concatenate([mean_grad, logvar_grad * 6 * sigmoid * (1 - sigmoid)], axis=1)
        encoder_grad = (latent_grad @ w2.T) * (1 - encoded ** 2)
        return [x.T @ encoder_grad, encoder_grad.sum(0), encoded.T @ latent_grad,
                latent_grad.sum(0), latent.T @ decoder_grad, decoder_grad.sum(0),
                decoded.T @ residual, residual.sum(0)]

    def score(self, x: np.ndarray) -> np.ndarray:
        _, _, _, reconstructed, _ = self._forward(x, False)
        return np.mean((x - reconstructed) ** 2, axis=1)


class Scorer:
    """Fit preprocessing and models on fit rows, then rank on separate calibration rows."""

    def __init__(self, name: str, seed: int):
        self.names = ["iforest", "lof", "ae"] if name == "ensemble" else [name]
        self.seed = seed

    def fit(self, train: np.ndarray, reference: np.ndarray) -> None:
        self.scaler = RobustScaler().fit(train)
        train = self._transform(train)
        reference = self._transform(reference)
        self.members = []
        for name in self.names:
            if name == "iforest":
                model = IsolationForest(n_estimators=48, max_samples=min(256, len(train)), random_state=self.seed)
            elif name == "lof":
                model = LocalOutlierFactor(n_neighbors=20, novelty=True)
            elif name == "ocsvm":
                model = OneClassSVM(gamma="scale", nu=.05)
            elif name in {"ae", "vae"}:
                model = Autoencoder(self.seed, name == "vae")
            else:
                raise ValueError(f"Unknown anomaly model {name}")
            model.fit(train)
            self.members.append((model, np.sort(self._raw(model, reference))))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x).clip(-15, 15)

    @staticmethod
    def _raw(model, x: np.ndarray) -> np.ndarray:
        return model.score(x) if isinstance(model, Autoencoder) else -model.score_samples(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        x = self._transform(x)
        ranks = []
        for model, reference in self.members:
            values = self._raw(model, x)
            positions = np.searchsorted(reference, values, side="left")
            # Preserve ordering in the tails beyond the calibration extrema.
            low, high = reference[0], reference[-1]
            scale = max(high - low, 1e-9)
            tails = np.arcsinh(np.minimum(values - low, 0) / scale)
            tails += np.arcsinh(np.maximum(values - high, 0) / scale)
            ranks.append((positions + tails) / len(reference))
        return np.mean(ranks, axis=0)
