"""
models.py — โมเดลจำแนกประเภท เขียนเองทั้งหมด (ไม่ใช้ scikit-learn / TensorFlow / PyTorch)

  1. KNNClassifier         — k เพื่อนบ้านใกล้ที่สุด (cosine similarity, ถ่วงน้ำหนัก)
  2. NeuralNetClassifier   — โครงข่ายประสาทเทียมแบบ feed-forward
        hidden=()          -> Softmax Regression (Multinomial Logistic Regression)
        hidden=(128, 64)   -> Multi-Layer Perceptron (MLP)
     เขียน forward pass, backpropagation, ReLU, Dropout, Softmax, Cross-entropy,
     L2 regularization, Adam optimizer, Early stopping ด้วยตนเอง
"""
import math

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, n):
    out = np.zeros((len(y), n), np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# ======================================================================
class KNNClassifier:
    name = "kNN"

    def __init__(self, k=7, n_classes=None):
        self.k = k
        self.n_classes = n_classes

    def fit(self, X, y, X_val=None, y_val=None, log=None):
        self.X_ = self._unit(X)
        self.y_ = np.asarray(y, np.int64)
        self.n_classes = self.n_classes or int(self.y_.max()) + 1
        self.history = {}
        return self

    @staticmethod
    def _unit(X):
        X = np.asarray(X, np.float32)
        return X / (np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-8)

    def predict_proba(self, X):
        S = self._unit(X) @ self.X_.T                   # cosine similarity
        k = min(self.k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        w = np.maximum(sims, 0) ** 4 + 1e-6               # เพื่อนบ้านที่ใกล้กว่าได้น้ำหนักมากกว่า
        P = np.full((len(X), self.n_classes), 0.01, np.float32)   # Laplace smoothing
        labels = self.y_[idx]
        for c in range(self.n_classes):
            P[:, c] += (w * (labels == c)).sum(axis=1)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def get_params(self):
        return {"X": self.X_, "y": self.y_, "k": np.array(self.k), "n_classes": np.array(self.n_classes)}

    @staticmethod
    def from_params(p):
        m = KNNClassifier(int(p["k"]), int(p["n_classes"]))
        m.X_, m.y_ = p["X"].astype(np.float32), p["y"].astype(np.int64)
        return m


# ======================================================================
class NeuralNetClassifier:
    def __init__(self, n_in, n_classes, hidden=(128, 64), lr=1e-3, l2=1e-4, dropout=0.25,
                 epochs=150, batch_size=64, patience=20, label_smoothing=0.05,
                 input_noise=0.05, seed=0):
        self.sizes = [n_in, *hidden, n_classes]
        self.n_classes = n_classes
        self.lr, self.l2, self.dropout = lr, l2, dropout
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.label_smoothing, self.input_noise = label_smoothing, input_noise
        self.rng = np.random.default_rng(seed)
        self.name = "Softmax Regression" if not hidden else "MLP " + "-".join(map(str, self.sizes))
        self._init_weights()

    # ---------------------------------------------------------- พารามิเตอร์
    def _init_weights(self):
        """He initialization: W ~ N(0, sqrt(2 / fan_in))"""
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append((self.rng.normal(0, math.sqrt(2.0 / fan_in), (fan_in, fan_out))).astype(np.float32))
            self.b.append(np.zeros(fan_out, np.float32))

    def _params(self):
        return self.W + self.b

    # ---------------------------------------------------------- forward / backward
    def _forward(self, X, train=False):
        acts, masks = [X], []
        a = X
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            if i < L - 1:
                a = np.maximum(z, 0)                                      # ReLU
                if train and self.dropout > 0:                             # inverted dropout
                    m = (self.rng.random(a.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                    a = a * m
                    masks.append(m)
                else:
                    masks.append(None)
                acts.append(a)
            else:
                a = softmax(z)
        return a, acts, masks

    def _backward(self, P, Y, acts, masks):
        """backpropagation ของ softmax + cross-entropy: dL/dz = (P - Y) / n"""
        n = len(Y)
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        delta = (P - Y) / n
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)                                # อนุพันธ์ ReLU
        return grads_W + grads_b

    def _loss(self, P, y):
        return float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())

    # ---------------------------------------------------------- training
    def fit(self, X, y, X_val=None, y_val=None, log=None):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.int64)
        Y = one_hot(y, self.n_classes)
        eps = self.label_smoothing
        Y = Y * (1 - eps) + eps / self.n_classes
        params = self._params()
        m = [np.zeros_like(p) for p in params]                               # Adam: โมเมนต์ที่ 1
        v = [np.zeros_like(p) for p in params]                               # Adam: โมเมนต์ที่ 2
        beta1, beta2, adam_eps, t = 0.9, 0.999, 1e-8, 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best, best_state, wait = float("inf"), None, 0

        for ep in range(1, self.epochs + 1):
            order = self.rng.permutation(len(X))
            for s in range(0, len(X), self.batch_size):
                idx = order[s:s + self.batch_size]
                xb = X[idx]
                if self.input_noise > 0:                                     # data augmentation ระดับฟีเจอร์
                    xb = xb + self.rng.normal(0, self.input_noise, xb.shape).astype(np.float32)
                P, acts, masks = self._forward(xb, train=True)
                grads = self._backward(P, Y[idx], acts, masks)
                t += 1
                params = self._params()
                for j, (p, g) in enumerate(zip(params, grads)):
                    m[j] = beta1 * m[j] + (1 - beta1) * g
                    v[j] = beta2 * v[j] + (1 - beta2) * g * g
                    m_hat = m[j] / (1 - beta1 ** t)
                    v_hat = v[j] / (1 - beta2 ** t)
                    p -= (self.lr * m_hat / (np.sqrt(v_hat) + adam_eps)).astype(np.float32)

            Ptr = self.predict_proba(X)
            self.history["train_loss"].append(self._loss(Ptr, y))
            self.history["train_acc"].append(float((Ptr.argmax(1) == y).mean()))
            if X_val is not None and len(X_val):
                Pv = self.predict_proba(X_val)
                vl = self._loss(Pv, y_val)
                self.history["val_loss"].append(vl)
                self.history["val_acc"].append(float((Pv.argmax(1) == y_val).mean()))
                if vl < best - 1e-4:                                         # early stopping
                    best, wait = vl, 0
                    best_state = [p.copy() for p in self._params()]
                else:
                    wait += 1
                if log and (ep % 10 == 0 or ep == 1):
                    log(f"    epoch {ep:3d}  loss {self.history['train_loss'][-1]:.4f}  "
                        f"val_loss {vl:.4f}  val_acc {self.history['val_acc'][-1] * 100:.1f}%")
                if wait >= self.patience:
                    if log:
                        log(f"    early stopping ที่ epoch {ep} (val_loss ไม่ดีขึ้น {self.patience} รอบ)")
                    break
        if best_state is not None:
            n = len(self.W)
            self.W, self.b = best_state[:n], best_state[n:]
        return self

    # ---------------------------------------------------------- inference
    def predict_proba(self, X):
        return self._forward(np.asarray(X, np.float32), train=False)[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def get_params(self):
        d = {"sizes": np.array(self.sizes)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        return d

    @staticmethod
    def from_params(p):
        sizes = [int(s) for s in p["sizes"]]
        m = NeuralNetClassifier(sizes[0], sizes[-1], tuple(sizes[1:-1]))
        m.W = [p[f"W{i}"].astype(np.float32) for i in range(len(sizes) - 1)]
        m.b = [p[f"b{i}"].astype(np.float32) for i in range(len(sizes) - 1)]
        return m
