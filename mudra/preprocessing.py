"""
preprocessing.py — Feature scaling และ PCA (ลดมิติข้อมูล) เขียนเองทั้งหมด
ไม่ใช้ np.linalg — หาค่าเวกเตอร์ลักษณะเฉพาะด้วย Orthogonal iteration + Jacobi method
"""
import math

import numpy as np


class StandardScaler:
    """z = (x - mean) / std   (เรียนรู้ mean/std จากชุด train เท่านั้น)"""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _norm(v):
    return math.sqrt(float(v @ v))


def orthonormalize(Z, rng=None):
    """
    QR แบบ Classical Gram-Schmidt ทำซ้ำ 2 รอบ (CGS2) — คืนเมทริกซ์ Q ที่คอลัมน์ตั้งฉากและยาว 1
    """
    rng = rng or np.random.default_rng(0)
    d, k = Z.shape
    Q = np.zeros((d, k), np.float64)
    for j in range(k):
        v = Z[:, j].astype(np.float64).copy()
        for _ in range(2):
            if j:
                v -= Q[:, :j] @ (Q[:, :j].T @ v)
        n = _norm(v)
        if n < 1e-10:                     # คอลัมน์ขึ้นต่อกัน -> สุ่มเวกเตอร์ใหม่
            v = rng.normal(size=d)
            for _ in range(2):
                if j:
                    v -= Q[:, :j] @ (Q[:, :j].T @ v)
            n = _norm(v)
        Q[:, j] = v / n
    return Q


def jacobi_eigh(A, tol=1e-10, max_sweeps=50):
    """
    หาค่าลักษณะเฉพาะ/เวกเตอร์ลักษณะเฉพาะของเมทริกซ์สมมาตร ด้วยวิธีหมุนของ Jacobi
    (หมุนทีละคู่ (p,q) เพื่อทำให้สมาชิกนอกแนวทแยงเป็นศูนย์ จนเมทริกซ์เกือบเป็นเมทริกซ์ทแยง)
    """
    A = np.array(A, np.float64)
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = math.sqrt(float(((A - np.diag(np.diag(A))) ** 2).sum()))
        if off < tol * max(1.0, math.sqrt(float((np.diag(A) ** 2).sum()))):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) < 1e-14:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                ap, aq = A[:, p].copy(), A[:, q].copy()
                A[:, p], A[:, q] = c * ap - s * aq, s * ap + c * aq
                ap, aq = A[p, :].copy(), A[q, :].copy()
                A[p, :], A[q, :] = c * ap - s * aq, s * ap + c * aq
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p], V[:, q] = c * vp - s * vq, s * vp + c * vq
    return np.diag(A).copy(), V


class PCA:
    """
    Principal Component Analysis
      1. covariance C = Xc^T Xc / (n-1)
      2. Orthogonal (subspace) iteration: Q <- orth(C Q) หาซับสเปซของค่าลักษณะเฉพาะที่ใหญ่ที่สุด k ตัว
      3. Rayleigh-Ritz: T = Q^T C Q แล้วแยกค่าลักษณะเฉพาะด้วย Jacobi -> ได้แกนหลักเรียงจากมากไปน้อย
      4. เลือกจำนวนแกนที่อธิบายความแปรปรวนได้ >= variance_target
    """

    def __init__(self, max_components=100, variance_target=0.97, n_iter=40, seed=0):
        self.max_components = max_components
        self.variance_target = variance_target
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, log=None):
        X = np.asarray(X, np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        C = Xc.T @ Xc / max(1, n - 1)
        total_var = float(np.trace(C))
        k = min(self.max_components + 10, d, n)
        rng = np.random.default_rng(self.seed)
        Q = orthonormalize(rng.normal(size=(d, k)), rng)
        for it in range(self.n_iter):
            Q = orthonormalize(C @ Q, rng)
        T = Q.T @ C @ Q
        T = (T + T.T) / 2
        evals, V = jacobi_eigh(T)
        order = np.argsort(-evals)
        evals, U = evals[order], Q @ V[:, order]
        ratio = np.maximum(evals, 0) / total_var
        cum = np.cumsum(ratio)
        n_keep = int(np.searchsorted(cum, self.variance_target) + 1)
        n_keep = max(2, min(n_keep, self.max_components, len(evals)))
        self.components_ = U[:, :n_keep].T.astype(np.float32)      # (k, d)
        self.explained_variance_ = evals[:n_keep]
        self.explained_variance_ratio_ = ratio[:n_keep]
        self.scale_ = float(math.sqrt(max(self.explained_variance_.mean(), 1e-12)))
        if log:
            log(f"  PCA: {d} มิติ -> {n_keep} มิติ (อธิบายความแปรปรวนได้ {cum[n_keep - 1] * 100:.1f}%)")
        return self

    def transform(self, X):
        Z = (np.asarray(X, np.float32) - self.mean_.astype(np.float32)) @ self.components_.T
        return (Z / self.scale_).astype(np.float32)

    def fit_transform(self, X, log=None):
        return self.fit(X, log).transform(X)
