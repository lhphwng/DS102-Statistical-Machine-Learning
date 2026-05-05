import numpy as np
import time

class SoftMarginSVM:
    def __init__(
        self,
        C: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        lr_schedule: str = "constant",   # "constant" | "decay"
        random_state: int = 42,
    ):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.random_state = random_state
        self.w = None
        self.b = None
        self.losses = []

    # ── helpers ──────────────────────────────
    def _hinge_loss(self, X, y):
        margins = y * (X @ self.w + self.b)           # (n,)
        hinge   = np.maximum(0.0, 1.0 - margins)
        return 0.5 * np.dot(self.w, self.w) + self.C * hinge.mean()

    def _current_lr(self, epoch: int) -> float:
        if self.lr_schedule == "decay":
            return self.lr / (1.0 + 0.1 * epoch)
        return self.lr

    # ── fit ──────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X : (n, d)  float32
        y : (n,)    int  with values {0, 1}
        """
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        y_pm = (2 * y - 1).astype(np.float32)         # map {0,1} → {-1,+1}

        # Xavier-like init
        self.w = rng.normal(0, 1.0 / np.sqrt(d), d).astype(np.float32)
        self.b = np.float32(0.0)

        print(f"\n{'─'*55}")
        print(f"  Soft-Margin SVM — NumPy SGD Training")
        print(f"  C={self.C}  lr={self.lr}  epochs={self.epochs}  batch={self.batch_size}")
        print(f"{'─'*55}")

        for epoch in range(self.epochs):
            t0  = time.time()
            lr  = self._current_lr(epoch)
            idx = rng.permutation(n)
            X_s = X[idx];  y_s = y_pm[idx]

            for start in range(0, n, self.batch_size):
                xb = X_s[start:start + self.batch_size]   # (bs, d)
                yb = y_s[start:start + self.batch_size]   # (bs,)
                bs = len(xb)

                margins = yb * (xb @ self.w + self.b)     # (bs,)
                mask    = (margins < 1.0).astype(np.float32)  # violated

                # Subgradient
                dw = self.w - (self.C / bs) * (mask * yb) @ xb
                db = -(self.C / bs) * (mask * yb).sum()

                self.w -= lr * dw
                self.b -= lr * db

            loss = self._hinge_loss(X, y_pm)
            self.losses.append(loss)
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{self.epochs}  loss={loss:.4f}  lr={lr:.5f}  {elapsed:.1f}s")

        print(f"{'─'*55}\n")
        return self

    # ── predict ──────────────────────────────
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        # map {-1,+1} back to {0,1}
        return ((scores >= 0).astype(np.int32))