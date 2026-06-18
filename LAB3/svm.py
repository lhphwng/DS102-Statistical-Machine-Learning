import numpy as np

class SVM:
    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=100,
                 batch_size=64, lr_decay=1e-4, random_seed=42):
        self.C             = C
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.lr_decay      = lr_decay
        self.random_seed   = random_seed

        self.w            = None
        self.b            = 0.0
        self.loss_history = []

    # ─────────────────────────────────────────
    # Ham noi bo
    # ─────────────────────────────────────────

    def _to_svm_labels(self, y):
        """Chuyen nhan {0,1} -> {-1, +1}."""
        return np.where(y == 0, -1.0, 1.0)

    def _hinge_loss(self, X, y_svm):
        """
        Tinh toan ham mat mat:
            L = (lam/2)||w||^2 + (1/N) * sum(max(0, 1 - yi*(w'xi+b)))
        """
        N   = len(X)
        lam = 1.0 / (self.C * N)
        margins  = y_svm * (X @ self.w + self.b)
        hinge    = np.maximum(0.0, 1.0 - margins).mean()
        reg      = 0.5 * lam * np.dot(self.w, self.w)
        return reg + hinge

    def _gradient(self, X_batch, y_batch):
        """
        Tinh gradient tren mot mini-batch.

        Tra ve:
            dw : gradient theo w, shape (D,)
            db : gradient theo b, scalar
        """
        N   = len(X_batch)
        lam = 1.0 / (self.C * N)

        margins = y_batch * (X_batch @ self.w + self.b)   # (N,)
        mask    = margins < 1.0                            # cac mau vi pham margin

        # Gradient hinge loss
        if mask.any():
            dw_hinge = -(y_batch[mask, None] * X_batch[mask]).mean(axis=0)
            db_hinge = -y_batch[mask].mean()
        else:
            dw_hinge = np.zeros_like(self.w)
            db_hinge = 0.0

        dw = lam * self.w + dw_hinge
        db = db_hinge
        return dw, db

    # ─────────────────────────────────────────
    # Giao dien chinh
    # ─────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Huan luyen SVM bang Mini-batch SGD.

        Tham so:
            X, y         : tap train
            X_val, y_val : tap validation (tuy chon, de theo doi loss)
            verbose      : in thong tin moi 10 epoch
        """
        rng  = np.random.default_rng(self.random_seed)
        N, D = X.shape
        lam  = 1.0 / (self.C * N)

        y_svm = self._to_svm_labels(y)

        # Khoi tao trong so nho gan 0
        self.w = rng.normal(0, 0.01, D).astype(np.float64)
        self.b = 0.0
        self.loss_history = []

        if verbose:
            print("  Bat dau huan luyen: N=" + str(N) + " D=" + str(D)
                  + " C=" + str(self.C) + " lr=" + str(self.learning_rate)
                  + " epochs=" + str(self.n_epochs))

        for epoch in range(1, self.n_epochs + 1):
            # Learning rate giam dan: lr_t = lr0 / (1 + decay * t)
            lr_t = self.learning_rate / (1.0 + self.lr_decay * epoch)

            # Xao tron du lieu moi epoch
            perm  = rng.permutation(N)
            X_shf = X[perm]
            y_shf = y_svm[perm]

            # Mini-batch SGD
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                dw, db = self._gradient(X_shf[start:end], y_shf[start:end])
                self.w -= lr_t * dw
                self.b -= lr_t * db

            # Ghi loss sau moi epoch
            loss = self._hinge_loss(X, y_svm)
            self.loss_history.append(loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                msg = "  Epoch " + str(epoch).rjust(4) + "/" + str(self.n_epochs) \
                      + "  loss=" + str(round(loss, 4))
                if X_val is not None and y_val is not None:
                    y_val_svm = self._to_svm_labels(y_val)
                    val_loss  = self._hinge_loss(X_val, y_val_svm)
                    val_acc   = np.mean(self.predict(X_val) == y_val)
                    msg += "  val_loss=" + str(round(val_loss, 4)) \
                         + "  val_acc=" + str(round(val_acc, 4))
                print(msg)

        if verbose:
            print("  Huan luyen hoan tat.")
        return self

    def predict(self, X):
        """
        Du doan nhan cho X.
        score = w'x + b >= 0  ->  1 (PNEUMONIA)
        score < 0             ->  0 (NORMAL)
        """
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, 0).astype(np.int32)
