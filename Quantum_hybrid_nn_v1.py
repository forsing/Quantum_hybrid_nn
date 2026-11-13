# Quantum_hybrid_nn 


import random
import numpy as np

random.seed(39)
np.random.seed(39)

from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_BALLS = 39
DRAW_SIZE = 7
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def load_draws(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            nums = [int(x) for x in parts[:DRAW_SIZE]]
            if len(nums) != DRAW_SIZE:
                raise ValueError(f"Neispravan red u {csv_path}: {line}")
            if any(n < 1 or n > NUM_BALLS for n in nums) or len(set(nums)) != DRAW_SIZE:
                raise ValueError(f"Nelegalna kombinacija u {csv_path}: {nums}")
            rows.append(sorted(nums))
    if len(rows) < 100:
        raise ValueError(f"Premalo kola u {csv_path}: {len(rows)}")
    return np.array(rows, dtype=np.int32)


def draws_to_binary_matrix(draws: np.ndarray) -> np.ndarray:
    t_len = len(draws)
    mat = np.zeros((t_len, NUM_BALLS), dtype=np.float64)
    rows_idx = np.arange(t_len)[:, None]
    mat[rows_idx, draws - 1] = 1.0
    return mat


def build_quantum_amplitude_features(
    binary_mat: np.ndarray, window_size: int = 16
) -> np.ndarray:
    t_len, n_balls = binary_mat.shape
    q_features = np.zeros((t_len, N_STATES), dtype=np.float64)

    rng = np.random.RandomState(39)
    entangle_angles = rng.uniform(-0.45, 0.45, size=(2, N_QUBITS, 2))

    weights = np.exp(-np.linspace(1.5, 0.0, window_size))
    weights /= np.sum(weights)

    active_share = float(n_balls) / float(N_STATES)
    tail_len = N_STATES - n_balls

    for t in range(t_len):
        start_idx = max(0, t + 1 - window_size)
        sub = binary_mat[start_idx : t + 1]
        w_sub = weights[window_size - len(sub) :]
        w_sub = w_sub / np.sum(w_sub)

        profile_39 = (w_sub @ sub) / float(DRAW_SIZE)
        profile_39 = np.maximum(profile_39, 1e-12)
        profile_39 /= np.sum(profile_39)

        amp_64 = np.zeros(N_STATES, dtype=np.float64)
        amp_64[:n_balls] = np.sqrt(active_share * profile_39)
        if tail_len > 0:
            amp_64[n_balls:] = np.sqrt((1.0 - active_share) / float(tail_len))

        amp_64 /= float(np.linalg.norm(amp_64))

        qc = QuantumCircuit(N_QUBITS)
        qc.initialize(amp_64.tolist(), list(range(N_QUBITS)))

        for layer in range(2):
            for q in range(N_QUBITS):
                qc.ry(float(entangle_angles[layer, q, 0]), q)
                qc.rz(float(entangle_angles[layer, q, 1]), q)
            for q in range(N_QUBITS - 1):
                qc.cx(q, q + 1)
            qc.cx(N_QUBITS - 1, 0)

        sv = Statevector.from_instruction(qc)
        q_features[t] = np.abs(sv.data) ** 2

    return q_features


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)


class LinearLayer:
    def __init__(self, n_in: int, n_out: int, rng: np.random.RandomState):
        scale = np.sqrt(2.0 / float(n_in))
        self.weights = rng.normal(0.0, scale, size=(n_out, n_in))
        self.bias = np.zeros((n_out, 1), dtype=np.float64)
        self._last_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_x = x
        return self.weights @ x + self.bias

    def backward(self, grad_out: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
        grad_w = grad_out @ self._last_x.T + weight_decay * self.weights
        grad_b = np.sum(grad_out, axis=1, keepdims=True)
        grad_in = self.weights.T @ grad_out
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
        return grad_in


class ReLULayer:
    def __init__(self):
        self._last_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_x = x
        return np.maximum(0.0, x)

    def backward(self, grad_out: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
        return grad_out * (self._last_x > 0.0).astype(np.float64)


class QuantumHybridDistributionNet:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 39):
        rng = np.random.RandomState(seed)
        self.layers = [
            LinearLayer(in_dim, hidden_dim, rng),
            ReLULayer(),
            LinearLayer(hidden_dim, out_dim, rng),
        ]

    def forward_logits(self, x_col: np.ndarray) -> np.ndarray:
        out = x_col
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_probs(self, x_rows: np.ndarray) -> np.ndarray:
        logits = self.forward_logits(x_rows.T)
        probs = softmax_batch(logits)
        return probs.T

    def train_chronological(
        self,
        x_rows: np.ndarray,
        y_binary: np.ndarray,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 0.03,
        weight_decay: float = 1e-3,
    ) -> None:
        n_samples = len(x_rows)
        target_dist = y_binary / float(DRAW_SIZE)

        for _ in range(epochs):
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                bx = x_rows[start:end].T
                bt = target_dist[start:end].T
                bsz = float(end - start)

                logits = self.forward_logits(bx)
                probs = softmax_batch(logits)
                grad_logits = (probs - bt) / bsz

                grad = grad_logits
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr=lr, weight_decay=weight_decay)


def evaluate_model(
    model: QuantumHybridDistributionNet, x_rows: np.ndarray, y_binary: np.ndarray
) -> tuple[float, float]:
    probs = model.predict_probs(x_rows)
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs, axis=1, keepdims=True)

    target_dist = y_binary / float(DRAW_SIZE)
    nll = -float(np.mean(np.sum(target_dist * np.log(probs), axis=1)))

    hits = 0.0
    n_samples = len(x_rows)
    for i in range(n_samples):
        top7 = np.argsort(probs[i])[-DRAW_SIZE:]
        actual = np.flatnonzero(y_binary[i] > 0.5)
        hits += float(len(set(top7.tolist()).intersection(actual.tolist())))

    return nll, hits / float(n_samples)


def select_next_combination(dist_39: np.ndarray) -> list[int]:
    if dist_39.shape != (NUM_BALLS,) or not np.all(np.isfinite(dist_39)):
        raise ValueError("Neispravna izlazna raspodela modela.")
    if abs(float(np.sum(dist_39)) - 1.0) > 1e-5:
        raise ValueError("Raspodela nije normalizovana.")
    top7_idx = np.argsort(dist_39)[-DRAW_SIZE:]
    combo = sorted(int(i + 1) for i in top7_idx)
    if len(combo) != DRAW_SIZE or len(set(combo)) != DRAW_SIZE:
        raise ValueError("Nelegalna izlazna kombinacija.")
    return combo


def run_pipeline_for_csv(csv_path: Path) -> None:
    draws = load_draws(csv_path)
    binary_mat = draws_to_binary_matrix(draws)
    t_len = len(binary_mat)

    n_trans = t_len - 1
    train_end = int(n_trans * 0.70)
    val_end = int(n_trans * 0.85)

    candidate_configs = (
        (12, 64),
        (24, 96),
    )

    best_cfg = next(iter(candidate_configs))
    best_val_nll = float("inf")
    best_val_hits = 0.0
    best_test_nll = float("inf")
    best_test_hits = 0.0

    for win_size, hidden_dim in candidate_configs:
        q_feats = build_quantum_amplitude_features(binary_mat, window_size=win_size)
        x_all = np.hstack([q_feats, binary_mat])
        _, in_dim = x_all.shape

        x_train = x_all[:train_end]
        y_train = binary_mat[1 : train_end + 1]

        x_val = x_all[train_end:val_end]
        y_val = binary_mat[train_end + 1 : val_end + 1]

        model_cv = QuantumHybridDistributionNet(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=NUM_BALLS, seed=39
        )
        model_cv.train_chronological(x_train, y_train, epochs=35, batch_size=64, lr=0.03)
        val_nll, val_hits = evaluate_model(model_cv, x_val, y_val)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_hits = val_hits
            best_cfg = (win_size, hidden_dim)

            x_test = x_all[val_end:n_trans]
            y_test = binary_mat[val_end + 1 : n_trans + 1]
            best_test_nll, best_test_hits = evaluate_model(model_cv, x_test, y_test)

    best_win, best_hidden = best_cfg
    q_feats_final = build_quantum_amplitude_features(binary_mat, window_size=best_win)
    x_final = np.hstack([q_feats_final, binary_mat])
    _, final_in_dim = x_final.shape

    x_full = x_final[:n_trans]
    y_full = binary_mat[1 : n_trans + 1]

    final_model = QuantumHybridDistributionNet(
        in_dim=final_in_dim, hidden_dim=best_hidden, out_dim=NUM_BALLS, seed=39
    )
    final_model.train_chronological(x_full, y_full, epochs=45, batch_size=64, lr=0.03)

    next_probs_mat = final_model.predict_probs(x_final[-1:])
    next_probs = next_probs_mat.ravel()
    next_probs = np.maximum(next_probs, 1e-12)
    next_probs /= np.sum(next_probs)
    next_combo = select_next_combination(next_probs)

    print(f"CSV: {csv_path.name} (kola: {t_len})")
    print(f"Validacija -> NLL: {best_val_nll:.4f} | Prosek pogodaka: {best_val_hits:.3f}")
    print(f"Test       -> NLL: {best_test_nll:.4f} | Prosek pogodaka: {best_test_hits:.3f}")
    print(f"NEXT       -> {next_combo}")
    print("-" * 52)


def main() -> None:
    print("Metoda: Quantum-Hybrid NN od nule (6 qubita)")
    print("Posebnost: Kvantne amplitude + analiticki backprop")
    print("=" * 52)
    for csv_path in FILES:
        run_pipeline_for_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum-Hybrid NN od nule (6 qubita)
Posebnost: Kvantne amplitude + analiticki backprop
====================================================

CSV: loto7_4718_k90.csv (kola: 4718)
Validacija -> NLL: 3.6774 | Prosek pogodaka: 1.242
Test       -> NLL: 3.6781 | Prosek pogodaka: 1.290
NEXT       -> [4, x, 9, y, 23, z, 28]

----------------------------------------------------

CSV: loto7_4718_k90_loto_2981.csv (kola: 2981)
Validacija -> NLL: 3.6798 | Prosek pogodaka: 1.268
Test       -> NLL: 3.6804 | Prosek pogodaka: 1.338
NEXT       -> [5, x, 21, y, 26, z, 37]
----------------------------------------------------

CSV: loto7_4718_k90_loto_plus_1737.csv (kola: 1737)
Validacija -> NLL: 3.6809 | Prosek pogodaka: 1.308
Test       -> NLL: 3.6867 | Prosek pogodaka: 1.253
NEXT       -> [4, x, 11, y, 29, z, 39]
----------------------------------------------------
"""



"""
1. Vektorizovani Batched Softmax sa numeričkom stabilizacijom (za regresiju raspodele):
	• Izlazni sloj od 39 čvorova direktno generiše diskretnu raspodelu verovatnoće nad brojevima 1-39 putem stabilizovanog Softmax-a (shifted = logits - max(logits)), gde je zbir svih 39 verovatnoća tačno 1.0.
2. Višeciljna Cross-Entropy funkcija gubitka sa mekim ciljevima (1/7):
	• Ciljni vektor svakog kola ima tačno 7 izvučenih brojeva sa masom 1/7 (zbir = 1.0). Vektorizovani gradijent \(\nabla_{\mathbf{Z}^L} \text{Loss} = \frac{1}{N}(\mathbf{Y}^L - \mathbf{T})\) matematički identičan i važi bez ikakve promene za višeciljnu raspodelu 1/7.
3. Analitički Backpropagation bez spoljnih crnih kutija + Kaiming inicijalizacija:
	• Čista matrična propagacija unazad (\(\nabla_{\mathbf{W}^i} = \nabla_{\mathbf{Z}^i}(\mathbf{Y}^{i-1})^T\)) i Kaiming inicijalizacija težina (\(\mathcal{N}(0, \sqrt{2 / n_{\text{in}}})\)) sa fiksnim semenom 39 sprečavaju nestajanje ili eksploziju gradijenta kroz skrivene slojeve sa ReLU aktivacijom.
4. Kvantni hibridni tok:
	• Kvantno-klasični hibrid ili kvantno-inspirisani parametrizovani sloj: skrivena reprezentacija iz mreže od nule (ili kvantni sloj amplituda na 6 qubita) mapira hronološki prozor prethodnih kola u logite/amplitude za 39 brojeva, trenirane hronološki bez mešanja redova


• Učitavanje i priprema podataka: Učitavaju se istorijski podaci iz tri CSV fajla (loto7_4716_k89.csv, loto7_4716_k89_loto_2980.csv, loto7_4716_k89_loto_plus_1736.csv) i konvertuju u binarne matrice i hronološke prozore bez mešanja redova.
• Kvantni sloj (6-qubit Statevector): Funkcija quantum_amplitude_kernel formira kvantno kolo sa 6 kubita (Qiskit), primenjuje rotacije uglova i ispliće stanja preko CNOT kapija, vraćajući vektor od 64 kvadrata amplituda.
• Klasični deo mreže: Sadrži linearne slojeve (LinearLayer sa Kaiming inicijalizacijom), ReLU aktivaciju (ReLULayer) i numerički stabilizovan softmax_batch za generisanje raspodele verovatnoće preko 39 brojeva.
• Trening i evaluacija: Klasa QuantumHybridDistributionNet spaja klasične slojeve i kvantne karakteristike, vršeći analitički backpropagation za unakrsnu entropiju (Cross-Entropy).
• Izvršavanje: Funkcija run_pipeline_for_csv sprovodi validaciju, testiranje i predviđa sledeću kombinaciju (select_next_combination) za svaki od unetih skupova podataka.
"""
