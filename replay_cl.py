"""Continual learning runner with replay buffer for NSL-KDD."""
import random
from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from packaging.version import Version
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Columns (same as ARF pipeline)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty"
]

TASK_LABELS = {
    "Normal": "normal",
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm"],
}

LABEL_TO_ID = {name: idx for idx, name in enumerate(TASK_LABELS.keys())}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TaskSplit:
    name: str
    X: torch.Tensor
    y: torch.Tensor


class ReplayBuffer:
    """Reservoir sampling buffer storing past examples."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_seen = 0

    def add(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Add samples one-by-one using reservoir sampling."""
        assert X.shape[0] == y.shape[0]
        for xi, yi in zip(X, y):
            self.num_seen += 1
            entry = (xi.detach().cpu(), yi.detach().cpu())
            if len(self.buffer) < self.max_size:
                self.buffer.append(entry)
            else:
                idx = random.randrange(self.num_seen)
                if idx < self.max_size:
                    self.buffer[idx] = entry

    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.buffer) == 0 or k <= 0:
            return torch.empty(0).to(DEVICE), torch.empty(0).to(DEVICE)
        k = min(k, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), k)
        X_samples = torch.stack([self.buffer[i][0] for i in indices]).to(DEVICE)
        y_samples = torch.stack([self.buffer[i][1] for i in indices]).to(DEVICE)
        return X_samples, y_samples

    def __len__(self) -> int:
        return len(self.buffer)


class ContinualMLP(nn.Module):
    """Simple three-layer MLP for tabular NSL-KDD data."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_kdd(train_path: Path, test_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df_train = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    df_test = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)

    df_train = filter_and_label(df_train)
    df_test = filter_and_label(df_test)

    preprocessor = build_preprocessor(df_train)
    features_train = preprocessor.fit_transform(df_train.drop(columns=["label", "attack_type", "difficulty"]))
    features_test = preprocessor.transform(df_test.drop(columns=["label", "attack_type", "difficulty"]))

    y_train = df_train["label"].map(LABEL_TO_ID).astype(int).values
    y_test = df_test["label"].map(LABEL_TO_ID).astype(int).values

    return (
        torch.from_numpy(features_train.astype(np.float32)),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(features_test.astype(np.float32)),
        torch.from_numpy(y_test).long(),
    )


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack_type"] = df["attack_type"].astype(str).str.lower()
    label_series = []
    for attack in df["attack_type"]:
        label_series.append(map_attack_to_label(attack))
    df["label"] = label_series
    df["label"] = df["label"].astype(str)
    return df


def map_attack_to_label(attack: str) -> str:
    for label, values in TASK_LABELS.items():
        if isinstance(values, list):
            if attack in values:
                return label
        elif attack == values:
            return label
    return "Normal"


def make_onehot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if Version(SKLEARN_VERSION) >= Version("1.2"):
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [
        c for c in train_df.columns
        if c not in categorical_cols + ["label", "attack_type", "difficulty"]
    ]

    transformers = [
        ("cat", make_onehot_encoder(), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_tasks_by_attack(
    X: torch.Tensor,
    y: torch.Tensor,
) -> List[TaskSplit]:
    """
    Create tasks by progressively introducing new attack categories.
    This simulates IDS concept drift: each task sees additional attacks,
    so a static baseline forgets the earlier ones while replay should help.
    """
    TASK_SCENARIOS = [
        ["Normal"],
        ["DoS"],
        ["Probe"],
        ["R2L"],
        ["U2R"],
    ]

    tasks: List[TaskSplit] = []
    for idx, labels in enumerate(TASK_SCENARIOS):
        label_ids = torch.tensor([LABEL_TO_ID[l] for l in labels], dtype=y.dtype, device=y.device)
        mask = torch.isin(y, label_ids)
        X_subset = X[mask]
        y_subset = y[mask]
        perm = torch.randperm(X_subset.shape[0])
        X_subset = X_subset[perm]
        y_subset = y_subset[perm]
        task_name = f"Task_{idx + 1}"
        tasks.append(TaskSplit(name=task_name, X=X_subset, y=y_subset))

    return tasks


def train_on_task(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    task: TaskSplit,
    replay_buffer: Optional[ReplayBuffer] = None,
    epochs: int = 3,
    batch_size: int = 256,
    replay_ratio: float = 0.3,
) -> None:
    model.train()
    dataset = TensorDataset(task.X.to(DEVICE), task.y.to(DEVICE))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            replay_size = int(len(X_batch) * replay_ratio) if replay_buffer else 0
            if replay_buffer and replay_size > 0:
                X_replay, y_replay = replay_buffer.sample(replay_size)
                if X_replay.numel() > 0:
                    X_combined = torch.cat([X_batch, X_replay], dim=0)
                    y_combined = torch.cat([y_batch, y_replay], dim=0)
                else:
                    X_combined = X_batch
                    y_combined = y_batch
            else:
                X_combined = X_batch
                y_combined = y_batch

            logits = model(X_combined)
            loss = criterion(logits, y_combined)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if replay_buffer is not None:
        replay_buffer.add(task.X.to(DEVICE), task.y.to(DEVICE))


def evaluate_model(model: nn.Module, task: TaskSplit) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        X = task.X.to(DEVICE)
        y = task.y.to(DEVICE)
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
    return acc, f1


def compute_AA_FM_BWT(acc_matrix: List[List[float]]) -> Tuple[float, float, float]:
    matrix = np.array(acc_matrix, dtype=float)
    T = matrix.shape[0]
    if T == 0:
        return 0.0, 0.0, 0.0

    final_row = matrix[-1]
    AA = float(np.mean(final_row))

    FM_list = []
    for k in range(T):
        FM_list.append(float(np.max(matrix[:, k]) - final_row[k]))
    FM = float(np.mean(FM_list))

    if T > 1:
        BWT = float(np.mean([final_row[k] - matrix[k, k] for k in range(T - 1)]))
    else:
        BWT = 0.0

    return AA, FM, BWT


def run_continual_training(
    tasks: List[TaskSplit],
    use_replay: bool,
    replay_ratio: float = 0.3,
    eval_task: Optional[TaskSplit] = None,
) -> dict:
    input_dim = tasks[0].X.shape[1]
    num_classes = len(LABEL_TO_ID)
    model = ContinualMLP(input_dim=input_dim, hidden_dim=256, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(max_size=5000) if use_replay else None

    T = len(tasks)
    acc_matrix: List[List[float]] = [[0.0] * T for _ in range(T)]
    per_task_metrics: List[Tuple[str, float]] = []
    test_acc_over_time: List[float] = []

    for t, task in enumerate(tasks):
        print(f"\nTraining {task.name} (Task {t + 1}/{T}) - {'with replay' if use_replay else 'baseline'}")
        train_on_task(model, optimizer, criterion, task, buffer, replay_ratio=replay_ratio)
        for k, eval_task in enumerate(tasks):
            acc, _ = evaluate_model(model, eval_task)
            acc_matrix[t][k] = acc
        per_task_metrics.append((task.name, acc_matrix[t][t]))
        print(f"  -> {task.name} accuracy: {acc_matrix[t][t]:.4f}")

        if eval_task is not None:
            test_acc, _ = evaluate_model(model, eval_task)
            test_acc_over_time.append(test_acc)

    AA, FM, BWT = compute_AA_FM_BWT(acc_matrix)
    return {
        "acc_matrix": acc_matrix,
        "AA": AA,
        "FM": FM,
        "BWT": BWT,
        "per_task": per_task_metrics,
        "test_acc": test_acc_over_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continual learning CLI")
    parser.add_argument("--tasks", type=int, default=5, help="Number of drift tasks")
    parser.add_argument("--period-size", type=int, default=8000, help="Samples per period")
    parser.add_argument("--replay-ratio", type=float, default=0.3, help="Replay ratio per batch")
    parser.add_argument("--full-data", action="store_true", help="Use the entire train split per task")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path("data")
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("NSL-KDD data files missing in data/; run download_nsl_kdd.py first.")

    X_train, y_train, X_test, y_test = load_kdd(train_path, test_path)
    tasks = create_tasks_by_attack(X_train, y_train)
    full_test_task = TaskSplit("Test_Full", X_test, y_test)

    baseline_metrics = run_continual_training(
        tasks, use_replay=False, replay_ratio=args.replay_ratio, eval_task=full_test_task
    )
    replay_metrics = run_continual_training(
        tasks, use_replay=True, replay_ratio=args.replay_ratio, eval_task=full_test_task
    )

    print("\n=== Continual Learning Metrics Comparison ===")
    print("Metric       Baseline    Replay")
    for name in ("AA", "FM", "BWT"):
        baseline_val = baseline_metrics[name]
        replay_val = replay_metrics[name]
        print(f"{name:<12}{baseline_val:>8.4f}{replay_val:>12.4f}")

    print("\nTest accuracy over tasks (Replay Buffer):")
    for idx, acc in enumerate(replay_metrics["test_acc"], 1):
        print(f"  After Task {idx}: {acc:.4f}")

    print("\nPer-task accuracy after training (Replay Buffer):")
    for idx, (task_name, acc) in enumerate(replay_metrics["per_task"], 1):
        print(f"  Task {idx} ({task_name}): {acc:.4f}")


if __name__ == "__main__":
    main()
