#!/usr/bin/env python3
# Copyright The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Wide-Deep model training benchmark.

Equivalent to the TensorFlow wide_deep benchmark, training on the
Census Income (Adult) dataset from UCI ML Repository.
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Use OpenMP for CPU parallelism - use available CPUs or default to 4
NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 4))
torch.set_num_threads(NUM_THREADS)

# Census Income dataset paths (pre-downloaded in Docker image)
TRAIN_DATA_PATH = "/data/adult.data"
TEST_DATA_PATH = "/data/adult.test"

# Column names for the Census dataset
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# Categorical columns
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

# Numerical columns
NUMERICAL_COLUMNS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

# Pre-compute column indices for efficiency
NUM_COL_INDICES = {col: COLUMNS.index(col) for col in NUMERICAL_COLUMNS}
CAT_COL_INDICES = {col: COLUMNS.index(col) for col in CATEGORICAL_COLUMNS}


class WideDeepModel(nn.Module):
    """Wide and Deep model combining memorization and generalization."""

    def __init__(self, num_features, embed_dims, hidden_dims, num_classes=2):
        super().__init__()

        # Wide component (linear model for memorization)
        self.wide = nn.Linear(num_features, num_classes)

        # Calculate deep input dimension
        total_embed_dim = sum(dim for _, dim in embed_dims.values())
        deep_input_dim = num_features + total_embed_dim

        # Deep component (DNN for generalization)
        layers = []
        in_dim = deep_input_dim
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(0.2)
            ])
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(in_dim, num_classes)

        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cat, dim)
            for name, (num_cat, dim) in embed_dims.items()
        })

    def forward(self, num_x, cat_x):
        # Wide path
        wide_out = self.wide(num_x)

        # Deep path with embeddings
        embed_list = []
        for name, embed_layer in self.embeddings.items():
            embed_list.append(embed_layer(cat_x[name]))

        deep_input = torch.cat([num_x] + embed_list, dim=1)
        deep_out = self.deep_out(self.deep(deep_input))

        # Combine wide and deep
        return wide_out + deep_out


def load_census_data(data_path):
    """Load and parse Census Income data."""
    data = []
    try:
        with open(data_path, 'r') as f:
            for line in f:
                # Skip empty lines and test file header
                line = line.strip()
                if not line or line.startswith('|'):
                    continue
                # Parse CSV
                fields = [field.strip() for field in line.split(',')]
                if len(fields) == len(COLUMNS):
                    data.append(fields)
    except FileNotFoundError as e:
        raise RuntimeError(f"Data file not found: {data_path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to read data file {data_path}: {e}") from e

    if not data:
        raise RuntimeError(f"No valid data loaded from {data_path}")

    return data


def process_data(train_data, test_data):
    """Process raw data into numerical and categorical features."""
    # Build vocabulary for categorical features
    cat_vocabs = {col: {} for col in CATEGORICAL_COLUMNS}

    for row in train_data + test_data:
        for col in CATEGORICAL_COLUMNS:
            idx = CAT_COL_INDICES[col]
            val = row[idx]
            if val not in cat_vocabs[col]:
                cat_vocabs[col][val] = len(cat_vocabs[col])

    def extract_features(data):
        num_features = []
        cat_features = {col: [] for col in CATEGORICAL_COLUMNS}
        labels = []

        for row in data:
            # Numerical features (using pre-computed indices)
            num_row = []
            for col in NUMERICAL_COLUMNS:
                idx = NUM_COL_INDICES[col]
                try:
                    num_row.append(float(row[idx]))
                except ValueError:
                    num_row.append(0.0)
            num_features.append(num_row)

            # Categorical features (using pre-computed indices)
            for col in CATEGORICAL_COLUMNS:
                idx = CAT_COL_INDICES[col]
                val = row[idx]
                cat_features[col].append(cat_vocabs[col].get(val, 0))

            # Label (income >50K)
            label_str = row[-1].strip().rstrip('.')
            labels.append(1 if '>50K' in label_str else 0)

        # Convert to numpy arrays
        num_features = np.array(num_features, dtype=np.float32)
        cat_features = {col: np.array(vals, dtype=np.int64) for col, vals in cat_features.items()}
        labels = np.array(labels, dtype=np.int64)

        return num_features, cat_features, labels

    # Get cardinalities for embeddings
    cat_cardinalities = {col: len(vocab) + 1 for col, vocab in cat_vocabs.items()}  # +1 for unknown

    train_num, train_cat, train_labels = extract_features(train_data)
    test_num, test_cat, test_labels = extract_features(test_data)

    # Normalize numerical features using training data statistics only
    train_mean = train_num.mean(axis=0)
    train_std = train_num.std(axis=0) + 1e-8
    train_num = (train_num - train_mean) / train_std
    test_num = (test_num - train_mean) / train_std  # Use training stats for test data

    return (train_num, train_cat, train_labels,
            test_num, test_cat, test_labels,
            cat_cardinalities)


def train():
    """Main training loop."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of threads: {torch.get_num_threads()}")

    # Load Census Income data (pre-downloaded in Docker image)
    print("Loading Census Income dataset...")
    train_data = load_census_data(TRAIN_DATA_PATH)
    test_data = load_census_data(TEST_DATA_PATH)
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Process data
    (train_num, train_cat, train_labels,
     test_num, test_cat, test_labels,
     cat_cardinalities) = process_data(train_data, test_data)

    # Convert to tensors (CPU only, no need for .to(device))
    train_num_t = torch.FloatTensor(train_num)
    train_cat_t = {name: torch.LongTensor(vals) for name, vals in train_cat.items()}
    train_labels_t = torch.LongTensor(train_labels)

    test_num_t = torch.FloatTensor(test_num)
    test_cat_t = {name: torch.LongTensor(vals) for name, vals in test_cat.items()}
    test_labels_t = torch.LongTensor(test_labels)

    # Model configuration
    # Embedding dim = min(50, cardinality // 2 + 1)
    embed_dims = {
        name: (card, min(50, card // 2 + 1))
        for name, card in cat_cardinalities.items()
    }

    model = WideDeepModel(
        num_features=len(NUMERICAL_COLUMNS),
        embed_dims=embed_dims,
        hidden_dims=[1024, 512, 256],
        num_classes=2
    )

    # Use torch.compile() for optimized execution (PyTorch 2.0+)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    # Note: TF benchmark uses batch_size=40 with 40 epochs.
    # We use full-batch training with more epochs for comparable total compute.
    n_epochs = 300
    batch_size = len(train_labels)  # Full batch for simpler CPU benchmarking

    print(f"Training for {n_epochs} epochs, batch_size={batch_size}")
    print("-" * 50)

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(train_num_t, train_cat_t)
        loss = criterion(outputs, train_labels_t)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {loss.item():.4f}")

    elapsed = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(test_num_t, test_cat_t)
        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == test_labels_t).float().mean().item()

    print("-" * 50)
    print(f"Training completed in {elapsed:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train()
