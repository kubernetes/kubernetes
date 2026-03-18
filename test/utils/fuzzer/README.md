# Exemplary Pod Fuzzer

## Overview
The **Exemplary Pod Fuzzer** is a specialized utility designed to generate high-fidelity synthetic Pod objects that mirror the data complexity and structural density of large-scale infrastructure agents. Its primary purpose is to provide a "ground truth" for stress-testing the Kubernetes control plane and validating critical memory optimizations, such as **string interning** and **field stripping**.

By replicating the specific metadata nesting and spec redundancy patterns found in massive clusters, this tool allows developers to benchmark the Resident Set Size (RSS) of core components (API server, Scheduler, Controller Manager) under realistic pressure without requiring a physical cluster of thousands of nodes.

## Key Features

### 1. High-Fidelity Generative Complexity
Replicates the primary sources of memory exhaustion in large clusters:
- **Nested `ManagedFields`**: Generates deep, hierarchical `FieldsV1` JSON structures (up to 10+ levels) with thousands of unique path markers (`f:`) to simulate complex Server-Side Apply history.
- **Spec Density**: Automatically injects 100+ unique environment variables and dozens of volumes into a base template to simulate "heavy" container specs.
- **Configurable Bloat**: Allows targeting specific byte counts for `ManagedFields` and `Annotations` to mimic the "metadata tax" of various controllers.

### 2. Interning & Deduplication Benchmarking
Specifically designed to validate string interning (e.g., via Go 1.23's `unique` package):
- **Identical Generative Data**: Generates complex field paths and environment variables once per template and reuses them across all pods. This ensures the API server can effectively deduplicate millions of strings during high-concurrency benchmarks.
- **Shared PodSpecs**: Batches of pods share an identical underlying `PodSpec` structure, ensuring that only the unique metadata (Name, UID) contributes to unique memory growth.

### 3. Safety by Default
- **Impossible Scheduling**: Templates use non-existent `nodeSelectors` and `schedulerNames` by default to ensure pods remain in a `Pending` state, focusing pressure purely on the control plane.
- **Sanitized Identifiers**: All generated keys use generic prefixes (e.g., `FUZZ_GEN_VARIABLE_001`) to ensure compliance and avoid proprietary data leakage.

### 4. High-Performance Injection
- **Concurrent Workers**: Uses an `errgroup`-based concurrent creator to inject or write thousands of pods per second.
- **Real-Time Progress**: CLI provides a live progress bar and performance metrics (pods/sec).

## The "Shape" of a Fuzzed Pod

A representative fuzzed pod from this tool (e.g., `complex-daemonset.yaml`) is designed to match the following structural profile:

- **Manifest Size**: ~100 KB to 500 KB (Configurable).
- **Metadata Density**:
    - **Field Paths (`f:`)**: ~1,000 to 20,000 unique JSON paths nested up to 7 levels deep.
    - **Manager History**: A deep history of `ManagedFields` entries representing multiple controllers.
    - **Annotations**: Large opaque blobs (e.g., 24KB) mimicking cluster-level agent data.
- **Spec Complexity**:
    - **Containers**: 5-7 containers (mixed Init and Runtime).
    - **Environment Variables**: 30 to 150 unique variables per container.
    - **Volumes/Mounts**: 15 to 30 unique volumes and mount points.

## Memory Benchmarking

The fuzzer is used to measure the RSS impact of high-density metadata on the Kubernetes control plane.

### **Benchmarking Procedure**

1.  **Create a Control-Plane Only Cluster**:
    ```bash
    kind create cluster --name fuzzer-benchmark
    ```

2.  **Prepare the Environment**:
    The `complex-daemonset.yaml` template requires a specific service account to be present in the namespace.
    ```bash
    kubectl create serviceaccount network-agent
    ```

3.  **Run the Benchmark**:
    Inject 50,000 representative pods (tuned to match production "infrastructure agent" profiles).
    ```bash
    go run test/utils/fuzzer/cmd/main.go \
      --count=50000 \
      --template=test/utils/fuzzer/templates/complex-daemonset.yaml \
      --concurrency=100 \
      --kubeconfig=$HOME/.kube/config
    ```

4.  **Verify Pod Count**:
    The most efficient way to verify the injection at scale is to query the API server's internal storage metrics:
    ```bash
    kubectl get --raw /metrics | grep 'apiserver_storage_objects{resource="pods"}'
    ```

5.  **Measure Memory**:
    ```bash
    docker exec fuzzer-benchmark-control-plane ps aux | grep -E "kube-apiserver|etcd"
    ```

### **Analysis: Latest Results (50,000 Pods on Kubernetes v1.35.0)**

The latest benchmark utilized the "heavy" configuration of the `complex-daemonset` template (~182 KB per manifest, ~5,200 nested field paths) running on **Kubernetes v1.35.0**.

| Process | Stable RSS | Peak RSS | Time to Inject |
| :--- | :--- | :--- | :--- |
| **kube-apiserver** | **13.64 GB** | 14.86 GB | **~98s** |
| **etcd** | **0.90 GB** | 11.29 GB | -- |


## Configuration Guide

Templates allow you to define the generative complexity of the synthetic data.

### Example Template Fields (`complex-daemonset.yaml`)

```yaml
name: "representative-daemonset"
# Generative Spec Complexity
envVarCount: 30       # Automatically injects 30 unique env vars into the first container
# Managed Fields Array
managedFieldCount: 2  # Cycles through the managers below to create 2 entries
managedFields:
- manager: "system-controller-manager"
  operation: "Update"
  fieldPathCount: 1100 # Generates 1,100 unique "f:" keys
  fieldPathDepth: 6    # Nests the keys up to 6 levels deep (Trie-like)
  length: 5000         # Adds an additional 5KB random string to this entry
# Static Annotations
annotations:
- key: "fuzzer.io/metadata-bloat"
  length: 24000        # Adds a 24KB random string annotation
baseSpec:
  # Standard PodSpec template
  containers: [...]
  volumes: [...]
```

### How to Update
1.  **Modify a Template**: Edit an existing YAML file in `test/utils/fuzzer/templates/` or create a new one.
2.  **Adjust Cardinality**: Increase `fieldPathCount` or `envVarCount` to test the limits of string table indexing in the API server.
3.  **Verify Structure**: Generate a single pod for inspection before running a scale test:
    ```bash
    go run test/utils/fuzzer/cmd/main.go --count=1 --template=my-template.yaml --out-dir=./test
    ```

## Project Structure
- `fuzzer.go`: Core library containing the generative engine and concurrent creator.
- `fuzzer_test.go`: Unit tests verifying spec stability, nesting logic, and interning support.
- `cmd/`: CLI implementation for cluster injection and manifest export.
- `templates/`: Pre-defined YAML profiles for scale-testing scenarios.
