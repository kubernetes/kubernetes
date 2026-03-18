# Exemplary Pod Fuzzer

## Overview
The **Exemplary Pod Fuzzer** is a specialized utility designed to generate high-fidelity synthetic Pod objects that mirror the data complexity and density of large-scale production environments. Its primary purpose is to provide a "ground truth" for stress-testing the Kubernetes control plane and validating critical memory optimizations, such as **string interning** and **field stripping**.

By replicating the specific metadata bloat and spec redundancy patterns found in massive clusters, this tool allows developers to benchmark the Resident Set Size (RSS) of core components (API server, Scheduler, Controller Manager) under realistic pressure without requiring a physical cluster of thousands of nodes.

## Key Features

### 1. High-Fidelity Metadata Bloat
Replicates the primary sources of memory exhaustion in large clusters:
- **`ManagedFields` Emulation**: Populates `FieldsV1` with schema-based JSON structures to simulate complex Server-Side Apply history.
- **Configurable Bloat**: Allows targeting specific byte counts for `ManagedFields` and `Annotations` to mimic the "metadata tax" of various controllers.

### 2. Interning & Deduplication Benchmarking
Specifically designed to validate string interning (e.g., via Go 1.23's `unique` package):
- **Identical Bloat Strings**: Generates large metadata strings once per template and reuses them across all pods. This ensures the API server has a consistent baseline for deduplication testing.
- **Shared PodSpecs**: Batches of pods can share a single `PodSpec` pointer in memory, simulating massive DaemonSets or StatefulSets.

### 3. Safety by Default
Ensures that fuzzing the control plane does not accidentally overwhelm worker node resources:
- **Impossible Scheduling**: Templates use non-existent `nodeSelectors` and `schedulerNames` by default.
- **Pending State**: Pods remain in a `Pending` state indefinitely, occupying memory in `etcd` and the API server without ever starting a container.

### 4. High-Performance Injection
- **Concurrent Workers**: Uses goroutines to generate and inject/write thousands of pods in seconds.
- **Flexible Export**: Supports direct cluster injection or exporting manifests to a local directory (YAML).
## Memory Benchmarking

The fuzzer is designed to measure the Resident Set Size (RSS) impact of large-scale metadata on the Kubernetes control plane.

### **Benchmarking Procedure**

1.  **Create a Control-Plane Only Cluster**:
    Use a configuration that omits worker nodes to ensure pods remain in a `Pending` state, focusing pressure purely on the API server and `etcd`.
    ```yaml
    # benchmark-config.yaml
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes: [- role: control-plane]
    ```
    ```bash
    kind create cluster --config benchmark-config.yaml --name fuzzer-benchmark
    ```

2.  **Prepare the Environment**:
    The `complex-daemonset.yaml` template requires a specific service account.
    ```bash
    kubectl create serviceaccount network-agent
    ```

3.  **Run the Benchmark**:
    Inject 50,000 pods directly into the API server using high concurrency and QPS.
    ```bash
    go run test/utils/fuzzer/cmd/main.go \
      --count=50000 \
      --template=test/utils/fuzzer/templates/complex-daemonset.yaml \
      --concurrency=100 \
      --kubeconfig=$HOME/.kube/config
    ```

4.  **Measure Memory**:
    Monitor the RSS of core processes inside the container:
    ```bash
    docker exec fuzzer-benchmark-control-plane ps aux | grep -E "kube-apiserver|etcd"
    ```

### **Analysis: Why is memory lower than anticipated?**

In recent benchmarks, 50,000 complex pods (each with ~40KB of metadata) resulted in only **~8.8 GB RSS** for the `kube-apiserver`, significantly lower than the 18-25 GB projected in earlier research.

**1. Effective String Interning**
The fuzzer pre-computes "bloat" strings (ManagedFields and Annotations) once per template. When these 50,000 objects are serialized/deserialized by the API server in Kubernetes 1.35+, the system identifies the identical strings. By storing the actual string content only once and using pointers (interning) for all subsequent occurrences, the memory "tax" per pod is reduced from tens of kilobytes to a few bytes.

**2. Optimized Protobuf Decoding**
Modern Kubernetes versions use optimized protobuf paths that aggressively deduplicate repetitive metadata fields (like Manager names and common JSON paths in `FieldsV1`), which previously accounted for over 50% of memory bloat.

**3. Direct Injection Efficiency**
By bypassing the disk (no YAML files) and using direct API injection, we reduce the temporary memory overhead associated with parsing large files and multiple `kubectl` process invocations.

## Project Structure
...

Templates allow you to define the "shape" of the synthetic data. Multiple profiles are included in the `templates/` directory:
- `representative-pod.yaml`: A basic pod with typical metadata bloat.
- `complex-daemonset.yaml`: A high-fidelity profile mimicking a complex infrastructure agent with multiple init/sidecar containers, 20+ volume mounts, and multi-manager `ManagedFields` history.

### Example Template Structure
```yaml
name: "complex-daemonset-pod"
baseSpec:
  initContainers: [...]
  containers: [...]
  volumes: [...]
managedFields:
- manager: "agent-controller-manager"
  operation: "Update"
  length: 8000 # Bloats the JSON to 8KB
annotations:
- key: "fuzzer.io/arch-complexity-blob"
  length: 15000 # Adds a 15KB annotation
```

## Project Structure
- `fuzzer.go`: Core library containing the fuzzer engine and creator utility.
- `fuzzer_test.go`: Unit tests verifying metadata stability and interning support.
- `cmd/`: CLI implementation for standalone manifest generation.
- `templates/`: Pre-defined YAML profiles for common scale-testing scenarios.
