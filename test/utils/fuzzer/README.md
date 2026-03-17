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

## Usage

### CLI Tool
The fuzzer includes a command-line utility to generate manifests for manual application or inspection.

```bash
go run test/utils/fuzzer/cmd/main.go --count=1000 --template=test/utils/fuzzer/templates/representative-pod.yaml
```

**Flags:**
- `--count`: Total number of pods to generate (default: 1000).
- `--template`: Path to the YAML profile defining the pod's shape.
- `--out-dir`: Local directory to write YAML manifests. If empty, a unique temporary directory is created.
- `--concurrency`: Number of concurrent workers for generation and disk I/O (default: 50).

### Library Integration
The fuzzer can be integrated into existing Go test suites for automated scale benchmarking.

```go
import "k8s.io/kubernetes/test/utils/fuzzer"

// 1. Load a profile
template, _ := fuzzer.LoadTemplateFromFile("test/utils/fuzzer/templates/representative-pod.yaml")

// 2. Initialize the creator
// Pass nil for clientset if only writing to local files
creator := fuzzer.NewExemplaryPodCreator(clientset, time.Now().UnixNano())

// 3. Inject 50,000 pods concurrently
err := creator.CreateExemplaryPods(ctx, template, 50000, 100)
```

## Template Definition
Templates allow you to define the "shape" of the synthetic data. A representative template (`representative-pod.yaml`) is included in the `templates/` directory.

### Example Template Structure
```yaml
name: "representative-pod"
baseSpec:
  containers:
  - name: "fuzz-target"
    image: "gcr.io/google-containers/pause:3.9"
  nodeSelector:
    disktype: "non-existent-ssd" # Safety
managedFields:
- manager: "kube-scheduler"
  operation: "Update"
  length: 5000 # Bloats the JSON to 5KB for interning tests
annotations:
- key: "fuzz.metadata/large-blob"
  length: 24000 # Adds a 24KB annotation
```

## Project Structure
- `fuzzer.go`: Core library containing the fuzzer engine and creator utility.
- `fuzzer_test.go`: Unit tests verifying metadata stability and interning support.
- `cmd/`: CLI implementation for standalone manifest generation.
- `templates/`: Pre-defined YAML profiles for common scale-testing scenarios.
