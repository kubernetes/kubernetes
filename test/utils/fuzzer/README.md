# Exemplary Pod Fuzzer

The Exemplary Pod Fuzzer is a high-fidelity synthetic pod generator designed to stress-test the Kubernetes control plane. It uses a **"Sanitize & Clone"** model: it takes a real production pod manifest as a "base", scrubs all PII/sensitive data, and preserves the exact structural complexity (volume mounts, env vars, managed fields) for benchmarking.

## Key Features

- **Direct Sanitization**: Automatically scrubs Names, Namespaces, UIDs, and OwnerReferences from a real pod.
- **Spec Randomization**: Randomizes all environment variable keys and values while maintaining the original count and structure.
- **ManagedFields Cloning**: Clones the exact history and field ownership of the base pod, but randomizes the internal JSON paths (supporting both `f:` and `k:` prefixes).
- **Safety by Default**: Forces pods to be unschedulable (Pending state) using non-existent nodeSelectors and schedulerNames.
- **Performance Optimized**: Uses high-performance `client-go` settings (500 QPS / 1000 Burst) and precomputes "Fuzzed Prototypes" to ensure string interning memory optimizations are correctly triggered.

## Usage

### 1. Generate Pod Manifests
To generate 1,000 fuzzed manifests based on a real pod:
```bash
go run test/utils/fuzzer/cmd/main.go \
  --base-pod path/to/real-pod.yaml \
  --name-prefix representative-pod \
  --namespace my-test-ns \
  --count 1000 \
  --out-dir ./generated-pods
```

### 2. Direct Cluster Injection
To inject 50,000 pods directly into a test cluster:
```bash
go run test/utils/fuzzer/cmd/main.go \
  --base-pod test/utils/fuzzer/templates/complex-daemonset.yaml \
  --count 50000 \
  --concurrency 100
```

## Flags

- `--base-pod`: Path to the real `v1.Pod` YAML manifest used as a structural source.
- `--name-prefix`: Prefix for the generated pod names (default: `fuzzed-pod`).
- `--namespace`: Target namespace for the generated pods (default: `fuzz-test`).
- `--count`: Number of pods to generate.
- `--offset`: Starting index for naming (useful for incremental runs).
- `--concurrency`: Number of concurrent workers (default: 50).
- `--out-dir`: If specified, write YAMLs to this directory instead of injecting into a cluster.

## Verification & Metrics

To quickly verify the number of pods in the API server storage without timing out:
```bash
kubectl get --raw /metrics | grep 'apiserver_storage_objects{resource="pods"}'
```
