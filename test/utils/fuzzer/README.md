# Exemplary Pod Fuzzer

The Exemplary Pod Fuzzer is a high-fidelity synthetic pod generator designed to stress-test the Kubernetes control plane. It uses a **"Sanitize & Clone"** model: it takes a real production pod manifest as a "base", scrubs all PII/sensitive data, and preserves the exact structural complexity (volume mounts, env vars, managed fields) for benchmarking.

## Key Features

- **Direct Sanitization**: Automatically scrubs Names, Namespaces, UIDs, and OwnerReferences from a real pod.
- **Spec Randomization**: Randomizes all environment variable keys and values while maintaining the original count and structure.
- **ManagedFields Cloning**: Clones the exact history and field ownership of the base pod, but randomizes the internal JSON paths (supporting both `f:` and `k:` prefixes).
- **Safety by Default**: Forces pods to be unschedulable (Pending state) using non-existent nodeSelectors and schedulerNames.
- **Performance Optimized**: Uses high-performance `client-go` settings (500 QPS / 1000 Burst) and precomputes "Fuzzed Prototypes" to ensure string interning memory optimizations are correctly triggered.

## Benchmarking with Kind

To perform a realistic 50,000 pod stress test on a local `kind` cluster, follow these steps:

### 1. Create a Large-Scale Kind Cluster
Standard `kind` clusters have a default `etcd` quota that is too small for 50k heavy pods. Use this configuration to increase the quota to 8GB:

```bash
cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    etcd:
      local:
        extraArgs:
          quota-backend-bytes: "8589934592"
EOF

kind create cluster --config kind-config.yaml --name fuzzer-test
```

### 2. Prepare the Namespace and Dependencies
Real production pods (like the templates in this repo) often have dependencies like specific **Namespaces** or **ServiceAccounts**. The fuzzer preserves these references. You must create them before injecting pods:

```bash
# Example: Using the complex-daemonset.yaml base
kubectl create namespace fuzz-test
kubectl create serviceaccount cilium -n fuzz-test
```

### 3. Run the Fuzzer
Inject the pods using high concurrency.

```bash
go run test/utils/fuzzer/cmd/main.go \
  --base-pod test/utils/fuzzer/templates/complex-daemonset.yaml \
  --namespace fuzz-test \
  --name-prefix representative-pod \
  --count 50000 \
  --concurrency 100
```

## Usage (General)

### Generate Pod Manifests to Disk
To generate fuzzed manifests for manual inspection:
```bash
go run test/utils/fuzzer/cmd/main.go \
  --base-pod path/to/real-pod.yaml \
  --name-prefix representative-pod \
  --namespace my-test-ns \
  --count 1000 \
  --out-dir ./generated-pods
```

## Flags

- `--base-pod`: Path to the real `v1.Pod` YAML manifest used as a structural source.
- `--name-prefix`: Prefix for the generated pod names (default: `fuzzed-pod`).
- `--namespace`: Target namespace for the generated pods (default: `fuzz-test`).
- `--count`: Number of pods to generate.
- `--offset`: Starting index for naming (useful for incremental runs).
- `--concurrency`: Number of concurrent workers (default: 50).
- `--kubeconfig`: Path to the kubeconfig file. Defaults to `$HOME/.kube/config`.
- `--out-dir`: If specified, write YAMLs to this directory instead of injecting into a cluster.

## Verification & Metrics

To quickly verify the number of pods in the API server storage without timing out:
```bash
kubectl get --raw /metrics | grep 'apiserver_storage_objects{resource="pods"}'
```

To measure control plane memory usage:
```bash
docker exec fuzzer-test-control-plane ps aux | grep -E "kube-apiserver|etcd|kube-scheduler"
```
