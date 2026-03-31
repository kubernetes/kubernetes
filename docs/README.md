# Kubernetes Repository Deep Dive Docs

This folder adds a code-oriented learning layer for `kubernetes/kubernetes`. It is intentionally different from the end-user documentation on `kubernetes.io`: the goal here is to explain how the repository is wired internally.

## Available language editions

- English: [`docs/en/README.md`](en/README.md)
- Chinese: [`docs/zh/README.md`](zh/README.md)

## What this series explains

- how the repository is laid out
- how `kube-apiserver`, `kube-scheduler`, `kube-controller-manager`, and `kubelet` fit together
- how a Pod request becomes a running workload
- how scheduler formulas turn raw resources into placement decisions
- how controller queues, retries, and kubelet pod workers actually work

## Source-first, not marketing-first

The documents in this folder are anchored in the repository itself, especially these paths:

- `cmd/`
- `pkg/`
- `staging/src/k8s.io/apiserver/`
- `staging/src/k8s.io/client-go/`
- `pkg/scheduler/`
- `pkg/controller/`
- `pkg/kubelet/`

## Reading order

1. [`docs/en/quick-start.md`](en/quick-start.md)
2. [`docs/en/architecture.md`](en/architecture.md)
3. [`docs/en/control-plane.md`](en/control-plane.md)
4. [`docs/en/math-theory.md`](en/math-theory.md)
5. [`docs/en/controller-kubelet.md`](en/controller-kubelet.md)

## 中文阅读顺序

1. [`docs/zh/quick-start.md`](zh/quick-start.md)
2. [`docs/zh/architecture.md`](zh/architecture.md)
3. [`docs/zh/control-plane.md`](zh/control-plane.md)
4. [`docs/zh/math-theory.md`](zh/math-theory.md)
5. [`docs/zh/controller-kubelet.md`](zh/controller-kubelet.md)
