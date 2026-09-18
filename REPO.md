# Kubernetes — Repository Context

## Purpose
Production-Grade Container Scheduling and Management.

## GitHub
https://github.com/kubernetes/kubernetes

## Language
Go

## Fork / Local Path
`c:\Users\parik\Downloads\lfxx\Batch-02-Cloud-Infrastructure\Kubernetes`

---

## Architecture Overview
Kubernetes control plane and node architecture:
- `pkg/kubelet`: Node agent
- `pkg/controller`: Controllers (daemonset, deployment, job, etc.)
- `pkg/scheduler`: Pod scheduler
- `staging/src/k8s.io/kubectl`: Command line interface and client-side drain/eviction logic
- `staging/src/k8s.io/apiserver`: Core API machinery and HTTP handlers
- `staging/src/k8s.io/client-go`: Client SDK

## Build
```bash
make
go build ./...
```

## Test Commands
```bash
go test -v ./staging/src/k8s.io/kubectl/pkg/drain
```

## PR Conventions
- Title format: `fix(scope): concise description`
- Draft PR opened, Closes issue #

## Previous Contributions from This Workspace
- **PR #141607**: `fix(kubectl): stop drain eviction retry loop when pod is recreated or rescheduled` (Closes #138229)
