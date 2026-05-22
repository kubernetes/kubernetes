# Kubernetes

## Communication Preferences

- Dry, concise, low-key humor. No flattery, no forced memes. Skip preambles and postambles.
- Comments explain "why", not "what".
- Error messages: actionable and specific. No vague "something went wrong" output.

## Constraints

- **Generated files are read-only.** Never hand-edit `zz_generated.*` or `generated.pb.go`. Run `make update`.
- **go.mod/go.work are generated.** Use `hack/pin-dependency.sh` + `hack/update-vendor.sh`. Never `go mod tidy`.
- **Staging is source of truth** for `k8s.io/*` (`staging/src/k8s.io/`). Never import `k8s.io/kubernetes` from staging.
- **Boilerplate required.** Every `.go` file needs the license header from `hack/boilerplate/boilerplate.go.txt`.

## Commands

Run `make help` for all available targets. Common workflows:

```
make test WHAT=./pkg/kubelet GOFLAGS=-v     # Unit tests (one package)
make test-integration WHAT=./test/integration/scheduler
make verify                                 # All verification checks
make update                                 # ALL generators and formatters
```

## Style

- Packages: lowercase, single word, match directory.
