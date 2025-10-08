# Node E2E Tests

## References

- [e2e-node-tests.md](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-node/e2e-node-tests.md)
- [test-suite.md](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-node/test-suite.md)

## test-suite.md Summary

**Core Design Philosophy**

Node e2e tests exist because testing kubelet doesn't need a full cluster — only kubelet + API server + etcd. The suite was purpose-built to manage these minimal components automatically, both in CI and locally.

**Two Goals, Two Entry Points**

1. **CI verification** — compile test code, run via ginkgo in an automated pipeline (prow/test-infra)
2. **Local development** — contributors can iterate on tests on their own machine without a full cluster

**Execution Flow (`make test-e2e-node`)**

```
build/root/Makefile  (line 291-292)
  target: test-e2e-node: ginkgo        ← first builds ginkgo binary as prerequisite
    → hack/make-rules/test-e2e-node.sh ← sets variables, checks REMOTE=true/false
      → [local mode, default]
          go run test/e2e_node/runner/local/run_local.go
            → builder.BuildGo()
                → BuildTargets(cgo=true)   CGO_ENABLED=1: cmd/kubelet
                → BuildTargets(cgo=false)  CGO_ENABLED=0: test/e2e_node/e2e_node.test
                                                           github.com/onsi/ginkgo/v2/ginkgo
                                                           cluster/gce/gci/mounter
                                                           test/e2e_node/plugins/gcp-credential-provider
            → ginkgo _output/.../e2e_node.test -- <flags>
      → [remote mode, REMOTE=true]
          go run test/e2e_node/runner/remote/run_remote.go
            → provisions GCP VM, copies binaries, runs suite remotely
```

**Build Method: Local vs Dockerized**

`BuildTargets` checks the `--use-dockerized-build` flag:
- **Default (false)**: runs `make -C <k8sRoot> WHAT=<targets>` directly on the host
- **Dockerized (true)**: runs `build/run.sh make WHAT=<targets> KUBE_BUILD_PLATFORMS=<arch>` inside Docker, enabling cross-compilation (e.g., building `windows/amd64` binaries from a Linux CI machine)

`BuildTargets` calls `IsDockerizedBuild()` to check the `--use-dockerized-build` flag and switch between the two paths. `getK8sBin` also calls it to locate the compiled binaries in the correct output directory.

**How `--target-build-arch` and `--use-dockerized-build` are passed**

These flags originate as environment variables read by `hack/make-rules/test-e2e-node.sh`:
```sh
# test-e2e-node.sh lines 125-126
use_dockerized_build=${USE_DOCKERIZED_BUILD:-"false"}
target_build_arch=${TARGET_BUILD_ARCH:-""}
```

- **Remote mode only**: the shell forwards them as CLI flags to the remote runner (lines 229-230):
  ```sh
  --use-dockerized-build="${use_dockerized_build}"
  --target-build-arch="${target_build_arch}"
  ```
- **Local mode**: these flags are NOT forwarded to `run_local.go` — the defaults baked into `flag.String` are used (`linux/amd64` on Linux, `windows/amd64` on Windows).

Example — cross-compiling Windows binaries from a Linux CI machine:
```sh
make test-e2e-node REMOTE=true \
  USE_DOCKERIZED_BUILD=true \
  TARGET_BUILD_ARCH=windows/amd64
```
This causes `BuildTargets` to run:
```sh
build/run.sh make WHAT="cmd/kubelet ..." KUBE_BUILD_PLATFORMS=windows/amd64
```
inside Docker instead of calling `make` directly on the host.

**Why CGO_ENABLED matters**

| Binary | CGO | Reason |
|---|---|---|
| `kubelet` | `CGO_ENABLED=1` | needs C libraries: seccomp, cgroups, libc DNS resolver |
| `e2e_node.test`, `ginkgo`, etc. | `CGO_ENABLED=0` | pure Go, statically linked, portable across machines |

`CGO_ENABLED=0` is critical for the remote runner — binaries are copied to a GCP VM that may have a different libc version. Static binaries have no shared library dependencies and also enable trivial cross-compilation.

**Three Operational Modes** (flags passed to the test binary)

- `--run-services-mode`: starts only etcd + API server + namespace controller
- `--run-kubelet-mode`: starts only kubelet
- `--system-validate-mode`: validates OS/environment prerequisites (docker, kernel features, etc.)
- *(no flag)*: runs the full test suite

**Test Suite Bootstrap (`e2e_node_suite_test.go`)**

The `SynchronizedBeforeSuite` hook runs before all tests:
1. Validates system requirements
2. Pre-pulls required container images
3. Starts background services (etcd → API server → namespace controller)
4. Waits for node to be ready

**Key Insight for Windows Work**

The services layer (`test/e2e_node/services/`) is where OS-specific adaptations matter — the files modified in the `dev` branch (`services/kubelet.go`, `services/server.go`, new `services/kubelet_windows.go`, `services/server_windows.go`) hook directly into the service startup sequence described here. Windows needs its own service management because the process lifecycle and binary names differ from Linux.

**`build_windows.go` in the Windows manual build workflow**

The `README-windows.md` manual workflow bypasses `run_local.go` entirely — binaries are built with explicit `go build` commands and `e2e_node.test.exe` is invoked directly. In this path, `builder.BuildGo()` and `builder.BuildTargets()` are never called, so `build_windows.go` is not part of the build chain.

However, the functions defined inside `build_windows.go` **are** called at test runtime:

| Function | Called from | Purpose |
|---|---|---|
| `GetKubeletServerBin()` | `services/kubelet.go`, `services/kubelet_windows.go` | returns path to `kubelet.exe` when the test suite starts the kubelet subprocess |
| `BuildGo()` | `runner/local/run_local.go`, `remote/node_e2e.go` | invoked only when using the automated runner with `--build-dependencies=true` |
| `IsDockerizedBuild()` + `GetTargetBuildArch()` | `runner/local/run_local.go`, `remote/node_e2e.go` | locates the build output directory |
| `IsTargetArchArm64()` | `remote/utils.go` | selects ARM64-specific GCP machine types (remote runner only) |

The critical one in the manual path is `GetKubeletServerBin()` — even without the runner, when `e2e_node.test.exe` internally starts kubelet as a subprocess, it calls this function. The Windows version returns `kubelet.exe` (with `.exe`) rather than `kubelet`, which is why `build_windows.go` must exist even for the manual workflow.

---

## Appendix: `KUBE_BUILD_PLATFORMS` and Cross-Compiling Windows Kubelet

**What it is:**

`KUBE_BUILD_PLATFORMS` is an env var that tells the Kubernetes build system which `OS/arch` targets to compile for (space-separated). Defined and consumed in `hack/lib/golang.sh`.

- If **unset**: builds for the host platform only
- If **set**: loops over each platform, sets `GOOS`/`GOARCH`, and cross-compiles
- Values are intersected against per-category allowed lists — unsupported targets are silently filtered out

**Supported platforms by category:**

| Category | Notable platforms |
|---|---|
| Server (apiserver, controller-manager) | `linux/*` only — no Windows |
| Node (kubelet) | `linux/*` + `windows/amd64` |
| Client (kubectl) | linux, darwin, `windows/amd64/386/arm64` |
| Test binaries | linux, darwin, `windows/amd64`, `windows/arm64` |

**Cross-building Windows kubelet on a Linux host:**

Direct (no Docker):
```sh
KUBE_BUILD_PLATFORMS=windows/amd64 make all WHAT=cmd/kubelet
# output: _output/local/go/bin/windows_amd64/kubelet.exe
```

Via Docker (recommended for CI):
```sh
build/run.sh make all WHAT=cmd/kubelet KUBE_BUILD_PLATFORMS=windows/amd64
```

**Why no C cross-compiler is needed for Windows:**

The CGO cross-compile toolchain in `hack/lib/golang.sh` is only defined for `linux/*` targets. Windows is absent — so `windows/amd64` always builds with `CGO_ENABLED=0`. Windows kubelet uses `golang.org/x/sys/windows` (pure Go syscall bindings) instead of C libraries, making it fully cross-compilable from Linux.
