# Windows Node End-To-End (e2e) tests

Node e2e tests are component tests meant for testing the Kubelet code on a custom single-host environment with the fake api server running on the same host. This directory include tests on the Windows platform and can be executed locally. The test framework and set of tests are replicating the non-Windows tests under [e2e_node](../e2e_node), however, Windows support is still in the early stage, and not all tests are supported yet. Remote execution is planned as the next step.

For general background on node e2e tests, see the SIG Node [Node End-to-End Tests guide](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-node/e2e-node-tests.md).

*Note: There is no scheduler running. The e2e tests have to do manual nodeName assignment.

# Running tests

## Locally

Why run tests *locally*? It is much faster than running tests remotely.

Prerequisites:
- [containerd](https://github.com/containerd/containerd/blob/main/docs/getting-started.md#installing-containerd-on-windows), 
containerd will need to be ran with admin privilege
- Working CNI
  - Ensure that you have a valid CNI configuration. For testing purposes, a [nat](https://www.jamessturtevant.com/posts/Windows-Containers-on-Windows-10-without-Docker-using-Containerd/) configuration should work. 

From the Kubernetes base directory, run with admin privilege:

```cmd
go build -o ./test/e2e_node_windows/_output/kubelet.exe .\cmd\kubelet\ 

go build -o ./test/e2e_node_windows/_output/kube-log-runner.exe .\staging\src\k8s.io\component-base\logs\kube-log-runner

go test ./test/e2e_node_windows -c 

 .\e2e_node_windows.test.exe --bearer-token=vQIYfdCt7wIFOZtO --test.v --test.paniconexit0 --container-runtime-endpoint "npipe://./pipe/containerd-containerd" --prepull-images=false --ginkgo.focus "when creating a windows static pod" --k8s-bin-dir ./test/e2e_node_windows/_output/
```

It will run the specified windows e2e node tests, which will in turn:
- Build the Kuberlet source code
- Start a local instance of *etcd*
- Start a local instance of *kube-apiserver*
- Start a local instance of *kubelet*
- Run the test using the locally started processes
- Output the test results to STDOUT
- Stop *kubelet*, *kube-apiserver*, and *etcd*

### Log files

Test execution produces several log streams:

- **Ginkgo / `go test` output** (pass/fail, test logs): written to STDOUT when `--test.v` is used. Redirect with shell pipes if you need a file copy (e.g. `... > run.log 2>&1`).
- **`services.log`** (etcd + kube-apiserver wrapper output): written to `<report-dir>/services.log`.
- **`kubelet.log`** (kubelet output): written to `<report-dir>/kubelet.log`.

`<report-dir>` is controlled by the `--report-dir=<path>` flag. If not set, the wrapper log files are created in the test binary's current working directory.

## Cross-building from Linux

The Kubernetes dockerized build handles cross-compilation automatically. To build the kubelet and the Windows node e2e test binary from a Linux host:

```bash
# Build kubelet.exe, kube-log-runner.exe and the e2e test binary for Windows
KUBE_BUILD_PLATFORMS=windows/amd64 make WHAT="cmd/kubelet staging/src/k8s.io/component-base/logs/kube-log-runner test/e2e_node_windows/e2e_node_windows.test"
```

Or build them separately:

```bash
# Build only kubelet.exe
KUBE_BUILD_PLATFORMS=windows/amd64 make WHAT="cmd/kubelet"

# Build only the e2e test binary
KUBE_BUILD_PLATFORMS=windows/amd64 make WHAT="test/e2e_node_windows/e2e_node_windows.test"
```

The output binaries will be placed under `_output/dockerized/bin/windows/amd64/`.

Once built, copy the binaries to the Windows test machine and run as described in the [Locally](#locally) section above.

