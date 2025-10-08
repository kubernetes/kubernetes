# Windows Node End-To-End (e2e) tests

Node e2e tests are component tests meant for testing the Kubelet code on a custom host environment,
 They are now supported on the Windows platform and can be executed locally; however, Windows support 
 is still in the early stage, and not all tests are supported yet.
 Remote execution is planned as the next step.

*Note: There is no scheduler running. The e2e tests have to do manual scheduling, e.g. by using `framework.PodClient`.*

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

