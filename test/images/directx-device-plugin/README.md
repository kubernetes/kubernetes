# DirectX device plugin for Kubernetes

## Requirements

- Windows Server 2019 1809 or above
- docker 19.03 or above
- kubelet for windows has to support device manager, PR made here : https://github.com/kubernetes/kubernetes/pull/80917

## Build

```bash
GOOS=windows GOARCH=amd64 go build -mod vendor -o k8s-directx-device-plugin.exe cmd/k8s-device-plugin/main.go
```

## Run

```powershell
c:\k\k8s-directx-device-plugin.exe
```

Available environments variables :
- `PLUGIN_SOCK_DIR`  default value is `c:\var\lib\kubelet\device-plugins\`
- `DIRECTX_GPU_MATCH_NAME` default value is `nvidia`

## How to use

You can now request resources of type microsoft.com/directx in the container definition, the plugin will automatically add class/5B45201D-F2F2-4F3B-85BB-30FF1F953599 as a container device (which is the Docker for Windows way of enabling GPUs in containers).

```yaml
...
spec:
  containers:
...
    resources:
      requests:
        microsoft.com/directx: "1"
...
```

## Links

- https://docs.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/gpu-acceleration
- https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/
- https://techcommunity.microsoft.com/t5/Containers/Bringing-GPU-acceleration-to-Windows-containers/ba-p/393939
