Device spec
===========

> PathOnHost:PathInContainer:Permissions:ContainerName

| Spec items      | Requirements                                                   |
|-----------------|----------------------------------------------------------------|
| PathOnHost      | is required                                                    |
| PathInContainer | if not specified will used value of PathOnHost                 |
| Permissions     | if not specified will used full privileges (mrw)               |
| ContainerName   | if not specified this device will be mapped to all containers  |

Examples
========
 - /dev/kvm:/dev/kvm:mrw           (full spec for all containers)
 - /dev/kvm:/dev/kvm:mrw:alpine1   (full spec for one container with name alpine1)
 - /dev/kvm:/dev/kvm-new:rw        (/dev/kvm -> /dev/kvm-new without mknod for all containers)
 - /dev/kvm::mrw                   (device will be mapped as one to one for all containers)
 - /dev/kvm                        (device will be mapped as one to one, and will be applied default permissions: "mrw")

Annotations example
===================
> Full spec
```yaml
metadata:
  annotations:
    device.alpha.kubernetes.io/allow-list: "/dev/kvm:/dev/kvm:mrw;/dev/fuse:/dev/fuse:mrw:alpine2"
```
> Short spec
```yaml
metadata:
  annotations:
    device.alpha.kubernetes.io/allow-list: "/dev/kvm;/dev/fuse"
```

K8s manifest
============
```bash
echo '
---
apiVersion: v1
kind: Pod
metadata:
  name: alpine
  labels:
    app: alpine
  annotations:
    device.alpha.kubernetes.io/allow-list: "/dev/kvm;/dev/fuse::rw:alpine2"
spec:
  containers:
  - name: alpine1
    image: alpine
    command: ["ls"]
    args: ["-la", "/dev"]
  - name: alpine2
    image: alpine
    command: ["ls"]
    args: ["-la", "/dev"]
' | kubectl create -f -
```

Show logs for container: alpine1
```bash
kubectl logs alpine alpine1
```

Show logs for container: alpine2
```bash
kubectl logs alpine alpine2
```
