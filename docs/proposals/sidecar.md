# Executes Commands in a Local Pod from Kubelet

Currently Kubelet uses [exec interface](https://github.com/kubernetes/kubernetes/tree/master/pkg/util/exec) to invoke executables that are installed on the host. Packages that contain the executables must be installed on the nodes and produce the same output that `exec` callers expect. As a result, package installation, version tracking, and upgrade complicate both Kubernetes configuration and development.

This proposal aims to containerize command in a local Pod and thus eliminates the host-package dependency. Kubelet can pull container images that have the required executables and run the executables inside the local Pod. 

## Use Cases

### Volume Plugins
* rbd volume plugin currently executes `rbd` command that is available in `ceph-common` package. Containerized `rbd` can execute `rbd` command from [ceph/base](https://github.com/ceph/ceph-docker/tree/master/base) container image.
* iSCSI volume plugin executes `iscsiadmin` command to configure iSCSI devices.
* Glusterfs volume plugin requires glusterfs-fuse package that provides mount helper to mount a Glusterfs volume. 


## Utilities that Vary on Different OSes
Some packages that Kubernetes use are missing on some OSes. [A recent example](https://github.com/kubernetes/kubernetes/pull/14109) illustrates that utilities used by some OSes are missing on other platforms. In such case, a dynamically pulled containers can resolve such differences.

## Scope

Local Pods are created and scheduled by Kubelet, rather than API server. Local Pods can be long running or exit after command is finished. Long running Pods are separately discussed in [Daemon Pod](https://github.com/kubernetes/kubernetes/issues/1518). This proposal focuses on short-lived Pods that exit after command is finished.

## Structure
A proposed interface to invoke a local Pod and execute a command is as the following:

```go
func RunCommandInContainer(pod *api.Pod, container *api.Container, cmd []string, args []string) ([]byte, error)
```
where `pod` is Pod, `container` is the a container spec in the `pod`. `cmd` is the executable's path and name. `args` is the executable's runtime argument. This function returns the combined output as `exec.CombinedOutput` and `error`. This function acts like `docker run`

`RunCommandInContainer` creates a Pod and Container and invokes new function in `ContainerCommandRunner` interface to create a container, exec the container, and obtain the output if needed.

`RunCommandInContainer` is expected to be added to Docker, rkt, and hyper runtime.

## Providing Local Container Images

Local container images can be hard-coded, user-provided, or centrally managed.

* Hard-coded. Local container callers explicitly specify the name, command, and environment of the container.
* User-provided. Pod authors provide the container information, along with other clauses in the Pod. 
* Centrally managed. A new k/v service is provided to map local containers and their intended uses. It is similar to Docker registry or yum repository service. Kubelet uses a reference key (e.g. "rbd client container") and looks up to find the matching container. 

