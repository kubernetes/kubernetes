# Kubelet Executes Commands in Sidecar Containers

Currently Kubelet uses [exec interface](https://github.com/kubernetes/kubernetes/tree/master/pkg/util/exec) to invoke executables that are installed on the host. Packages that contain the executables must be installed on the nodes and produce the same output that `exec` callers expect. As a result, package installation, version tracking, and upgrade complicate both Kubernetes configuration and development.

This Sidecar container proposal eliminates the host-package dependency by allowing Kubelet to pull container images that have the required executables and run the executables inside the container. 

## Use Cases

### Volume Plugins
* rbd volume plugin currently executes `rbd` command that is available in `ceph-common` package. A rbd sidecar container can execute `rbd` command from [ceph/base](https://github.com/ceph/ceph-docker/tree/master/base) container image.
* iSCSI volume plugin executes `iscsiadmin` command to configure iSCSI devices.
* Glusterfs volume plugin requires glusterfs-fuse package that provides mount helper to mount a Glusterfs volume. 


## Utilities that Vary on Different OSes
Some packages that Kubernetes use are missing on some OSes. [A recent example](https://github.com/kubernetes/kubernetes/pull/14109) illustrates that utilities used by some OSes are missing on other platforms. In such case, a dynamically pulled sidecar containers can resolve such differences.

## Scope

Sidecar containers are created and scheduled by Kubelet, rather than API server. Sidecar containers can be long running or exit after command is finished. Long running containers are separately discussed in [Daemon Pod](https://github.com/kubernetes/kubernetes/issues/1518). This proposal focuses on short-lived containers that exit after command is finished.

## Structure
A proposed interface to invoke a sidecar container and execute a command is as the following:

```go
func RunContainerInSidecarContainer(containerImageName string, cmd []string, args []string) ([]byte, error)
```
where `containerImageName` is the container's image name that is used to pull from image repository. `cmd` is the executable's path and name. `args` is the executable's runtime argument. This function returns the combined output as `exec.CombinedOutput` and `error`. This function acts like `docker run`

`RunContainerInSidecarContainer` creates a Pod and Container and invokes new function in `ContainerCommandRunner` interface to create a container, exec the container, and obtain the output if needed.


## Providing Sidecar Container Images

Sidecar container images can be hard-coded, user-provided, or centrally managed.

* Hard-coded. Sidecar container callers explicitly specify the name, command, and environment of the container.
* User-provided. Pod authors provide the container information, along with other clauses in the Pod. 
* Centrally managed. A new k/v service is provided to map sidecar containers and their intended uses. It is similar to Docker registry or yum repository service. Kubelet uses a reference key (e.g. "rbd client container") and looks up to find the matching container. 

