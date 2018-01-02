# Using rkt with Kubernetes (aka "rktnetes")

[Kubernetes][k8s] is a system for managing containerized applications across a cluster of machines.
Kubernetes runs all applications in containers.
In the default setup, this is performed using the Docker engine, but Kubernetes also features support for using rkt as its container runtime backend.
This allows a Kubernetes cluster to leverage some of rkt's security features and native pod support.

## Configuring rkt as the Kubernetes container runtime

The container runtime is configured at the _kubelet_ level.
The kubelet is the agent that runs on each machine to manage containers.
The kubelet provides several flags to use rkt as the container runtime:

- `--container-runtime=rkt` Sets the node's container runtime to rkt.
- `--rkt-api-endpoint=HOST:PORT` Sets the endpoint of the rkt API service. Default to `localhost:15441`.
- `--rkt-path=PATH_TO_RKT_BINARY` Sets the path of the rkt binary. If empty, it will search for rkt in `$PATH`.
- `--rkt-stage1-image=STAGE1_NAME` Sets the name of the stage1 image, e.g. `coreos.com/rkt/stage1-coreos`. If not set, the default stage1 image (`coreos.com/rkt/stage1-coreos`) is used.

Check the [rktnetes getting started guide][rktnetes] for information about setting up and using a rktnetes cluster.

## Configuring rkt using supported setup tools
The [coreos-kubernetes][coreos-kubernetes] and [coreos-baremetal][coreos-baremetal] repos both support configuring rkt as the Kubernetes runtime out of the box.

Check out the coreos-kubernetes repo if you want to spin up a cluster on [AWS][k8s-on-aws] or [locally with Vagrant][k8s-on-vagrant]. The common configuration option here is setting `CONTAINER_RUNTIME` environment variable to rkt.

For baremetal, check out the Kubernetes guides [here][k8s-baremetal].

## Using [Minikube][minikube]

Minikube is a tool that makes it easy to run Kubernetes locally. It launches a single-node cluster inside a VM aimed at users looking to try out Kubernetes. Follow the instructions in the Minikube [Quickstart][minikube-quickstart] section on how to get started with rktnetes.

### Current Status

Integration of rkt as a container runtime was officially [announced in the Kubernetes 1.3 release][k8s-1.3-release].
Known issues and tips for using rkt with Kubernetes can be found in the [rktnetes notes][rktnetes-notes].


[coreos-baremetal]: https://github.com/coreos/coreos-baremetal
[coreos-kubernetes]: https://github.com/coreos/coreos-kubernetes
[k8s]: http://kubernetes.io
[k8s-1.3-release]: http://blog.kubernetes.io/2016/07/rktnetes-brings-rkt-container-engine-to-Kubernetes.html
[k8s-on-aws]: https://coreos.com/kubernetes/docs/latest/kubernetes-on-aws.html
[k8s-baremetal]: https://github.com/coreos/coreos-baremetal/blob/master/Documentation/kubernetes.md
[k8s-on-vagrant]: https://coreos.com/kubernetes/docs/latest/kubernetes-on-vagrant-single.html
[minikube]: https://github.com/kubernetes/minikube
[minikube-quickstart]: https://github.com/kubernetes/minikube/blob/master/README.md#quickstart
[rktnetes]: http://kubernetes.io/docs/getting-started-guides/rkt/
[rktnetes-notes]: http://kubernetes.io/docs/getting-started-guides/rkt/notes/
