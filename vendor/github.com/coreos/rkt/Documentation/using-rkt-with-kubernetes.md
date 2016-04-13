# Using rkt with Kubernetes (aka "rktnetes")

[Kubernetes](http://kubernetes.io) is a system for managing containerized applications across a cluster of machines.
Kubernetes runs all applications in containers.
In the default setup, this is performed using the Docker engine, but Kubernetes also features support for using rkt as its container runtime backend.
This allows a Kubernetes cluster to leverage some of rkt's security features and native pod support.

## Configuring rkt as the Kubernetes container runtime

The container runtime is configured at the _kubelet_ level.
The kubelet is the agent that runs on each machine to manage containers.
The kubelet provides several flags to use rkt as the container runtime:
- `--container-runtime=rkt` chooses rkt as the runtime.
- `--rkt-path` sets the rkt binary path.
- `--rkt-stage1-image` sets the stage1 image path.

The [getting started with rkt guide][] in the upstream Kubernetes documentation provides more detailed information about how to launch a kubernetes cluster with rkt, how to debug it, and more.

[getting started with rkt guide]: http://kubernetes.io/docs/getting-started-guides/rkt/

### Current Status

Integration of rkt as a container runtime for Kubernetes is under active development.
For the latest information on the progress of the integration, check out [this Google doc][rkt-k8s-checklist] which tracks the detailed status of implemented functionality.

[rkt-k8s-checklist]: https://docs.google.com/document/d/1dYxInIUDTm4HEArQ9Hom_1NhYw22WrXWdglnaLjtQsI/edit
