<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [GPU support](#gpu-support)
  - [Objective](#objective)
  - [Background](#background)
  - [Detailed discussion](#detailed-discussion)
    - [Inventory](#inventory)
    - [Scheduling](#scheduling)
    - [The runtime](#the-runtime)
      - [NVIDIA support](#nvidia-support)
    - [Event flow](#event-flow)
    - [Too complex for now: nvidia-docker](#too-complex-for-now-nvidia-docker)
  - [Implementation plan](#implementation-plan)
    - [V0](#v0)
      - [Scheduling](#scheduling-1)
      - [Runtime](#runtime)
      - [Other](#other)
  - [Future work](#future-work)
    - [V1](#v1)
    - [V2](#v2)
    - [V3](#v3)
    - [Undetermined](#undetermined)
  - [Security considerations](#security-considerations)

<!-- END MUNGE: GENERATED_TOC -->

# GPU support

Author: @therc

Date: Apr 2016

Status: Design in progress, early implementation of requirements

## Objective

Users should be able to request GPU resources for their workloads, as easily as
for CPU or memory. Kubernetes should keep an inventory of machines with GPU
hardware, schedule containers on appropriate nodes and set up the container
environment with all that's necessary to access the GPU. All of this should
eventually be supported for clusters on either bare metal or cloud providers.

## Background

An increasing number of workloads, such as machine learning and seismic survey
processing, benefits from offloading computations to graphic hardware. While not
as tuned as traditional, dedicated high performance computing systems such as
MPI, a Kubernetes cluster can still be a great environment for organizations
that need a variety of additional, "classic" workloads, such as database, web
serving, etc.

GPU support is hard to provide extensively and will thus take time to tame
completely, because

- different vendors expose the hardware to users in different ways
- some vendors require fairly tight coupling between the kernel driver
controlling the GPU and the libraries/applications that access the hardware
- it adds more resource types (whole GPUs, GPU cores, GPU memory)
- it can introduce new security pitfalls
- for systems with multiple GPUs, affinity matters, similarly to NUMA
considerations for CPUs
- running GPU code in containers is still a relatively novel idea

## Detailed discussion

Currently, this document is mostly focused on the basic use case: run GPU code
on AWS `g2.2xlarge` EC2 machine instances using Docker. It constitutes a narrow
enough scenario that it does not require large amounts of generic code yet. GCE
doesn't support GPUs at all; bare metal systems throw a lot of extra variables
into the mix.

Later sections will outline future work to support a broader set of hardware,
environments and container runtimes.

### Inventory

Before any scheduling can occur, we need to know what's available out there. In
v0, we'll hardcode capacity detected by the kubelet based on a flag,
`--experimental-nvidia-gpu`. This will result in the user-defined resource
`alpha.kubernetes.io/nvidia-gpu` to be reported for `NodeCapacity` and
`NodeAllocatable`, as well as as a node label.

### Scheduling

GPUs will be visible as first-class resources. In v0, we'll only assign whole
devices; sharing among multiple pods is left to future implementations. It's
probable that GPUs will exacerbate the need for [a rescheduler](rescheduler.md)
or pod priorities, especially if the nodes in a cluster are not homogeneous.
Consider these two cases:

> Only half of the machines have a GPU and they're all busy with other
workloads. The other half of the cluster is doing very little work. A GPU
workload comes, but it can't schedule, because the devices are sitting idle on
nodes that are running something else and the nodes with little load lack the
hardware.

> Some or all the machines have two graphic cards each. A number of jobs get
scheduled, requesting one device per pod. The scheduler puts them all on
different machines, spreading the load, perhaps by design. Then a new job comes
in, requiring two devices per pod, but it can't schedule anywhere, because all
we can find, at most, is one unused device per node.

### The runtime

Once we know where to run the container, it's time to set up its environment. At
a minimum, we'll need to map the host device(s) into the container. Because each
manufacturer exposes different device nodes (`/dev/ati/card0`, `/dev/nvidia0`,
but also the required `/dev/nvidiactl` and `/dev/nvidia-uvm`), some of the logic
needs to be hardware-specific, mapping from a logical device to a list of device
nodes necessary for software to talk to it.

Support binaries and libraries are often versioned along with the kernel module,
so there should be further hooks to project those under `/bin` and some kind of
`/lib` before the application is started. This can be done for Docker with the
use of a versioned [Docker
volume](https://docs.docker.com/engine/tutorials/dockervolumes/) or
with upcoming Kubernetes-specific hooks such as init containers and volume
containers. In v0, images are expected to bundle everything they need.

#### NVIDIA support

The first implementation and testing ground will be for NVIDIA devices, by far
the most common setup.

In v0, the `--experimental-nvidia-gpu` flag will also result in the host devices
(limited to those required to drive the first card, `nvidia0`) to be mapped into
the container by the dockertools library.

### Event flow

This is what happens before and after an user schedules a GPU pod.

1. Administrator installs a number of Kubernetes nodes with GPUs. The correct
kernel modules and device nodes under `/dev/` are present.

1. Administrator makes sure the latest CUDA/driver versions are installed.

1. Administrator enables `--experimental-nvidia-gpu` on kubelets

1. Kubelets update node status with information about the GPU device, in addition
to cAdvisor's usual data about CPU/memory/disk

1. User creates a Docker image compiling their application for CUDA, bundling
the necessary libraries. We ignore any versioning requirements in the image
using labels based on [NVIDIA's
conventions](https://github.com/NVIDIA/nvidia-docker/blob/64510511e3fd0d00168eb076623854b0fcf1507d/tools/src/nvidia-docker/utils.go#L13).

1. User creates a pod using the image, requiring
`alpha.kubernetes.io/nvidia-gpu: 1`

1. Scheduler picks a node for the pod

1. The kubelet notices the GPU requirement and maps the three devices. In
Docker's engine-api, this means it'll add them to the Resources.Devices list.

1. Docker runs the container to completion

1. The scheduler notices that the device is available again

### Too complex for now: nvidia-docker

For v0, we discussed at length, but decided to leave aside initially the
[nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker). The plugin is
an officially supported solution, thus avoiding a lot of new low level code, as
it takes care of functionality such as:

- creating a Docker volume with binaries such as `nvidia-smi` and shared
libraries
- providing HTTP endpoints that monitoring tools can use to collect GPU metrics
- abstracting details such as `/dev` entry names for each device, as well as
control ones like `nvidiactl`

The `nvidia-docker` wrapper also verifies that the CUDA version required by a
given image is supported by the host drivers, through inspection of well-known
image labels, if present. We should try to provide equivalent checks, either
for CUDA or OpenCL.

This is current sample output from `nvidia-docker-plugin`, wrapped for
readability:

    $ curl -s localhost:3476/docker/cli
    --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0
    --volume-driver=nvidia-docker
    --volume=nvidia_driver_352.68:/usr/local/nvidia:ro

It runs as a daemon listening for HTTP requests on port 3476. The endpoint above
returns flags that need to be added to the Docker command line in order to
expose GPUs to the containers. There are optional URL arguments to request
specific devices if more than one are present on the system, as well as specific
versions of the support software. An obvious improvement is an additional
endpoint for JSON output.

The unresolved question is whether `nvidia-docker-plugin` would run standalone
as it does today (called over HTTP, perhaps with endpoints for a new Kubernetes
resource API) or whether the relevant code from its `nvidia` package should be
linked directly into kubelet. A partial list of tradeoffs:

|                     | External binary                                                                                   | Linked in                                                    |
|---------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| Use of cgo          | Confined to binary                                                                                | Linked into kubelet, but with lazy binding                   |
| Expandibility       | Limited if we run the plugin, increased if library is used to build a Kubernetes-tailored daemon. | Can reuse the `nvidia` library as we prefer                  |
| Bloat               | None                                                                                              | Larger kubelet, even for systems without GPUs                |
| Reliability         | Need to handle the binary disappearing at any time                                                | Fewer headeaches                                             |
| (Un)Marshalling     | Need to talk over JSON                                                                            | None                                                         |
| Administration cost | One more daemon to install, configure and monitor                                                 | No extra work required, other than perhaps configuring flags |
| Releases            | Potentially on its own schedule                                                                   | Tied to Kubernetes'                                          |

## Implementation plan

### V0

The first two tracks can progress in parallel.

#### Scheduling

1. Define new resource `alpha.kubernetes.io:nvidia-gpu` in `pkg/api/types.go`
and co.
1. Plug resource into feasability checks used by kubelet, scheduler and
schedulercache. Maybe gated behind a flag?
1. Plug resource into resource_helpers.go
1. Plug resource into the limitranger

#### Runtime

1. Add kubelet config parameter to enable the resource
1. Make kubelet's `setNodeStatusMachineInfo` report the resource
1. Add a Devices list to container.RunContainerOptions
1. Use it from DockerManager's runContainer
1. Do the same for rkt (stretch goal)
1. When a pod requests a GPU, add the devices to the container options

#### Other

1. Add new resource to `kubectl describe` output. Optional for non-GPU users?
1. Administrator documentation, with sample scripts
1. User documentation

## Future work

Above all, we need to collect feedback from real users and use that to set
priorities for any of the items below.

### V1

- Perform real detection of the installed hardware
- Figure a standard way to avoid bundling of shared libraries in images
- Support fractional resources so multiple pods can share the same GPU
- Support bare metal setups
- Report resource usage

### V2

- Support multiple GPUs with resource hierarchies and affinities
- Support versioning of resources (e.g. "CUDA v7.5+")
- Build resource plugins into the kubelet?
- Support other device vendors
- Support Azure?
- Support rkt?

### V3

- Support OpenCL (so images can be device-agnostic)

### Undetermined

It makes sense to turn the output of this project (external resource plugins,
etc.) into a more generic abstraction at some point.


## Security considerations

There should be knobs for the cluster administrator to only allow certain users
or roles to schedule GPU workloads. Overcommitting or sharing the same device
across different pods is not considered safe. It should be possible to segregate
such GPU-sharing pods by user, namespace or a combination thereof.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/gpu-support.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
