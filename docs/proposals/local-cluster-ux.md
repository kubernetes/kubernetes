<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/local-cluster-ux.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Local Cluster Experience

This proposal attempts to improve the existing local cluster experience for kubernetes.
The current local cluster experience is sub-par and often not functional.
There are several options to setup a local cluster (docker, vagrant, linux processes, etc) and we do not test any of them continuously.
Here are some highlighted issues:
- Docker based solution breaks with docker upgrades, does not support DNS, and many kubelet features are not functional yet inside a container.
- Vagrant based solution are too heavy and have mostly failed on OS X.
- Local linux cluster is poorly documented and is undiscoverable.
From an end user perspective, they want to run a kubernetes cluster. They care less about *how* a cluster is setup locally and more about what they can do with a functional cluster.


## Primary Goals

From a high level the goal is to make it easy for a new user to run a Kubernetes cluster and play with curated examples that require least amount of knowledge about Kubernetes.
These examples will only use kubectl and only a subset of Kubernetes features that are available will be exposed.

- Works across multiple OSes - OS X, Linux and Windows primarily.
- Single command setup and teardown UX.
- Unified UX across OSes
- Minimal dependencies on third party software.
- Minimal resource overhead.
- Eliminate any other alternatives to local cluster deployment.

## Secondary Goals

- Enable developers to use the local cluster for kubernetes development.

## Non Goals

- Simplifying kubernetes production deployment experience. [Kube-deploy](https://github.com/kubernetes/kube-deploy) is attempting to tackle this problem.
- Supporting all possible deployment configurations of Kubernetes like various types of storage, networking, etc.


## Local cluster requirements

- Includes all the master components & DNS (Apiserver, scheduler, controller manager, etcd and kube dns)
- Basic auth
- Service accounts should be setup
- Kubectl should be auto-configured to use the local cluster
- Tested & maintained as part of Kubernetes core

## Existing solutions

Following are some of the existing solutions that attempt to simplify local cluster deployments.

### [Spread](https://github.com/redspread/spread)

Spread's UX is great!
It is adapted from monokube and includes DNS as well.
It satisfies almost all the requirements, excepting that of requiring docker to be pre-installed.
It has a loose dependency on docker.
New releases of docker might break this setup.

### [Kmachine](https://github.com/skippbox/kmachine)

Kmachine is adapted from docker-machine.
It exposes the entire docker-machine CLI.
It is possible to repurpose Kmachine to meet all our requirements.

### [Monokube](https://github.com/polvi/monokube)

Single binary that runs all kube master components.
Does not include DNS.
This is only a part of the overall local cluster solution.

### Vagrant

The kube-up.sh script included in Kubernetes release supports a few Vagrant based local cluster deployments.
kube-up.sh is not user friendly.
It typically takes a long time for the cluster to be set up using vagrant and often times is unsuccessful on OS X.
The [Core OS single machine guide](https://coreos.com/kubernetes/docs/latest/kubernetes-on-vagrant-single.html)  uses Vagrant as well and it just works.
Since we are targeting a single command install/teardown experience, vagrant needs to be an implementation detail and not be exposed to our users.

## Proposed Solution

To avoid exposing users to third party software and external dependencies, we will build a toolbox that will be shipped with all the dependencies including all kubernetes components, hypervisor, base image, kubectl, etc.
*Note: Docker provides a [similar toolbox](https://www.docker.com/products/docker-toolbox).*
This "Localkube" tool will be referred to as "Minikube" in this proposal to avoid ambiguity against Spread's existing ["localkube"](https://github.com/redspread/localkube).
The final name of this tool is TBD. Suggestions are welcome!

Minikube will provide a unified CLI to interact with the local cluster.
The CLI will support only a few operations:
    - **Start** - creates & starts a local cluster along with setting up kubectl & networking (if necessary)
    - **Stop** - suspends the local cluster & preserves cluster state
    - **Delete** - deletes the local cluster completely
    - **Upgrade** - upgrades internal components to the latest available version (upgrades are not guaranteed to preserve cluster state)

For running and managing the kubernetes components themselves,  we can re-use [Spread's localkube](https://github.com/redspread/localkube).
Localkube is a self-contained go binary that includes all the master components including DNS and runs them using multiple go threads.
Each Kubernetes release will include a localkube binary that has been tested exhaustively.

To support Windows and OS X, minikube will use [libmachine](https://github.com/docker/machine/tree/master/libmachine) internally to create and destroy virtual machines.
Minikube will be shipped with an hypervisor (virtualbox) in the case of OS X.
Minikube will include a base image that will be well tested.

In the case of Linux, since the cluster can be run locally, we ideally want to avoid setting up a VM.
Since docker is the only fully supported runtime as of Kubernetes v1.2, we can initially use docker to run and manage localkube.
There is risk of being incompatible with the existing version of docker.
By using a VM, we can avoid such incompatibility issues though.
Feedback from the community will be helpful here.

If the goal is to run outside of a VM, we can have minikube prompt the user if docker is unavailable or version is incompatible.
Alternatives to docker for running the localkube core includes using [rkt](https://coreos.com/rkt/docs/latest/), setting up systemd services, or a System V Init script depending on the distro.

To summarize the pipeline is as follows:

##### OS X / Windows

minikube -> libmachine -> virtualbox/hyper V -> linux VM -> localkube

##### Linux

minikube -> docker -> localkube

### Alternatives considered

#### Bring your own docker

##### Pros

- Kubernetes users will probably already have it
- No extra work for us
- Only one VM/daemon, we can just reuse the existing one

##### Cons

- Not designed to be wrapped, may be unstable
- Might make configuring networking difficult on OS X and Windows
- Versioning and updates will be challenging. We can mitigate some of this with testing at HEAD, but we'll - inevitably hit situations where it's infeasible to work with multiple versions of docker.
- There are lots of different ways to install docker, networking might be challenging if we try to support many paths.

#### Vagrant

##### Pros

- We control the entire experience
- Networking might be easier to build
- Docker can't break us since we'll include a pinned version of Docker
- Easier to support rkt or hyper in the future
- Would let us run some things outside of containers (kubelet, maybe ingress/load balancers)

##### Cons

- More work
- Extra resources (if the user is also running docker-machine)
- Confusing if there are two docker daemons (images built in one can't be run in another)
- Always needs a VM, even on Linux
- Requires installing and possibly understanding Vagrant.

## Releases & Distribution

- Minikube will be released independent of Kubernetes core in order to facilitate fixing of issues that are outside of Kubernetes core.
- The latest version of Minikube is guaranteed to support the latest release of Kubernetes, including documentation.
- The Google Cloud SDK will package minikube and provide utilities for configuring kubectl to use it, but will not in any other way wrap minikube.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/local-cluster-ux.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
