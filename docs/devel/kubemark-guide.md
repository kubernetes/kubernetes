<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubemark User Guide

## Introduction

Kubemark is a performance testing tool which allows users to run experiments on
simulated clusters. The primary use case is scalability testing, as simulated
clusters can be much bigger than the real ones. The objective is to expose
problems with the master components (API server, controller manager or
scheduler) that appear only on bigger clusters (e.g. small memory leaks).

This document serves as a primer to understand what Kubemark is, what it is not,
and how to use it.

## Architecture

On a very high level Kubemark cluster consists of two parts: real master
components and a set of “Hollow” Nodes. The prefix “Hollow” means an
implementation/instantiation of a component with all “moving” parts mocked out.
The best example is HollowKubelet, which pretends to be an ordinary Kubelet, but
does not start anything, nor mount any volumes - it just lies it does. More
detailed design and implementation details are at the end of this document.

Currently master components run on a dedicated machine(s), and HollowNodes run
on an ‘external’ Kubernetes cluster. This design has a slight advantage, over
running master components on external cluster, of completely isolating master
resources from everything else.

## Requirements

To run Kubemark you need a Kubernetes cluster (called `external cluster`)
for running all your HollowNodes and a dedicated machine for a master.
Master machine has to be directly routable from HollowNodes. You also need an
access to some Docker repository.

Currently scripts are written to be easily usable by GCE, but it should be
relatively straightforward to port them to different providers or bare metal.

## Common use cases and helper scripts

Common workflow for Kubemark is:
- starting a Kubemark cluster (on GCE)
- running e2e tests on Kubemark cluster
- monitoring test execution and debugging problems
- turning down Kubemark cluster

Included in descriptions there will be comments helpful for anyone who’ll want to
port Kubemark to different providers.

### Starting a Kubemark cluster

To start a Kubemark cluster on GCE you need to create an external kubernetes
cluster (it can be GCE, GKE or anything else) by yourself, make sure that kubeconfig
points to it by default, build a kubernetes release (e.g. by running
`make quick-release`) and run `test/kubemark/start-kubemark.sh` script.
This script will create a VM for master components, Pods for HollowNodes
and do all the setup necessary to let them talk to each other. It will use the
configuration stored in `cluster/kubemark/config-default.sh` - you can tweak it
however you want, but note that some features may not be implemented yet, as
implementation of Hollow components/mocks will probably be lagging behind ‘real’
one. For performance tests interesting variables are `NUM_NODES` and
`MASTER_SIZE`. After start-kubemark script is finished you’ll have a ready
Kubemark cluster, a kubeconfig file for talking to the Kubemark cluster is
stored in `test/kubemark/kubeconfig.kubemark`.

Currently we're running HollowNode with limit of 0.05 a CPU core and ~60MB or
memory, which taking into account default cluster addons and fluentD running on
an 'external' cluster, allows running ~17.5 HollowNodes per core.

#### Behind the scene details:

Start-kubemark script does quite a lot of things:

- Creates a master machine called hollow-cluster-master and PD for it (*uses
gcloud, should be easy to do outside of GCE*)

- Creates a firewall rule which opens port 443\* on the master machine (*uses
gcloud, should be easy to do outside of GCE*)

- Builds a Docker image for HollowNode from the current repository and pushes it
to the Docker repository (*GCR for us, using scripts from
`cluster/gce/util.sh` - it may get tricky outside of GCE*)

- Generates certificates and kubeconfig files, writes a kubeconfig locally to
`test/kubemark/kubeconfig.kubemark` and creates a Secret which stores kubeconfig for
HollowKubelet/HollowProxy use (*used gcloud to transfer files to Master, should
be easy to do outside of GCE*).

- Creates a ReplicationController for HollowNodes and starts them up. (*will
work exactly the same everywhere as long as MASTER_IP will be populated
correctly, but you’ll need to update docker image address if you’re not using
GCR and default image name*)

- Waits until all HollowNodes are in the Running phase (*will work exactly the
same everywhere*)

<sub>\* Port 443 is a secured port on the master machine which is used for all
external communication with the API server. In the last sentence *external*
means all traffic coming from other machines, including all the Nodes, not only
from outside of the cluster. Currently local components, i.e. ControllerManager
and Scheduler talk with API server using insecure port 8080.</sub>

### Running e2e tests on Kubemark cluster

To run standard e2e test on your Kubemark cluster created in the previous step
you execute `test/kubemark/run-e2e-tests.sh` script. It will configure ginkgo to
use Kubemark cluster instead of something else and start an e2e test. This
script should not need any changes to work on other cloud providers.

By default (if nothing will be passed to it) the script will run a Density '30
test. If you want to run a different e2e test you just need to provide flags you want to be
passed to `hack/ginkgo-e2e.sh` script, e.g. `--ginkgo.focus="Load"` to run the
Load test.

By default, at the end of each test, it will delete namespaces and everything
under it (e.g. events, replication controllers) on Kubemark master, which takes
a lot of time. Such work aren't needed in most cases: if you delete your
Kubemark cluster after running `run-e2e-tests.sh`; you don't care about
namespace deletion performance, specifically related to etcd; etc. There is a
flag that enables you to avoid namespace deletion: `--delete-namespace=false`.
Adding the flag should let you see in logs: `Found DeleteNamespace=false,
skipping namespace deletion!`

### Monitoring test execution and debugging problems

Run-e2e-tests prints the same output on Kubemark as on ordinary e2e cluster, but
if you need to dig deeper you need to learn how to debug HollowNodes and how
Master machine (currently) differs from the ordinary one.

If you need to debug master machine you can do similar things as you do on your
ordinary master. The difference between Kubemark setup and ordinary setup is
that in Kubemark etcd is run as a plain docker container, and all master
components are run as normal processes. There’s no Kubelet overseeing them. Logs
are stored in exactly the same place, i.e. `/var/logs/` directory. Because
binaries are not supervised by anything they won't be restarted in the case of a
crash.

To help you with debugging from inside the cluster startup script puts a
`~/configure-kubectl.sh` script on the master. It downloads `gcloud` and
`kubectl` tool and configures kubectl to work on unsecured master port (useful
if there are problems with security). After the script is run you can use
kubectl command from the master machine to play with the cluster.

Debugging HollowNodes is a bit more tricky, as if you experience a problem on
one of them you need to learn which hollow-node pod corresponds to a given
HollowNode known by the Master. During self-registeration HollowNodes provide
their cluster IPs as Names, which means that if you need to find a HollowNode
named `10.2.4.5` you just need to find a Pod in external cluster with this
cluster IP. There’s a helper script
`test/kubemark/get-real-pod-for-hollow-node.sh` that does this for you.

When you have a Pod name you can use `kubectl logs` on external cluster to get
logs, or use a `kubectl describe pod` call to find an external Node on which
this particular HollowNode is running so you can ssh to it.

E.g. you want to see the logs of HollowKubelet on which pod `my-pod` is running.
To do so you can execute:

```
$ kubectl kubernetes/test/kubemark/kubeconfig.kubemark describe pod my-pod
```

Which outputs pod description and among it a line:

```
Node:				1.2.3.4/1.2.3.4
```

To learn the `hollow-node` pod corresponding to node `1.2.3.4` you use
aforementioned script:

```
$ kubernetes/test/kubemark/get-real-pod-for-hollow-node.sh 1.2.3.4
```

which will output the line:

```
hollow-node-1234
```

Now you just use ordinary kubectl command to get the logs:

```
kubectl --namespace=kubemark logs hollow-node-1234
```

All those things should work exactly the same on all cloud providers.

### Turning down Kubemark cluster

On GCE you just need to execute `test/kubemark/stop-kubemark.sh` script, which
will delete HollowNode ReplicationController and all the resources for you. On
other providers you’ll need to delete all this stuff by yourself.

## Some current implementation details

Kubemark master uses exactly the same binaries as ordinary Kubernetes does. This
means that it will never be out of date. On the other hand HollowNodes use
existing fake for Kubelet (called SimpleKubelet), which mocks its runtime
manager with `pkg/kubelet/dockertools/fake_manager.go`, where most logic sits.
Because there’s no easy way of mocking other managers (e.g. VolumeManager), they
are not supported in Kubemark (e.g. we can’t schedule Pods with volumes in them
yet).

As the time passes more fakes will probably be plugged into HollowNodes, but
it’s crucial to make it as simple as possible to allow running a big number of
Hollows on a single core.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/kubemark-guide.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
