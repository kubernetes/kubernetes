# Kubemark proposal

## Goal of this document

This document describes a design of Kubemark - a system that allows performance testing of a Kubernetes cluster. It describes the
assumption, high level design and discusses possible solutions for lower-level problems. It is supposed to be a starting point for more
detailed discussion.

## Current state and objective

Currently performance testing happens on ‘live’ clusters of up to 100 Nodes. It takes quite a while to start such cluster or to push
updates to all Nodes, and it uses quite a lot of resources. At this scale the amount of wasted time and used resources is still acceptable.
In the next quarter or two we’re targeting 1000 Node cluster, which will push it way beyond ‘acceptable’ level. Additionally we want to
enable people without many resources to run scalability tests on bigger clusters than they can afford at given time. Having an ability to
cheaply run scalability tests will enable us to run some set of them on "normal" test clusters, which in turn would mean ability to run
them on every PR.

This means that we need a system that will allow for realistic performance testing on (much) smaller number of “real” machines. First
assumption we make is that Nodes are independent, i.e. number of existing Nodes do not impact performance of a single Node. This is not
entirely true, as number of Nodes can increase latency of various components on Master machine, which in turn may increase latency of Node
operations, but we’re not interested in measuring this effect here. Instead we want to measure how number of Nodes and the load imposed by
Node daemons affects the performance of Master components.

## Kubemark architecture overview

The high-level idea behind Kubemark is to write library that allows running artificial "Hollow" Nodes that will be able to simulate a
behavior of real Kubelet and KubeProxy in a single, lightweight binary. Hollow components will need to correctly respond to Controllers
(via API server), and preferably, in the fullness of time, be able to ‘replay’ previously recorded real traffic (this is out of scope for
initial version). To teach Hollow components replaying recorded traffic they will need to store data specifying when given Pod/Container
should die (e.g. observed lifetime). Such data can be extracted e.g. from etcd Raft logs, or it can be reconstructed from Events. In the
initial version we only want them to be able to fool Master components and put some configurable (in what way TBD) load on them.

When we have Hollow Node ready, we’ll be able to test performance of Master Components by creating a real Master Node, with API server,
Controllers, etcd and whatnot, and create number of Hollow Nodes that will register to the running Master.

To make Kubemark easier to maintain when system evolves Hollow components will reuse real "production" code for Kubelet and KubeProxy, but
will mock all the backends with no-op or very simple mocks. We believe that this approach is better in the long run than writing special
"performance-test-aimed" separate version of them. This may take more time to create an initial version, but we think maintenance cost will
be noticeably smaller.

### Option 1

For the initial version we will teach Master components to use port number to identify Kubelet/KubeProxy. This will allow running those
components on non-default ports, and in the same time will allow to run multiple Hollow Nodes on a single machine. During setup we will
generate credentials for cluster communication and pass them to HollowKubelet/HollowProxy to use. Master will treat all HollowNodes as
normal ones.

![Kubmark architecture diagram for option 1](Kubemark_architecture.png?raw=true "Kubemark architecture overview")
*Kubmark architecture diagram for option 1*

### Option 2

As a second (equivalent) option we will run Kubemark on top of 'real' Kubernetes cluster, where both Master and Hollow Nodes will be Pods.
In this option we'll be able to use Kubernetes mechanisms to streamline setup, e.g. by using Kubernetes networking to ensure unique IPs for
Hollow Nodes, or using Secrets to distribute Kubelet credentials. The downside of this configuration is that it's likely that some noise
will appear in Kubemark results from either CPU/Memory pressure from other things running on Nodes (e.g. FluentD, or Kubelet) or running
cluster over an overlay network. We believe that it'll be possible to turn off cluster monitoring for Kubemark runs, so that the impact
of real Node daemons will be minimized, but we don't know what will be the impact of using higher level networking stack. Running a
comparison will be an interesting test in itself.

### Discussion

Before taking a closer look at steps necessary to set up a minimal Hollow cluster it's hard to tell which approach will be simpler. It's
quite possible that the initial version will end up as hybrid between running the Hollow cluster directly on top of VMs and running the
Hollow cluster on top of a Kubernetes cluster that is running on top of VMs. E.g. running Nodes as Pods in Kubernetes cluster and Master
directly on top of VM.

## Things to simulate

In real Kubernetes on a single Node we run two daemons that communicate with Master in some way: Kubelet and KubeProxy.

### KubeProxy

As a replacement for KubeProxy we'll use HollowProxy, which will be a real KubeProxy with injected no-op mocks everywhere it makes sense.

### Kubelet

As a replacement for Kubelet we'll use HollowKubelet, which will be a real Kubelet with injected no-op or simple mocks everywhere it makes
sense.

Kubelet also exposes cadvisor endpoint which is scraped by Heapster, healthz to be read by supervisord, and we have FluentD running as a
Pod on each Node that exports logs to Elasticsearch (or Google Cloud Logging). Both Heapster and Elasticsearch are running in Pods in the
cluster so do not add any load on a Master components by themselves. There can be other systems that scrape Heapster through proxy running
on Master, which adds additional load, but they're not the part of default setup, so in the first version we won't simulate this behavior.

In the first version we’ll assume that all started Pods will run indefinitely if not explicitly deleted. In the future we can add a model
of short-running batch jobs, but in the initial version we’ll assume only serving-like Pods.

### Heapster

In addition to system components we run Heapster as a part of cluster monitoring setup. Heapster currently watches Events, Pods and Nodes
through the API server. In the test setup we can use real heapster for watching API server, with mocked out piece that scrapes cAdvisor
data from Kubelets.

### Elasticsearch and Fluentd

Similarly to Heapster Elasticsearch runs outside the Master machine but generates some traffic on it. Fluentd “daemon” running on Master
periodically sends Docker logs it gathered to the Elasticsearch running on one of the Nodes. In the initial version we omit Elasticsearch,
as it produces only a constant small load on Master Node that does not change with the size of the cluster.

## Necessary work

There are three more or less independent things that needs to be worked on:
- HollowNode implementation, creating a library/binary that will be able to listen to Watches and respond in a correct fashion with Status
updates. This also involves creation of a CloudProvider that can produce such Hollow Nodes, or making sure that HollowNodes can correctly
self-register in no-provider Master.
- Kubemark setup, including figuring networking model, number of Hollow Nodes that will be allowed to run on a single “machine”, writing
setup/run/teardown scripts (in [option 1](#option-1)), or figuring out how to run Master and Hollow Nodes on top of Kubernetes
(in [option 2](#option-2))
- Creating a Player component that will send requests to the API server putting a load on a cluster. This involves creating a way to
specify desired workload. This task is
very well isolated from the rest, as it is about sending requests to the real API server. Because of that we can discuss requirements
separately.

## Concerns

Network performance most likely won't be a problem for the initial version if running on directly on VMs rather than on top of a Kubernetes
cluster, as Kubemark will be running on standard networking stack (no cloud-provider software routes, or overlay network is needed, as we
don't need custom routing between Pods). Similarly we don't think that running Kubemark on Kubernetes virtualized cluster networking will
cause noticeable performance impact, but it requires testing.

On the other hand when adding additional features it may turn out that we need to simulate Kubernetes Pod network. In such, when running
'pure' Kubemark we may try one of the following:
  - running overlay network like Flannel or OVS instead of using cloud providers routes,
  - write simple network multiplexer to multiplex communications from the Hollow Kubelets/KubeProxies on the machine.

In case of Kubemark on Kubernetes it may turn that we run into a problem with adding yet another layer of network virtualization, but we
don't need to solve this problem now.

## Work plan

- Teach/make sure that Master can talk to multiple Kubelets on the same Machine [option 1](#option-1):
  - make sure that Master can talk to a Kubelet on non-default port,
  - make sure that Master can talk to all Kubelets on different ports,
- Write HollowNode library:
  - new HollowProxy,
  - new HollowKubelet,
  - new HollowNode combining the two,
  - make sure that Master can talk to two HollowKubelets running on the same machine
- Make sure that we can run Hollow cluster on top of Kubernetes [option 2](#option-2)
- Write a player that will automatically put some predefined load on Master, <- this is the moment when it’s possible to play with it and is useful by itself for
scalability tests. Alternatively we can just use current density/load tests,
- Benchmark our machines - see how many Watch clients we can have before everything explodes,
- See how many HollowNodes we can run on a single machine by attaching them to the real master <- this is the moment it starts to useful
- Update kube-up/kube-down scripts to enable creating “HollowClusters”/write a new scripts/something, integrate HollowCluster with a Elasticsearch/Heapster equivalents,
- Allow passing custom configuration to the Player

## Future work

In the future we want to add following capabilities to the Kubemark system:
- replaying real traffic reconstructed from the recorded Events stream,
- simulating scraping things running on Nodes through Master proxy.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/kubemark.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
