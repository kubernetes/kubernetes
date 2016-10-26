
## Background

We have a goal to be able to scale to 1000-node clusters by end of 2015.
As a result, we need to be able to run some kind of regression tests and deliver
a mechanism so that developers can test their changes with respect to performance.

Ideally, we would like to run performance tests also on PRs - although it might
be impossible to run them on every single PR, we may introduce a possibility for
a reviewer to trigger them if the change has non obvious impact on the performance
(something like "k8s-bot run scalability tests please" should be feasible).

However, running performance tests on 1000-node clusters (or even bigger in the
future is) is a non-starter. Thus, we need some more sophisticated infrastructure
to simulate big clusters on relatively small number of machines and/or cores.

This document describes two approaches to tackling this problem.
Once we have a better understanding of their consequences, we may want to
decide to drop one of them, but we are not yet in that position.


## Proposal 1 - Kubmark

In this proposal we are focusing on scalability testing of master components.
We do NOT focus on node-scalability - this issue should be handled separately.

Since we do not focus on the node performance, we don't need real Kubelet nor
KubeProxy - in fact we don't even need to start real containers.
All we actually need is to have some Kubelet-like and KubeProxy-like components
that will be simulating the load on apiserver that their real equivalents are
generating (e.g. sending NodeStatus updated, watching for pods, watching for
endpoints (KubeProxy), etc.).

What needs to be done:

1. Determine what requests both KubeProxy and Kubelet are sending to apiserver.
2. Create a KubeletSim that is generating the same load on apiserver that the
   real Kubelet, but is not starting any containers. In the initial version we
   can assume that pods never die, so it is enough to just react on the state
   changes read from apiserver.
	 TBD: Maybe we can reuse a real Kubelet for it by just injecting some "fake"
   interfaces to it?
3. Similarly create a KubeProxySim that is generating the same load on apiserver
   as a real KubeProxy. Again, since we are not planning to talk to those
   containers, it basically doesn't need to do anything apart from that.
	 TBD: Maybe we can reuse a real KubeProxy for it by just injecting some "fake"
   interfaces to it?
4. Refactor kube-up/kube-down scripts (or create new ones) to allow starting
   a cluster with KubeletSim and KubeProxySim instead of real ones and put
   a bunch of them on a single machine.
5. Create a load generator for it (probably initially it would be enough to
   reuse tests that we use in gce-scalability suite).


## Proposal 2 - Oversubscribing

The other method we are proposing is to oversubscribe the resource,
or in essence enable a single node to look like many separate nodes even though
they reside on a single host. This is a well established pattern in many different
cluster managers (for more details see
http://www.uscms.org/SoftwareComputing/Grid/WMS/glideinWMS/doc.prd/index.html ).
There are a couple of different ways to accomplish this, but the most viable method
is to run privileged kubelet pods under a hosts kubelet process. These pods then
register back with the master via the introspective service using modified names
as not to collide.

Complications may currently exist around container tracking and ownership in docker.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/scalability-testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
