# Hierarchical Cloud Providers

## Related Discussions

- [Kubernetes issue #357][5]

## Background

Operators have the opportunity to drive up utilization by running a
multi-tenant cluster; that is, to share a cluster's resources between
Kubernetes and other workloads such as lower-priority batch jobs.

The Kubernetes cloud provider API defines an abstraction for groups of
hosts, and is designed to sit between Kubernetes and a source of compute
resources such as a hosting provider.  However, there may be multiple
layers of management between Kubernetes and those hosts.  Flattening
these multiple axes of choice complicates a complete representation of
the underlying layers.  From [experience][6] gained when integrating
Kubernetes with Apache Mesos, the easiest way forward is to represent
the "top" management layer, potentially yielding suboptimal
implementations that lack valuable information about underlying layers.

## Motivation

One intermediate control layer is that of the cluster resource manager.
These resource accounting systems can help address the multi-tenancy
problem, and can typically run on hosts obtained from any source.

For example, Apache Mesos can run on top of virtual machines purchased
from Google Cloud Platform.  In this situation, the operator would
like to use the membership of the Mesos cluster as a source of truth
without losing the high-fidelity information about the underlying
resources (e.g. failure domains, inter-node latency guarantees, etc.)

Besides [Kubernetes on Mesos][1], other integration projects that could
benefit from explicit support are [Kubernetes on YARN][2] and Kubernetes
on [OpenStack][3].

## Approaches

One relatively low-impact way to accomplish this could be to formalize
the notion of hierarchical cloud provider layers.  In the case of the
example given above, the Mesos cloud provider layer could wrap a Google
Cloud Platform cloud provider instance, delegating certain functions to
the underlying provider and overriding others.

Some questions that come to mind are:

1. Currently each cloud provider implementation must register itself with
   a unique name.  How might this work with a hierarchy of providers?

1. How should hierarchical providers be bootstrapped?

Another approach is to redesign the cloud provider API, or provide
a separate API to accomodate these use cases, as discussed briefly in
[#2770][4]

[1]: http://github.com/mesosphere/kubernetes-mesos                                        "Kubernetes-Mesos"
[2]: http://github.com/hortonworks/kubernetes-yarn                                        "Kubernetes-Yarn"
[3]: https://www.openstack.org                                                            "Openstack"
[4]: https://github.com/GoogleCloudPlatform/kubernetes/issues/2770                        "Issue: Create first-class cloudprovider API"
[5]: https://github.com/GoogleCloudPlatform/kubernetes/issues/357                         "Issue: Cleanly split core services and schedulers"
[6]: https://github.com/mesosphere/kubernetes-mesos/blob/v0.4.1/pkg/cloud/mesos/mesos.go  "Kubernetes-Mesos cloud provider implementation"
