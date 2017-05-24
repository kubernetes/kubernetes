# Introduction

Kubernetes provides a default DNS service known as `kube-dns`. The amount of its
backend servers is assigned based on the intitial estimation. This proposal proposes
a methodology to horizontally auto-scale DNS service based on real-time cluster status.

## Motivation

Current DNS service deployment in kubernetes is not comprehensive and already caused
undesired issues:
- Number of backend servers is unmanaged. When DNS request's QPS increases as the
cluster size grows up, DNS workload will possbly surppass the capacity of the initial
backend servers. As the opposite, it would be a big waste of resource if the actual
DNS requests go way below than the initial capacity.
- Users may scale the DNS servers on their behalves. But the number of the backend
servers may not be maintained during a cluster upgrade, which may affect other running
applications that depend on this DNS service.

# Design

## Overview

The solution is to create a standalone pod called `cluster-proportional-autoscaler`,
which will discover the target object (could be one of ReplicationController, Deployment
or ReplicaSet) and scale the Replica parameter based on current cluster status. The
ultimate purpose of this autoscaler is to support the DNS horizontal auto-scaling feature
that based on DNS specific metrics, but current implementation only monitors numbers of
schedulable nodes and cores in the cluster.

## Cluster-proportional-autoscaler implementation

Details of this cluster-proportional-autoscaler implementation could be found on
[kubernetes-incubator/cluster-proportional-autoscaler](
https://github.com/kubernetes-incubator/cluster-proportional-autoscaler).
Bullet points as below:
- An autoscaler pod runs a Kubernetes Golang API client to connect to the Apiserver and
polls for the number of nodes and cores in the cluster.
- A desired replica count would be calculated and applied to the target object based on
current schedulable nodes/cores and the given scaling parameters.
- The scaling parameters and data points are provided via a ConfigMap to the autoscaler
 and it refreshes its parameters table every poll interval to be up to date with the
 latest desired scaling parameters.
- On-the-fly changes of the scaling parameters are allowed without rebuilding or
restarting the autoscaler pod.
- The autoscaler provides a controller interface to support multiple control patterns.
Current supported control patterns are `linear` and `ladder`. More comprehensive control
patterns that consider custom metrics may be developed in the future.

## Integration of kube-dns and cluster-proportional-autoscaler

The plan for the integrating `kube-dns` and `cluster-proportional-autoscaler` is to deploy
the autoscaler with `Deployment` as a kubernetes addon. A set of default scaling parameters
and scaling target `kube-dns` will be passed in when the autoscaler is created. This
autoscaler addon could be turned on/off via the enviroment variable `ENABLE_DNS_HORIZONTAL_AUTOSCALER`
during startup(following the same fashion as the other addons). It could not be enabled when
`kube-dns` itself is disabled. Details of this implementation could be found on pull request
[#33239](https://github.com/kubernetes/kubernetes/pull/33239).

The default scaling parameters are recommended instead of mandatory. Users will be able
to tune the scaling params through editing the corresponding ConfigMap object. Default
parameters would be created only if there is no existing DNS scaling parameters. Current
default scaling parameters use the `linear` control pattern and define three fields:
`coresPerReplica`, `nodesPerReplica`, and `min`(please see [autoscaler README](
https://github.com/kubernetes-incubator/cluster-proportional-autoscaler/blob/master/README.md) for detail).
The exact values for these three fields are set based on a rough estimation. The whole control
pattern will be improved in the future.

Expect DNS horizontal autoscaling to recover under below disorder scenarios:
- Autoscaler Deployment got deleted. Will be re-created by the [`Addon Manager`](
https://github.com/kubernetes/kubernetes/tree/master/cluster/addons/addon-manager).
- Autoscaler Pod got deleted or evicted. Will be re-created by the `Deployment`.
- ConfigMap that stores scaling parameters got deleted. Will be re-created by the
`cluster-proportional-autoscaler`

## Behavior Changes expected

With this DNS horizontal autoscaling feature enabled, all manual operations that scale
`kube-dns` will not work as expected because the autoscaler pod assumes ownership of
`kube-dns` replicas count.

## Remedy solutions

If the DNS horizontal autoscaling feature is enable initially but the users want to turn
it off after, below are the remedy solutions:
- With the write access permission to the master node:
    - Remove the corresponding manifest file for this autoscaler. The autoscaler Deployment
    and the underlying pod will be deleted.
- Without the write access permission to the master node:
    - Change the replicas count of the autoscaler Deployment to `0` will stop this feature.

## Testing

On [kubernetes-incubator/cluster-proportional-autoscaler](
https://github.com/kubernetes-incubator/cluster-proportional-autoscaler), basic unit tests
covered most of the functionalities on the autoscaler aspects. There is [a specific mocking
test](
https://github.com/kubernetes-incubator/cluster-proportional-autoscaler/blob/master/pkg/autoscaler/autoscaler_test.go)
that mockes the Apiserver's behavior. The complete autoscaling functionality is
examined with this mocked Apiserver through different scenarios including cluster size changed
and ConfigMap parameters changed.

Beside these unit tests, there are a few e2e tests(being added in pull
request [#33239](https://github.com/kubernetes/kubernetes/pull/33239)) on the kubernetes
aspect that against a real cluster. It covers cases like cluster size changed, parameters
changed, ConfigMap got deleted, autoscaler pod got deleted, etc.

# Future work

As mention before, one growing direction for this DNS horizontal autoscaling feature is to
scale `kube-dns` based on DNS specific metrics. The current implementation, which utilizes
number of nodes and cores, is not practical enough.

On another aspect, this functionality seems to be a fit for custom metric case in [Horizontal
Pod Autoscaler](http://kubernetes.io/docs/user-guide/horizontal-pod-autoscaling/). We may
consider embrace this Custom Metric feature for DNS horizontal autoscaling in the future,
giving that it may have lower maintenance overhead and well defined configuration. Whether
to use and when to combine with the HPA feature depends on how the real implementations of
[Custom Metrics API](https://github.com/kubernetes/kubernetes/pull/34586) and [Hotizontal
Pod Autoscaler Version 2](https://github.com/kubernetes/kubernetes/pull/34754) go.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/dns-horizontal-autoscaling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
