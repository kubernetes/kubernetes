<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/examples/redis-cluster/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Reliable Redis Cluster 3.0 on Kubernetes

The following document describes the deployment of a reliable Redis 3.0 Cluster on Kubernetes. It deploys 6 redis nodes to form redis cluster consisting of 3 masters and 3 slaves.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](../../docs/getting-started-guides/) for installation instructions for your platform.

Presented here approach works only when Kubernetes cluster operates in [iptables](https://github.com/kubernetes/kubernetes/issues/3760) mode (eg. by running `kube-proxy` with option `--proxy-mode=iptables`).
When working with [Google Container Engine](https://cloud.google.com/container-engine/) this mode can be set by running:

```sh
examples/redis-cluster/scripts/set-gke-k8s-proxy-mode.sh iptables
```

If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Deployment architecture

![Redis 3.0 Cluster at Kubernetes - deployment architecture](doc/redis-cluster.png?raw=true "Redis 3.0 Cluster at Kubernetes - deployment architecture")

### Deployment description

##### Used docker image of redis node

In this example there is used docker image: [balon/docker-redis:3.0.7](https://hub.docker.com/r/balon/docker-redis/). Dockerfile is available as well in [image](image/) subfolder of this example. This image is based on official [redis:3.0.7](https://hub.docker.com/_/redis/) image and adds on top of it possibility to configure it in [12-factor](http://12factor.net/) way via environment variables (`CLUSTER_CONFIG`, `REDIS_CONFIG`).

##### Preconfigured, unique redis node IDs

Each separate Redis Cluster that is running in the same (multi-tenant) Kubernetes cluster - needs to have Redis Node ID's which are:

- constant during Redis cluster lifecycle
- unique within Kubernetes cluster
- preconfigured and known up-front

Suggested format for Redis Node ID: `[K8S-NAMESPACE-NAME]--------redis-node-[XXX]`, full Redis Node ID needs to be String with length = 40 characters.

Example:

```
redis-cluster-001---------redis-node-101
redis-cluster-001---------redis-node-102
redis-cluster-001---------redis-node-201
redis-cluster-001---------redis-node-202
redis-cluster-001---------redis-node-301
redis-cluster-001---------redis-node-302
```

##### Redis cluster topology

Presented Redis cluster represents minimal reliable topology: 3 masters + 3 slaves.

Each redis node needs to have pre-provisioned `nodes.conf` file which includes:
- initial redis cluster topology
- redis node IDs
- addresses of other nodes

`nodes.conf` file can be provisioned in 12-factor way via `CLUSTER_CONFIG` multi-line environment variable. In this initial cluster configuration there is need to provide IP:port for all other redis cluster nodes. As this information is not known prior to creation of services & pod's - it can be parametrized by Kubernetes standard [service environment variables](../../docs/user-guide/services.md#environment-variables) `{SVCNAME}_SERVICE_HOST` and `{SVCNAME}_SERVICE_PORT`. Each node in `nodes.conf` is referenced by IP:port of corresponding Kubernetes service.

All IP's through which nodes are referenced in `nodes.conf` - will be automatically resolved to particular pod IPs after cluster is started and nodes establish connections with each other. Because Redis cluster nodes base on source address of peer node in incoming connections. This is reason why kube-proxy needs to operate in iptables mode. This behaviour will be modified in Redis 3.2 via introduction of `announce-ip` & `announce-port` config properties.

Each node in its initial `nodes.conf` is configured as `myself,master` for particular slot (aka. shard). Rest of nodes are referenced as slaves. Particular roles will be negotiated and agreed between connected nodes after cluster is started and formed.

All timestamps & epoch are configured as `0` in initial `nodes.conf`

In presented Redis cluster architecture - there is no difference between first initial start of nodes to form a cluster and eg. starting of single node after failure. Redis nodes can start in any order.

##### Kubernetes objects representing cluster

Presented example assumes that each Redis cluster node:
- is configured as single container within separate Pod template in separate ReplicationController (`replicas: 1`).
- has separate dedicated Service pointing to its Pod

All Services must be created before ReplicationControllers.

### Possible issues & problems

- only iptables proxy-mode is supported (userspace not supported)
- because initially each node starts as master, in case of very specific network partitioning there can be formed two separate clusters (3 masters vs 3 masters). Such case has very low probability to occur (but theoretically it is possible)
- after nodes connect to each other (initially via Service ip:port) they refresh their knowledge about cluster topology and reference each other via direct ip:port (of Pod not Service). This is not an issue, but not very clean solution. This behaviour will be modified in Redis 3.2 via introduction of `announce-ip` and `announce-port` config properties.
- to setup initial cluster topology there is provided initial `nodes.conf` which is in internal Redis format. This format is likely to be changed in further versions of Redis.

### Contributors

- [@balonus](https://github.com/balonus) - Michal Balinski <michal.balinski at gmail.com>
- [@widgetpl](https://github.com/widgetpl) - Michal Dziedziela <michal.dziedziela at gmail.com>

### tl; dr

For those of you who are impatient, here is the summary of commands we ran in this tutorial:

###### Create a redis cluster

```sh
kubectl create -f examples/redis-cluster/redis-cluster.yaml
```

```sh
service "redis-001-node-101" created
service "redis-001-node-102" created
service "redis-001-node-201" created
service "redis-001-node-202" created
service "redis-001-node-301" created
service "redis-001-node-302" created
replicationcontroller "redis-001-node-101" created
replicationcontroller "redis-001-node-102" created
replicationcontroller "redis-001-node-201" created
replicationcontroller "redis-001-node-202" created
replicationcontroller "redis-001-node-301" created
replicationcontroller "redis-001-node-302" created
```

###### Check redis cluster topology & status

```sh
examples/redis-cluster/scripts/redis-cluster-info.sh
```

```sh

======================================
Pod: redis-001-node-101-si1sb
redis-cluster-001---------redis-node-101 10.0.1.45:6379 myself,slave redis-cluster-001---------redis-node-102 0 0 3 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 master - 0 1457201607622 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 slave redis-cluster-001---------redis-node-202 0 1457201607523 1 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 master - 0 1457201607422 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 master - 0 1457201607523 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 slave redis-cluster-001---------redis-node-301 0 1457201607422 2 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:4
cluster_stats_messages_sent:3284
cluster_stats_messages_received:3241
======================================
Pod: redis-001-node-102-xpiap
redis-cluster-001---------redis-node-101 10.0.1.45:6379 slave redis-cluster-001---------redis-node-102 0 1457201611101 4 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 myself,master - 0 0 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 slave redis-cluster-001---------redis-node-202 0 1457201611302 1 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 master - 0 1457201611201 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 master - 0 1457201611101 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 slave redis-cluster-001---------redis-node-301 0 1457201610898 2 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:4
cluster_stats_messages_sent:3360
cluster_stats_messages_received:3315
======================================
Pod: redis-001-node-201-6e5ln
redis-cluster-001---------redis-node-101 10.0.1.45:6379 slave redis-cluster-001---------redis-node-102 0 1457201614638 4 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 master - 0 1457201614537 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 myself,slave redis-cluster-001---------redis-node-202 0 0 0 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 master - 0 1457201614638 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 master - 0 1457201614638 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 slave redis-cluster-001---------redis-node-301 0 1457201614739 2 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:1
cluster_stats_messages_sent:3392
cluster_stats_messages_received:3366
======================================
Pod: redis-001-node-202-z6pnz
redis-cluster-001---------redis-node-101 10.0.1.45:6379 slave redis-cluster-001---------redis-node-102 0 1457201618029 4 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 master - 0 1457201618129 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 slave redis-cluster-001---------redis-node-202 0 1457201618129 1 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 myself,master - 0 0 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 master - 0 1457201618234 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 slave redis-cluster-001---------redis-node-301 0 1457201618328 2 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:1
cluster_stats_messages_sent:3435
cluster_stats_messages_received:3403
======================================
Pod: redis-001-node-301-lro34
redis-cluster-001---------redis-node-101 10.0.1.45:6379 slave redis-cluster-001---------redis-node-102 0 1457201621648 4 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 master - 0 1457201621548 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 slave redis-cluster-001---------redis-node-202 0 1457201621952 1 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 master - 0 1457201621452 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 myself,master - 0 0 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 slave redis-cluster-001---------redis-node-301 0 1457201621853 2 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:2
cluster_stats_messages_sent:3548
cluster_stats_messages_received:3511
======================================
Pod: redis-001-node-302-4xt64
redis-cluster-001---------redis-node-101 10.0.1.45:6379 slave redis-cluster-001---------redis-node-102 0 1457201625475 4 connected
redis-cluster-001---------redis-node-102 10.0.1.46:6379 master - 0 1457201625475 4 connected 0-5460
redis-cluster-001---------redis-node-201 10.0.1.47:6379 slave redis-cluster-001---------redis-node-202 0 1457201625475 1 connected
redis-cluster-001---------redis-node-202 10.0.2.25:6379 master - 0 1457201625073 1 connected 5461-10922
redis-cluster-001---------redis-node-301 10.0.1.48:6379 master - 0 1457201625475 2 connected 10923-16383
redis-cluster-001---------redis-node-302 10.0.2.26:6379 myself,slave redis-cluster-001---------redis-node-301 0 0 0 connected
cluster_state:ok
cluster_slots_assigned:16384
cluster_slots_ok:16384
cluster_slots_pfail:0
cluster_slots_fail:0
cluster_known_nodes:6
cluster_size:3
cluster_current_epoch:4
cluster_my_epoch:2
cluster_stats_messages_sent:3585
cluster_stats_messages_received:3554
```

###### Check redis cluster logs

```sh
examples/redis-cluster/scripts/redis-cluster-logs.sh
```

###### Delete redis cluster

```sh
kubectl delete -f examples/redis-cluster/redis-cluster.yaml
```



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/redis-cluster/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->