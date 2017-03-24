## Reliable, Multi-Node Redis on Kubernetes

The following document describes the deployment of a reliable, multi-node Redis on Kubernetes.  It deploys a master with replicated slaves, as well as replicated redis sentinels which are use for health checking and failover.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](../../../docs/getting-started-guides/) for installation instructions for your platform.

### Turning up an initial master/sentinel pod and sentinel service.

A [_Pod_](../../../docs/user-guide/pods.md) is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.

We will use the shared network namespace to bootstrap our Redis cluster.  In particular, the very first sentinel needs to know how to find the master (subsequent sentinels just ask the first sentinel).  Because all containers in a Pod share a network namespace, the sentinel can simply look at ```$(hostname -i):6379```.

In Kubernetes a [_Service_](../../../docs/user-guide/services.md) describes a set of Pods that perform the same task.  For example, the set of nodes in a Cassandra cluster, or even the single node we created above.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods (or the single Pod we've already created) available via the Kubernetes API.

In Redis, we will use a Kubernetes Service to provide a discoverable endpoints for the Redis sentinels in the cluster.  From the sentinels Redis clients can find the master, and then the slaves and other relevant info for the cluster.  This enables new members to join the cluster when failures occur.

Here is the config for the initial master/entinel pod and sentinel service: [redis-bootstrap.yaml](redis-bootstrap.yaml)


Create them as follows:

```sh
kubectl create -f examples/storage/redis/redis-bootstrap.yaml
```

### Turning up replicated redis servers

So far, what we have done is pretty manual, and not very fault-tolerant.  If the ```redis-master``` pod that we previously created is destroyed for some reason (e.g. a machine dying) our Redis service goes away with it.

In Kubernetes a [_Replication Controller_](../../../docs/user-guide/replication-controller.md) is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Let's create two Replication Controllers with 3 replica each for sentinel and redis servers.

The bulk of the controller configs are actually identical to the redis-master pod definition above.  It forms the template or "cookie cutter" that defines what it means to be a member of this set.

Create the controllers:

```sh
kubectl create -f examples/storage/redis/redis-rc.yaml
```

The redis-sentinel Relication Controller will "adopt" the existing master/sentinel pod and create 2 more sentinel replicas. The redis Replication Controller will create 3 replicas of redis server.

Unlike our original redis-master pod, these pods exist independently, and they use the ```redis-sentinel-service``` that we defined above to discover and join the cluster.

After the replicas join the cluster, the redis cluster will have 1 master + 3 slaves + 3 sentinels.

### How client use the redis cluster

To access the redis cluster, the client can get the redis master IP address (the port number is 6379) by below command:

```sh
master=$(redis-cli -h ${REDIS_SENTINEL_SERVICE_HOST} -p ${REDIS_SENTINEL_SERVICE_PORT} --csv SENTINEL get-master-addr-by-name mymaster | tr ',' ' ' | cut -d' ' -f1)
```

### How fault-tolerant works

As a reliable redis cluster, it can automatically recover when any of the pod is down. Now let's take a close look at how this works.

If one of the redis server pod is down:

  1. The redis replication controller notices that it's desired state is 3 replicas, but there are currently only 2 replicas, and so it creates a new redis server to bring the replica count back up to 3
  2. The newly created redis server pod can use the ```redis-sentinel-service``` to get master information and join the cluster as slave.

If one of the sentinel pod is down:

  1. The redis-sentinel replication controller notices that it's desired state is 3 replicas, but there are currently only 2 replicas, and so it creates a new redis sentinel to bring the replica count back up to 3
  2. The newly created sentinel pod can use the ```redis-sentinel-service``` to get master information and join the cluster as sentinel.
  
If the master/sentinel pod is down:
  
  1. The redis-sentinel replication controller notices that it's desired state is 3 replicas, but there are currently only 2 replicas, and so it creates a new sentinel to bring the replica count back up to 3
  2. The newly created sentinel pod can use the ```redis-sentinel-service``` to get master information. But since the master is also down, it cannot connect to the master. In this case, the new sentinel will enter a loop to repeatedly get the master information and try to connect... 
  3. The existing redis sentinels themselves, realize that the master has disappeared from the cluster, and begin the election procedure for selecting a new master.  They perform this election and selection, and choose one of the existing redis server replicas to be the new master.
  4. Once the newly master is selected, the new created sentinel pod can get it's information and joins the cluster as sentinel.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/redis/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
