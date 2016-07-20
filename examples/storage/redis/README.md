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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Reliable, Scalable Redis on Kubernetes

The following document describes the deployment of a reliable, multi-node Redis on Kubernetes.  It deploys a master with replicated slaves, as well as replicated redis sentinels which are use for health checking and failover.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](../../../docs/getting-started-guides/) for installation instructions for your platform.

### A note for the impatient

This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Turning up an initial master/sentinel pod.

A [_Pod_](../../../docs/user-guide/pods.md) is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.

We will used the shared network namespace to bootstrap our Redis cluster.  In particular, the very first sentinel needs to know how to find the master (subsequent sentinels just ask the first sentinel).  Because all containers in a Pod share a network namespace, the sentinel can simply look at ```$(hostname -i):6379```.

Here is the config for the initial master and sentinel pod: [redis-master.yaml](redis-master.yaml)


Create this master as follows:

```sh
kubectl create -f examples/storage/redis/redis-master.yaml
```

### Turning up a sentinel service

In Kubernetes a [_Service_](../../../docs/user-guide/services.md) describes a set of Pods that perform the same task.  For example, the set of nodes in a Cassandra cluster, or even the single node we created above.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods (or the single Pod we've already created) available via the Kubernetes API.

In Redis, we will use a Kubernetes Service to provide a discoverable endpoints for the Redis sentinels in the cluster.  From the sentinels Redis clients can find the master, and then the slaves and other relevant info for the cluster.  This enables new members to join the cluster when failures occur.

Here is the definition of the sentinel service: [redis-sentinel-service.yaml](redis-sentinel-service.yaml)

Create this service:

```sh
kubectl create -f examples/storage/redis/redis-sentinel-service.yaml
```

### Turning up replicated redis servers

So far, what we have done is pretty manual, and not very fault-tolerant.  If the ```redis-master``` pod that we previously created is destroyed for some reason (e.g. a machine dying) our Redis service goes away with it.

In Kubernetes a [_Replication Controller_](../../../docs/user-guide/replication-controller.md) is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Replication Controllers will "adopt" existing pods that match their selector query, so let's create a Replication Controller with a single replica to adopt our existing Redis server. Here is the replication controller config: [redis-controller.yaml](redis-controller.yaml)

The bulk of this controller config is actually identical to the redis-master pod definition above.  It forms the template or "cookie cutter" that defines what it means to be a member of this set.

Create this controller:

```sh
kubectl create -f examples/storage/redis/redis-controller.yaml
```

We'll do the same thing for the sentinel.  Here is the controller config: [redis-sentinel-controller.yaml](redis-sentinel-controller.yaml)

We create it as follows:

```sh
kubectl create -f examples/storage/redis/redis-sentinel-controller.yaml
```

### Scale our replicated pods

Initially creating those pods didn't actually do anything, since we only asked for one sentinel and one redis server, and they already existed, nothing changed.  Now we will add more replicas:

```sh
kubectl scale rc redis --replicas=3
```

```sh
kubectl scale rc redis-sentinel --replicas=3
```

This will create two additional replicas of the redis server and two additional replicas of the redis sentinel.

Unlike our original redis-master pod, these pods exist independently, and they use the ```redis-sentinel-service``` that we defined above to discover and join the cluster.

### Delete our manual pod

The final step in the cluster turn up is to delete the original redis-master pod that we created manually.  While it was useful for bootstrapping discovery in the cluster, we really don't want the lifespan of our sentinel to be tied to the lifespan of one of our redis servers, and now that we have a successful, replicated redis sentinel service up and running, the binding is unnecessary.

Delete the master as follows:

```sh
kubectl delete pods redis-master
```

Now let's take a close look at what happens after this pod is deleted.  There are three things that happen:

  1. The redis replication controller notices that its desired state is 3 replicas, but there are currently only 2 replicas, and so it creates a new redis server to bring the replica count back up to 3
  2. The redis-sentinel replication controller likewise notices the missing sentinel, and also creates a new sentinel.
  3. The redis sentinels themselves, realize that the master has disappeared from the cluster, and begin the election procedure for selecting a new master.  They perform this election and selection, and chose one of the existing redis server replicas to be the new master.

### Conclusion

At this point we now have a reliable, scalable Redis installation.  By scaling the replication controller for redis servers, we can increase or decrease the number of read-slaves in our cluster.  Likewise, if failures occur, the redis-sentinels will perform master election and select a new master.

**NOTE:** since redis 3.2 some security measures (bind to 127.0.0.1 and `--protected-mode`) are enabled by default. Please read about this in http://antirez.com/news/96


### tl; dr

For those of you who are impatient, here is the summary of commands we ran in this tutorial:

```
# Create a bootstrap master
kubectl create -f examples/storage/redis/redis-master.yaml

# Create a service to track the sentinels
kubectl create -f examples/storage/redis/redis-sentinel-service.yaml

# Create a replication controller for redis servers
kubectl create -f examples/storage/redis/redis-controller.yaml

# Create a replication controller for redis sentinels
kubectl create -f examples/storage/redis/redis-sentinel-controller.yaml

# Scale both replication controllers
kubectl scale rc redis --replicas=3
kubectl scale rc redis-sentinel --replicas=3

# Delete the original master pod
kubectl delete pods redis-master
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/redis/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
