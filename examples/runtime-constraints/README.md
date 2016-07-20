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
[here](http://releases.k8s.io/release-1.3/examples/runtime-constraints/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Runtime Constraints example

This example demonstrates how Kubernetes enforces runtime constraints for compute resources.

### Prerequisites

For the purpose of this example, we will spin up a 1 node cluster using the Vagrant provider that
is not running with any additional add-ons that consume node resources.  This keeps our demonstration
of compute resources easier to follow by starting with an empty cluster.

```
$ export KUBERNETES_PROVIDER=vagrant
$ export NUM_NODES=1
$ export KUBE_ENABLE_CLUSTER_MONITORING=none
$ export KUBE_ENABLE_CLUSTER_DNS=false
$ export KUBE_ENABLE_CLUSTER_UI=false
$ cluster/kube-up.sh
```

We should now have a single node cluster running 0 pods.

```
$ cluster/kubectl.sh get nodes
NAME         LABELS                              STATUS    AGE
10.245.1.3   kubernetes.io/hostname=10.245.1.3   Ready     17m
$ cluster/kubectl.sh get pods --all-namespaces
```

When demonstrating runtime constraints, it's useful to show what happens when a node is under heavy load.  For
this scenario, we have a single node with 2 cpus and 1GB of memory to demonstrate behavior under load, but the
results extend to multi-node scenarios.

### CPU requests

Each container in a pod may specify the amount of CPU it requests on a node.  CPU requests are used at schedule time, and represent a minimum amount of CPU that should be reserved for your container to run.

When executing your container, the Kubelet maps your containers CPU requests to CFS shares in the Linux kernel.  CFS CPU shares do not impose a ceiling on the actual amount of CPU the container can use.  Instead, it defines a relative weight across all containers on the system for how much CPU time the container should get if there is CPU contention.

Let's demonstrate this concept using a simple container that will consume as much CPU as possible.

```
$ cluster/kubectl.sh run cpuhog \
    --image=busybox \
    --requests=cpu=100m \
    -- md5sum /dev/urandom
```

This will create a single pod on your minion that requests 1/10 of a CPU, but it has no limit on how much CPU it may actually consume
on the node.

To demonstrate this, if you SSH into your machine, you will see it is consuming as much CPU as possible on the node.

```
$ vagrant ssh minion-1
$ sudo docker stats $(sudo docker ps -q)
CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
6b593b1a9658        0.00%               1.425 MB/1.042 GB   0.14%               1.038 kB/738 B
ae8ae4ffcfe4        150.06%             831.5 kB/1.042 GB   0.08%               0 B/0 B
```

As you can see, its consuming 150% of the total CPU.

If we scale our replication controller to 20 pods, we should see that each container is given an equal proportion of CPU time.

```
$ cluster/kubectl.sh scale rc/cpuhog --replicas=20
```

Once all the pods are running, you will see on your node that each container is getting approximately an equal proportion of CPU time.

```
$ sudo docker stats $(sudo docker ps -q)
CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
089e2d061dee        9.24%               786.4 kB/1.042 GB   0.08%               0 B/0 B
0be33d6e8ddb        10.48%              823.3 kB/1.042 GB   0.08%               0 B/0 B
0f4e3c4a93e0        10.43%              786.4 kB/1.042 GB   0.08%               0 B/0 B
```

Each container is getting 10% of the CPU time per their scheduling request, and we are unable to schedule more.

As you can see CPU requests are used to schedule pods to the node in a manner that provides weighted distribution of CPU time
when under contention.  If the node is not being actively consumed by other containers, a container is able to burst up to as much
available CPU time as possible.  If there is contention for CPU, CPU time is shared based on the requested value.

Let's delete all existing resources in preparation for the next scenario.  Verify all the pods are deleted and terminated.

```
$ cluster/kubectl.sh delete rc --all
$ cluster/kubectl.sh get pods
NAME      READY     STATUS    RESTARTS   AGE
```

### CPU limits

So what do you do if you want to control the maximum amount of CPU that your container can burst to use in order provide a consistent
level of service independent of CPU contention on the node?  You can specify an upper limit on the total amount of CPU that a pod's
container may consume.

To enforce this feature, your node must run a docker version >= 1.7, and your operating system kernel must
have support for CFS quota enabled.  Finally, your the Kubelet must be started with the following flag:

```
kubelet --cpu-cfs-quota=true
```

To demonstrate, let's create the same pod again, but this time set an upper limit to use 50% of a single CPU.

```
$ cluster/kubectl.sh run cpuhog \
    --image=busybox \
    --requests=cpu=100m \
    --limits=cpu=500m \
    -- md5sum /dev/urandom
```

Let's SSH into the node, and look at usage stats.

```
$ vagrant ssh minion-1
$ sudo su
$ docker stats $(docker ps -q)
CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
2a196edf7de2        47.38%              835.6 kB/1.042 GB   0.08%               0 B/0 B
...
```

As you can see, the container is no longer allowed to consume all available CPU on the node.  Instead, it is being limited to use
50% of a CPU over every 100ms period.  As a result, the reported value will be in the range of 50% but may oscillate above and below.

Let's delete all existing resources in preparation for the next scenario.  Verify all the pods are deleted and terminated.

```
$ cluster/kubectl.sh delete rc --all
$ cluster/kubectl.sh get pods
NAME      READY     STATUS    RESTARTS   AGE
```

### Memory requests

By default, a container is able to consume as much memory on the node as possible.  In order to improve placement of your
pods in the cluster, it is recommended to specify the amount of memory your container will require to run.  The scheduler
will then take available node memory capacity into account prior to binding your pod to a node.

Let's demonstrate this by creating a pod that runs a single container which requests 100Mi of memory.  The container will
allocate and write to 200MB of memory every 2 seconds.

```
$ cluster/kubectl.sh run memhog \
   --image=derekwaynecarr/memhog \
   --requests=memory=100Mi \
   --command \
   -- /bin/sh -c "while true; do memhog -r100 200m; sleep 1; done"
```

If you look at output of docker stats on the node:

```
$ docker stats $(docker ps -q)
CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
2badf74ae782        0.00%               1.425 MB/1.042 GB   0.14%               816 B/348 B
a320182967fa        105.81%             214.2 MB/1.042 GB   20.56%              0 B/0 B

```

As you can see, the container is using approximately 200MB of memory, and is only limited to the 1GB of memory on the node.

We scheduled against 100Mi, but have burst our memory usage to a greater value.

We refer to this as memory having __Burstable__ quality of service for this container.

Let's delete all existing resources in preparation for the next scenario.  Verify all the pods are deleted and terminated.

```
$ cluster/kubectl.sh delete rc --all
$ cluster/kubectl.sh get pods
NAME      READY     STATUS    RESTARTS   AGE
```

### Memory limits

If you specify a memory limit, you can constrain the amount of memory your container can use.

For example, let's limit our container to 200Mi of memory, and just consume 100MB.

```
$ cluster/kubectl.sh run memhog \
   --image=derekwaynecarr/memhog \
   --limits=memory=200Mi \
   --command -- /bin/sh -c "while true; do memhog -r100 100m; sleep 1; done"
```

If you look at output of docker stats on the node:

```
$ docker stats $(docker ps -q)
CONTAINER           CPU %               MEM USAGE/LIMIT     MEM %               NET I/O
5a7c22ae1837        125.23%             109.4 MB/209.7 MB   52.14%              0 B/0 B
c1d7579c9291        0.00%               1.421 MB/1.042 GB   0.14%               1.038 kB/816 B
```

As you can see, we are limited to 200Mi memory, and are only consuming 109.4MB on the node.

Let's demonstrate what happens if you exceed your allowed memory usage by creating a replication controller
whose pod will keep being OOM killed because it attempts to allocate 300MB of memory, but is limited to 200Mi.

```
$ cluster/kubectl.sh run memhog-oom    --image=derekwaynecarr/memhog    --limits=memory=200Mi    --command -- memhog -r100 300m
```

If we describe the created pod, you will see that it keeps restarting until it ultimately goes into a CrashLoopBackOff.

The reason it is killed and restarts is because it is OOMKilled as it attempts to exceed its memory limit.

```
$ cluster/kubectl.sh get pods
NAME               READY     STATUS             RESTARTS   AGE
memhog-oom-gj9hw   0/1       CrashLoopBackOff   2          26s
$ cluster/kubectl.sh describe pods/memhog-oom-gj9hw | grep -C 3 "Terminated"
      memory:           200Mi
    State:          Waiting
      Reason:           CrashLoopBackOff
    Last Termination State: Terminated
      Reason:           OOMKilled
      Exit Code:        137
      Started:          Wed, 23 Sep 2015 15:23:58 -0400
```

Let's clean-up before proceeding further.

```
$ cluster/kubectl.sh delete rc --all
```

### What if my node runs out of memory?

If you only schedule __Guaranteed__ memory containers, where the request is equal to the limit, then you are not in major danger of
causing an OOM event on your node.  If any individual container consumes more than their specified limit, it will be killed.

If you schedule __BestEffort__ memory containers, where the request and limit is not specified, or __Burstable__ memory containers, where
the request is less than any specified limit, then it is possible that a container will request more memory than what is actually available on the node.

If this occurs, the system will attempt to prioritize the containers that are killed based on their quality of service.  This is done
by using the OOMScoreAdjust feature in the Linux kernel which provides a heuristic to rank a process between -1000 and 1000.  Processes
with lower values are preserved in favor of processes with higher values.  The system daemons (kubelet, kube-proxy, docker) all run with
low OOMScoreAdjust values.

In simplest terms, containers with __Guaranteed__ memory containers are given a lower value than __Burstable__ containers which has
a lower value than __BestEffort__ containers.  As a consequence, containers with __BestEffort__ should be killed before the other tiers.

To demonstrate this, let's spin up a set of different replication controllers that will over commit the node.

```
$ cluster/kubectl.sh run mem-guaranteed --image=derekwaynecarr/memhog --replicas=2 --requests=cpu=10m --limits=memory=600Mi --command -- memhog -r100000 500m
$ cluster/kubectl.sh run mem-burstable --image=derekwaynecarr/memhog --replicas=2 --requests=cpu=10m,memory=600Mi --command -- memhog -r100000 100m
$ cluster/kubectl.sh run mem-besteffort --replicas=10 --image=derekwaynecarr/memhog --requests=cpu=10m --command -- memhog -r10000 500m
```

This will induce a SystemOOM

```
$ cluster/kubectl.sh get events | grep OOM
43m       8m        178       10.245.1.3             Node                                                        SystemOOM          {kubelet 10.245.1.3}        System OOM encountered
```

If you look at the pods:

```
$ cluster/kubectl.sh get pods
NAME                   READY     STATUS             RESTARTS   AGE
...
mem-besteffort-zpnpm   0/1       CrashLoopBackOff   4          3m
mem-burstable-n0yz1    1/1       Running            0          4m
mem-burstable-q3dts    1/1       Running            0          4m
mem-guaranteed-fqsw8   1/1       Running            0          4m
mem-guaranteed-rkqso   1/1       Running            0          4m
```

You see that our BestEffort pod goes in a restart cycle, but the pods with greater levels of quality of service continue to function.

As you can see, we rely on the Kernel to react to system OOM events.  Depending on how your host operating
system was configured, and which process the Kernel ultimately decides to kill on your Node, you may experience unstable results.  In addition, during an OOM event, while the kernel is cleaning up processes, the system may experience significant periods of slow down or appear unresponsive.  As a result, while the system allows you to overcommit on memory, we recommend to not induce a Kernel sys OOM.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/runtime-constraints/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
