
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

# Cloud Native Deployments of Cassandra using Kubernetes

## Table of Contents

  - [Prerequisites](#prerequisites)
  - [Cassandra Docker](#cassandra-docker)
  - [tl;dr Quickstart](#tldr-quickstart)
  - [Step 1: Create a Cassandra Service](#step-1-create-a-cassandra-service)
  - [Step 2: Use a Replication Controller to create Cassandra node pods](#step-2-use-a-replication-controller-to-create-cassandra-node-pods)
  - [Step 3: Scale up the Cassandra cluster](#step-3-scale-up-the-cassandra-cluster)
  - [Step 4: Delete the Replication Controller](#step-4-delete-the-replication-controller)
  - [Step 5: Use a DaemonSet instead of a Replication Controller](#step-5-use-a-daemonset-instead-of-a-replication-controller)
  - [Step 6: Resource Cleanup](#step-6-resource-cleanup)
  - [Seed Provider Source](#seed-provider-source)

The following document describes the development of a _cloud native_
[Cassandra](http://cassandra.apache.org/) deployment on Kubernetes.  When we say
_cloud native_, we mean an application which understands that it is running
within a cluster manager, and uses this cluster management infrastructure to
help implement the application.  In particular, in this instance, a custom
Cassandra `SeedProvider` is used to enable Cassandra to dynamically discover
new Cassandra nodes as they join the cluster.

This example also uses some of the core components of Kubernetes:

- [_Pods_](../../../docs/user-guide/pods.md)
- [ _Services_](../../../docs/user-guide/services.md)
- [_Replication Controllers_](../../../docs/user-guide/replication-controller.md)
- [_Daemon Sets_](../../../docs/admin/daemons.md)

## Prerequisites

This example assumes that you have a Kubernetes version >=1.2 cluster installed and running,
and that you have installed the [`kubectl`](../../../docs/user-guide/kubectl/kubectl.md)
command line tool somewhere in your path.  Please see the
[getting started guides](../../../docs/getting-started-guides/)
for installation instructions for your platform.

This example also has a few code and configuration files needed.  To avoid
typing these out, you can `git clone` the Kubernetes repository to your local
computer.

## Cassandra Docker

The pods use the [```gcr.io/google-samples/cassandra:v9```](image/Dockerfile)
image from Google's [container registry](https://cloud.google.com/container-registry/docs/).
The docker is based on `debian:jessie` and includes OpenJDK 8. This image
includes a standard Cassandra installation from the Apache Debian repo.  Through the use
of environment variables you are able to change values that are inserted into the `cassandra.yaml`.

| ENV VAR       | DEFAULT VALUE  |
| ------------- |:-------------: |
| CASSANDRA_CLUSTER_NAME | 'Test Cluster'  |
| CASSANDRA_NUM_TOKENS  | 32               |
| CASSANDRA_RPC_ADDRESS | 0.0.0.0          |

### Custom Seed Provider

A custom [`SeedProvider`](https://svn.apache.org/repos/asf/cassandra/trunk/src/java/org/apache/cassandra/locator/SeedProvider.java)
is included for running Cassandra on top of Kubernetes.  In Cassandra, a
`SeedProvider` bootstraps the gossip protocol that Cassandra uses to find other
Cassandra nodes. Seed addresses are hosts deemed as contact points. Cassandra
instances use the seed list to find each other and learn the topology of the
ring. The [`KubernetesSeedProvider`](java/src/main/java/io/k8s/cassandra/KubernetesSeedProvider.java)
discovers Cassandra seeds IP addresses vis the Kubernetes API, those Cassandra
instances are defined within the Cassandra Service.

Refer to the custom seed provider [README](java/README.md) for further
`KubernetesSeedProvider` configurations. For this example you should not need
to customize the Seed Provider configurations.

See the [image](image/) directory of this example for specifics on
how the container docker image was built and what it contains.

You may also note that we are setting some Cassandra parameters (`MAX_HEAP_SIZE`
and `HEAP_NEWSIZE`), and adding information about the
[namespace](../../../docs/user-guide/namespaces.md).
We also tell Kubernetes that the container exposes
both the `CQL` and `Thrift` API ports.  Finally, we tell the cluster
manager that we need 0.1 cpu (0.1 core).

## tl;dr Quickstart

If you want to jump straight to the commands we will run,
here are the steps:

```sh
# create a service to track all cassandra nodes
kubectl create -f examples/storage/cassandra/cassandra-service.yaml

# create a replication controller to replicate cassandra nodes
kubectl create -f examples/storage/cassandra/cassandra-controller.yaml

# validate the Cassandra cluster. Substitute the name of one of your pods.
kubectl exec -ti cassandra-xxxxx -- nodetool status

# scale up the Cassandra cluster
kubectl scale rc cassandra --replicas=4

# delete the replication controller
kubectl delete rc cassandra

# then, create a daemonset to place a cassandra node on each kubernetes node
kubectl create -f examples/storage/cassandra/cassandra-daemonset.yaml --validate=false

# resource cleanup
kubectl delete service -l app=cassandra
kubectl delete daemonset cassandra
```

## Step 1: Create a Cassandra Service

A Kubernetes _[Service](../../../docs/user-guide/services.md)_ describes a set of
[_Pods_](../../../docs/user-guide/pods.md) that perform the same task. In
Kubernetes, the atomic unit of an application is a Pod: one or more containers
that _must_ be scheduled onto the same host.

An important use for a Service is to create a load balancer which
distributes traffic across members of the set of Pods.  But a Service can also
be used as a standing query which makes a dynamically changing set of Pods
available via the Kubernetes API.  We'll show that in this example.

Here is the service description:

<!-- BEGIN MUNGE: EXAMPLE cassandra-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    app: cassandra
  name: cassandra
spec:
  ports:
    - port: 9042
  selector:
    app: cassandra
```

[Download example](cassandra-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-service.yaml -->

An important thing to note here is the `selector`. It is a query over labels,
that identifies the set of Pods contained by this Service.  In this case the
selector is `app=cassandra`. If there are any pods with that label, they will be
selected for membership in this service. We'll see that in action shortly.

Create the Cassandra service as follows:

```console
$ kubectl create -f examples/storage/cassandra/cassandra-service.yaml
```


## Step 2: Use a Replication Controller to create Cassandra node pods

As we noted above, in Kubernetes, the atomic unit of an application is a
[_Pod_](../../../docs/user-guide/pods.md).
A Pod is one or more containers that _must_ be scheduled onto
the same host.  All containers in a pod share a network namespace, and may
optionally share mounted volumes.

A Kubernetes
_[Replication Controller](../../../docs/user-guide/replication-controller.md)_
is responsible for replicating sets of identical pods.  Like a
Service, it has a selector query which identifies the members of its set.
Unlike a Service, it also has a desired number of replicas, and it will create
or delete Pods to ensure that the number of Pods matches up with its
desired state.

The Replication Controller, in conjunction with the Service we just defined,
will let us easily build a replicated, scalable Cassandra cluster.

Let's create a replication controller with two initial replicas.

<!-- BEGIN MUNGE: EXAMPLE cassandra-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: cassandra
  # The labels will be applied automatically
  # from the labels in the pod template, if not set
  # labels:
    # app: cassandra
spec:
  replicas: 2
  # The selector will be applied automatically
  # from the labels in the pod template, if not set.
  # selector:
      # app: cassandra
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
        - command:
            - /run.sh
          resources:
            limits:
              cpu: 0.5
          env:
            - name: MAX_HEAP_SIZE
              value: 512M
            - name: HEAP_NEWSIZE
              value: 100M
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
          image: gcr.io/google-samples/cassandra:v9
          name: cassandra
          ports:
            - containerPort: 7000
              name: intra-node
            - containerPort: 7001
              name: tls-intra-node
            - containerPort: 7199
              name: jmx
            - containerPort: 9042
              name: cql
            # If you need it it is going away in C* 4.0
            #- containerPort: 9160
            #  name: thrift
          volumeMounts:
            - mountPath: /cassandra_data
              name: data
      volumes:
        - name: data
          emptyDir: {}
```

[Download example](cassandra-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-controller.yaml -->

There are a few things to note in this description.

The `selector` attribute contains the controller's selector query. It can be
explicitly specified, or applied automatically from the labels in the pod
template if not set, as is done here.

The pod template's label, `app:cassandra`, matches the Service selector
from Step 1. This is how pods created by this replication controller are picked up
by the Service."

The `replicas` attribute specifies the desired number of replicas, in this
case 2 initially.  We'll scale up to more shortly.



Create the Replication Controller:

```console

$ kubectl create -f examples/storage/cassandra/cassandra-controller.yaml

```

You can list the new controller:

```console

$ kubectl get rc -o wide
NAME        DESIRED   CURRENT   AGE       CONTAINER(S)   IMAGE(S)                             SELECTOR
cassandra   2         2         11s       cassandra      gcr.io/google-samples/cassandra:v9   app=cassandra

```

Now if you list the pods in your cluster, and filter to the label
`app=cassandra`, you should see two Cassandra pods. (The `wide` argument lets
you see which Kubernetes nodes the pods were scheduled onto.)

```console

$ kubectl get pods -l="app=cassandra" -o wide
NAME              READY     STATUS    RESTARTS   AGE       NODE
cassandra-21qyy   1/1       Running   0          1m        kubernetes-minion-b286
cassandra-q6sz7   1/1       Running   0          1m        kubernetes-minion-9ye5

```

Because these pods have the label `app=cassandra`, they map to the service we
defined in Step 1.

You can check that the Pods are visible to the Service using the following service endpoints query:

```console

$ kubectl get endpoints cassandra -o yaml
apiVersion: v1
kind: Endpoints
metadata:
  creationTimestamp: 2015-06-21T22:34:12Z
  labels:
    app: cassandra
  name: cassandra
  namespace: default
  resourceVersion: "944373"
  selfLink: /api/v1/namespaces/default/endpoints/cassandra
  uid: a3d6c25f-1865-11e5-a34e-42010af01bcc
subsets:
- addresses:
  - ip: 10.244.3.15
    targetRef:
      kind: Pod
      name: cassandra
      namespace: default
      resourceVersion: "944372"
      uid: 9ef9895d-1865-11e5-a34e-42010af01bcc
  ports:
  - port: 9042
    protocol: TCP

```

To show that the `SeedProvider` logic is working as intended, you can use the
`nodetool` command to examine the status of the Cassandra cluster.  To do this,
use the `kubectl exec` command, which lets you run `nodetool` in one of your
Cassandra pods.  Again, substitute `cassandra-xxxxx` with the actual name of one
of your pods.

```console

$ kubectl exec -ti cassandra-xxxxx -- nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.0.5  74.09 KB   256     100.0%            86feda0f-f070-4a5b-bda1-2eeb0ad08b77  rack1
UN  10.244.3.3  51.28 KB   256     100.0%            dafe3154-1d67-42e1-ac1d-78e7e80dce2b  rack1

```

## Step 3: Scale up the Cassandra cluster

Now let's scale our Cassandra cluster to 4 pods.  We do this by telling the
Replication Controller that we now want 4 replicas.

```sh

$ kubectl scale rc cassandra --replicas=4

```

You can see the new pods listed:

```console

$ kubectl get pods -l="app=cassandra" -o wide
NAME              READY     STATUS    RESTARTS   AGE       NODE
cassandra-21qyy   1/1       Running   0          6m        kubernetes-minion-b286
cassandra-81m2l   1/1       Running   0          47s       kubernetes-minion-b286
cassandra-8qoyp   1/1       Running   0          47s       kubernetes-minion-9ye5
cassandra-q6sz7   1/1       Running   0          6m        kubernetes-minion-9ye5

```

In a few moments, you can examine the Cassandra cluster status again, and see
that the new pods have been detected by the custom `SeedProvider`:

```console

$ kubectl exec -ti cassandra-xxxxx -- nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.0.6  51.67 KB   256     48.9%             d07b23a5-56a1-4b0b-952d-68ab95869163  rack1
UN  10.244.1.5  84.71 KB   256     50.7%             e060df1f-faa2-470c-923d-ca049b0f3f38  rack1
UN  10.244.1.6  84.71 KB   256     47.0%             83ca1580-4f3c-4ec5-9b38-75036b7a297f  rack1
UN  10.244.0.5  68.2 KB    256     53.4%             72ca27e2-c72c-402a-9313-1e4b61c2f839  rack1

```

## Step 4: Delete the Replication Controller

Before you start Step 5, __delete the replication controller__ you created above:

```sh

$ kubectl delete rc cassandra

```

## Step 5: Use a DaemonSet instead of a Replication Controller

In Kubernetes, a [_Daemon Set_](../../../docs/admin/daemons.md) can distribute pods
onto Kubernetes nodes, one-to-one.  Like a _ReplicationController_, it has a
selector query which identifies the members of its set.  Unlike a
_ReplicationController_, it has a node selector to limit which nodes are
scheduled with the templated pods, and replicates not based on a set target
number of pods, but rather assigns a single pod to each targeted node.

An example use case: when deploying to the cloud, the expectation is that
instances are ephemeral and might die at any time. Cassandra is built to
replicate data across the cluster to facilitate data redundancy, so that in the
case that an instance dies, the data stored on the instance does not, and the
cluster can react by re-replicating the data to other running nodes.

`DaemonSet` is designed to place a single pod on each node in the Kubernetes
cluster.  That will give us data redundancy. Let's create a
daemonset to start our storage cluster:

<!-- BEGIN MUNGE: EXAMPLE cassandra-daemonset.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: DaemonSet
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      # Filter to specific nodes:
      # nodeSelector:
      #  app: cassandra
      containers:
        - command:
            - /run.sh
          env:
            - name: MAX_HEAP_SIZE
              value: 512M
            - name: HEAP_NEWSIZE
              value: 100M
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
          image: gcr.io/google-samples/cassandra:v9
          name: cassandra
          ports:
            - containerPort: 7000
              name: intra-node
            - containerPort: 7001
              name: tls-intra-node
            - containerPort: 7199
              name: jmx
            - containerPort: 9042
              name: cql
              # If you need it it is going away in C* 4.0
              #- containerPort: 9160
              #  name: thrift
          resources:
            requests:
              cpu: 0.5
          volumeMounts:
            - mountPath: /cassandra_data
              name: data
      volumes:
        - name: data
          emptyDir: {}
```

[Download example](cassandra-daemonset.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-daemonset.yaml -->

Most of this Daemonset definition is identical to the ReplicationController
definition above; it simply gives the daemon set a recipe to use when it creates
new Cassandra pods, and targets all Cassandra nodes in the cluster.

Differentiating aspects are the `nodeSelector` attribute, which allows the
Daemonset to target a specific subset of nodes (you can label nodes just like
other resources), and the lack of a `replicas` attribute due to the 1-to-1 node-
pod relationship.

Create this daemonset:

```console

$ kubectl create -f examples/storage/cassandra/cassandra-daemonset.yaml

```

You may need to disable config file validation, like so:

```console

$ kubectl create -f examples/storage/cassandra/cassandra-daemonset.yaml --validate=false

```

You can see the daemonset running:

```console

$ kubectl get daemonset
NAME        DESIRED   CURRENT   NODE-SELECTOR
cassandra   3         3         <none>

```

Now, if you list the pods in your cluster, and filter to the label
`app=cassandra`, you should see one (and only one) new cassandra pod for each
node in your network.

```console

$ kubectl get pods -l="app=cassandra" -o wide
NAME              READY     STATUS    RESTARTS   AGE       NODE
cassandra-ico4r   1/1       Running   0          4s        kubernetes-minion-rpo1
cassandra-kitfh   1/1       Running   0          1s        kubernetes-minion-9ye5
cassandra-tzw89   1/1       Running   0          2s        kubernetes-minion-b286

```

To prove that this all worked as intended, you can again use the `nodetool`
command to examine the status of the cluster.  To do this, use the `kubectl
exec` command to run `nodetool` in one of your newly-launched cassandra pods.

```console

$ kubectl exec -ti cassandra-xxxxx -- nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.0.5  74.09 KB   256     100.0%            86feda0f-f070-4a5b-bda1-2eeb0ad08b77  rack1
UN  10.244.4.2  32.45 KB   256     100.0%            0b1be71a-6ffb-4895-ac3e-b9791299c141  rack1
UN  10.244.3.3  51.28 KB   256     100.0%            dafe3154-1d67-42e1-ac1d-78e7e80dce2b  rack1

```

**Note**: This example had you delete the cassandra Replication Controller before
you created the Daemonset.  This is because – to keep this example simple – the
RC and the Daemonset are using the same `app=cassandra` label (so that their pods map to the
service we created, and so that the SeedProvider can identify them).

If we didn't delete the RC first, the two resources would conflict with
respect to how many pods they wanted to have running. If we wanted, we could support running
both together by using additional labels and selectors.

## Step 6: Resource Cleanup

When you are ready to take down your resources, do the following:

```console

$ kubectl delete service -l app=cassandra
$ kubectl delete daemonset cassandra

```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/cassandra/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
