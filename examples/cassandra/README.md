<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Cloud Native Deployments of Cassandra using Kubernetes

The following document describes the development of a _cloud native_ [Cassandra](http://cassandra.apache.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application.  In particular, in this instance, a custom Cassandra ```SeedProvider``` is used to enable Cassandra to dynamically discover new Cassandra nodes as they join the cluster.

This document also attempts to describe the core components of Kubernetes: _Pods_, _Services_, and _Replication Controllers_.

### Prerequisites

This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](../../docs/getting-started-guides/) for installation instructions for your platform.

This example also has a few code and configuration files needed.  To avoid typing these out, you can ```git clone``` the Kubernetes repository to you local computer.

### A note for the impatient

This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Simple Single Pod Cassandra Node

In Kubernetes, the atomic unit of an application is a [_Pod_](../../docs/user-guide/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.
In this simple case, we define a single container running Cassandra for our pod:

<!-- BEGIN MUNGE: EXAMPLE cassandra-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  replicas: 1
  selector:
    name: cassandra
  template:
    metadata:
      labels:
        name: cassandra
    spec:
      containers:
        - command:
            - /run.sh
          resources:
            limits:
              cpu: 0.1
          env:
            - name: MAX_HEAP_SIZE
              value: 512M
            - name: HEAP_NEWSIZE
              value: 100M
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          image: gcr.io/google_containers/cassandra:v6
          name: cassandra
          ports:
            - containerPort: 9042
              name: cql
            - containerPort: 9160
              name: thrift
          volumeMounts:
            - mountPath: /cassandra_data
              name: data
      volumes:
        - name: data
          emptyDir: {}
```

[Download example](cassandra-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-controller.yaml -->

There are a few things to note in this description.  First is that we are running the ```kubernetes/cassandra``` image.  This is a standard Cassandra installation on top of Debian.  However it also adds a custom [```SeedProvider```](https://svn.apache.org/repos/asf/cassandra/trunk/src/java/org/apache/cassandra/locator/SeedProvider.java) to Cassandra.  In Cassandra, a ```SeedProvider``` bootstraps the gossip protocol that Cassandra uses to find other nodes.  The ```KubernetesSeedProvider``` discovers the Kubernetes API Server using the built in Kubernetes discovery service, and then uses the Kubernetes API to find new nodes (more on this later)

You may also note that we are setting some Cassandra parameters (```MAX_HEAP_SIZE``` and ```HEAP_NEWSIZE```) and adding information about the [namespace](../../docs/user-guide/namespaces.md).  We also tell Kubernetes that the container exposes both the ```CQL``` and ```Thrift``` API ports.  Finally, we tell the cluster manager that we need 0.5 cpu (0.5 core).

In theory could create a single Cassandra pod right now but since `KubernetesSeedProvider` needs to learn what nodes are in the Cassandra deployment we need to create a service first.

### Cassandra Service

In Kubernetes a _[Service](../../docs/user-guide/services.md)_ describes a set of Pods that perform the same task.  For example, the set of Pods in a Cassandra cluster can be a Kubernetes Service, or even just the single Pod we created above.  An important use for a Service is to create a load balancer which distributes traffic across members of the set of Pods.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods (or the single Pod we've already created) available via the Kubernetes API.  This is the way that we use initially use Services with Cassandra.

Here is the service description:

<!-- BEGIN MUNGE: EXAMPLE cassandra-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  ports:
    - port: 9042
  selector:
    name: cassandra
```

[Download example](cassandra-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-service.yaml -->

The important thing to note here is the ```selector```. It is a query over labels, that identifies the set of _Pods_ contained by the _Service_.  In this case the selector is ```name=cassandra```.  If you look back at the Pod specification above, you'll see that the pod has the corresponding label, so it will be selected for membership in this Service.

Create this service as follows:

```console
$ kubectl create -f examples/cassandra/cassandra-service.yaml
```

Now, as the service is running, we can create the first Cassandra pod using the mentioned specification.

```console
$ kubectl create -f examples/cassandra/cassandra.yaml
```

After a few moments, you should be able to see the pod running, plus its single container:

```console
$ kubectl get pods cassandra
NAME        READY     STATUS    RESTARTS   AGE
cassandra   1/1       Running   0          55s
```

You can also query the service endpoints to check if the pod has been correctly selected.

```console
$ kubectl get endpoints cassandra -o yaml
apiVersion: v1
kind: Endpoints
metadata:
  creationTimestamp: 2015-06-21T22:34:12Z
  labels:
    name: cassandra
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

### Adding replicated nodes

Of course, a single node cluster isn't particularly interesting.  The real power of Kubernetes and Cassandra lies in easily building a replicated, scalable Cassandra cluster.

In Kubernetes a _[Replication Controller](../../docs/user-guide/replication-controller.md)_ is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Replication controllers will "adopt" existing pods that match their selector query, so let's create a replication controller with a single replica to adopt our existing Cassandra pod.

<!-- BEGIN MUNGE: EXAMPLE cassandra-controller.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  replicas: 1
  selector:
    name: cassandra
  template:
    metadata:
      labels:
        name: cassandra
    spec:
      containers:
        - command:
            - /run.sh
          resources:
            limits:
              cpu: 0.1
          env:
            - name: MAX_HEAP_SIZE
              value: 512M
            - name: HEAP_NEWSIZE
              value: 100M
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          image: gcr.io/google_containers/cassandra:v6
          name: cassandra
          ports:
            - containerPort: 9042
              name: cql
            - containerPort: 9160
              name: thrift
          volumeMounts:
            - mountPath: /cassandra_data
              name: data
      volumes:
        - name: data
          emptyDir: {}
```

[Download example](cassandra-controller.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-controller.yaml -->

Most of this replication controller definition is identical to the Cassandra pod definition above, it simply gives the replication controller a recipe to use when it creates new Cassandra pods.  The other differentiating parts are the ```selector``` attribute which contains the controller's selector query, and the ```replicas``` attribute which specifies the desired number of replicas, in this case 1.

Create this controller:

```console
$ kubectl create -f examples/cassandra/cassandra-controller.yaml
```

Now this is actually not that interesting, since we haven't actually done anything new.  Now it will get interesting.

Let's scale our cluster to 2:

```console
$ kubectl scale rc cassandra --replicas=2
```

Now if you list the pods in your cluster, and filter to the label ```name=cassandra```, you should see two cassandra pods:

```console 
$ kubectl get pods -l="name=cassandra"
NAME              READY     STATUS    RESTARTS   AGE
cassandra         1/1       Running   0          3m
cassandra-af6h5   1/1       Running   0          28s
```

Notice that one of the pods has the human readable name ```cassandra``` that you specified in your config before, and one has a random string, since it was named by the replication controller.

To prove that this all works, you can use the ```nodetool``` command to examine the status of the cluster.  To do this, use the ```kubectl exec``` command to run ```nodetool``` in one of your Cassandra pods.

```console
$ kubectl exec -ti cassandra -- nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.0.5  74.09 KB   256     100.0%            86feda0f-f070-4a5b-bda1-2eeb0ad08b77  rack1
UN  10.244.3.3  51.28 KB   256     100.0%            dafe3154-1d67-42e1-ac1d-78e7e80dce2b  rack1
```

Now let's scale our cluster to 4 nodes:

```sh
$ kubectl scale rc cassandra --replicas=4
```

In a few moments, you can examine the status again:

```sh
$ kubectl exec -ti cassandra -- nodetool status
Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens  Owns (effective)  Host ID                               Rack
UN  10.244.2.3  57.61 KB   256     49.1%             9d560d8e-dafb-4a88-8e2f-f554379c21c3  rack1
UN  10.244.1.7  41.1 KB    256     50.2%             68b8cc9c-2b76-44a4-b033-31402a77b839  rack1
UN  10.244.0.5  74.09 KB   256     49.7%             86feda0f-f070-4a5b-bda1-2eeb0ad08b77  rack1
UN  10.244.3.3  51.28 KB   256     51.0%             dafe3154-1d67-42e1-ac1d-78e7e80dce2b  rack1
```

### tl; dr;

For those of you who are impatient, here is the summary of the commands we ran in this tutorial.

```sh
# create a service to track all cassandra nodes
kubectl create -f examples/cassandra/cassandra-service.yaml

# create a single cassandra node
kubectl create -f examples/cassandra/cassandra.yaml

# create a replication controller to replicate cassandra nodes
kubectl create -f examples/cassandra/cassandra-controller.yaml

# scale up to 2 nodes
kubectl scale rc cassandra --replicas=2

# validate the cluster
kubectl exec -ti cassandra -- nodetool status

# scale up to 4 nodes
kubectl scale rc cassandra --replicas=4
```

### Seed Provider Source

See [here](java/src/io/k8s/cassandra/KubernetesSeedProvider.java).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cassandra/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
