## Cloud Native Deployments of Cassandra using Kubernetes

The following document describes the development of a _cloud native_ [Cassandra](http://cassandra.apache.org/) deployment on Kubernetes.  When we say _cloud native_ we mean an application which understands that it is running within a cluster manager, and uses this cluster management infrastructure to help implement the application.  In particular, in this instance, a custom Cassandra ```SeedProvider``` is used to enable Cassandra to dynamically discover new Cassandra nodes as they join the cluster.

This document also attempts to describe the core components of Kubernetes: _Pods_, _Services_, and _Replication Controllers_.

### Prerequisites
This example assumes that you have a Kubernetes cluster installed and running, and that you have installed the ```kubectl``` command line tool somewhere in your path.  Please see the [getting started](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/getting-started-guides) for installation instructions for your platform.

### A note for the impatient
This is a somewhat long tutorial.  If you want to jump straight to the "do it now" commands, please see the [tl; dr](#tl-dr) at the end.

### Simple Single Pod Cassandra Node
In Kubernetes, the atomic unit of an application is a [_Pod_](../../docs/pods.md).  A Pod is one or more containers that _must_ be scheduled onto the same host.  All containers in a pod share a network namespace, and may optionally share mounted volumes.  In this simple case, we define a single container running Cassandra for our pod:

```yaml
apiVersion: v1beta3
kind: Pod
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  containers:
  - args:
    - /run.sh
    resources:
      limits:
        cpu: "1"
    image: kubernetes/cassandra:v2
    name: cassandra
    ports:
    - name: cql
      containerPort: 9042
    - name: thrift
      containerPort: 9160
    volumeMounts:
    - name: data
      mountPath: /cassandra_data
    env:
    - name: MAX_HEAP_SIZE
      value: 512M
    - name: HEAP_NEWSIZE
      value: 100M
    - name: KUBERNETES_API_PROTOCOL
      value: http
  volumes:
    - name: data
      emptyDir: {}
```

There are a few things to note in this description.  First is that we are running the ```kubernetes/cassandra``` image.  This is a standard Cassandra installation on top of Debian.  However it also adds a custom [```SeedProvider```](https://svn.apache.org/repos/asf/cassandra/trunk/src/java/org/apache/cassandra/locator/SeedProvider.java) to Cassandra.  In Cassandra, a ```SeedProvider``` bootstraps the gossip protocol that Cassandra uses to find other nodes.  The ```KubernetesSeedProvider``` discovers the Kubernetes API Server using the built in Kubernetes discovery service, and then uses the Kubernetes API to find new nodes (more on this later)

You may also note that we are setting some Cassandra parameters (```MAX_HEAP_SIZE``` and ```HEAP_NEWSIZE```).  We also tell Kubernetes that the container exposes both the ```CQL``` and ```Thrift``` API ports.  Finally, we tell the cluster manager that we need 1 cpu (1 core).

Given this configuration, we can create the pod as follows

```sh
$ kubectl create -f cassandra.yaml
```

After a few moments, you should be able to see the pod running:

```sh
$ kubectl get pods cassandra
POD         IP           CONTAINER(S)   IMAGE(S)                  HOST                                    LABELS           STATUS    CREATED      MESSAGE
cassandra   10.244.3.3                                            kubernetes-minion-sft2/104.197.42.181   name=cassandra   Running   21 seconds
                         cassandra      kubernetes/cassandra:v2                                                            Running   3 seconds
```


### Adding a Cassandra Service
In Kubernetes a _[Service](../../docs/services.md)_ describes a set of Pods that perform the same task.  For example, the set of nodes in a Cassandra cluster, or even the single node we created above.  An important use for a Service is to create a load balancer which distributes traffic across members of the set.  But a _Service_ can also be used as a standing query which makes a dynamically changing set of Pods (or the single Pod we've already created) available via the Kubernetes API.  This is the way that we use initially use Services with Cassandra.

Here is the service description:
```yaml
apiVersion: v1beta3
kind: Service
metadata:
  labels:
    name: cassandra
  name: cassandra
spec:
  ports:
    - port: 9042
      targetPort: 9042
  selector:
    name: cassandra
```

The important thing to note here is the ```selector```. It is a query over labels, that identifies the set of _Pods_ contained by the _Service_.  In this case the selector is ```name=cassandra```.  If you look back at the Pod specification above, you'll see that the pod has the corresponding label, so it will be selected for membership in this Service.

Create this service as follows:
```sh
$ kubectl create -f cassandra-service.yaml
```

Once the service is created, you can query it's endpoints:
```sh
$ kubectl get endpoints cassandra -o yaml
apiVersion: v1beta3
kind: Endpoints
metadata:
  creationTimestamp: 2015-04-23T17:21:27Z
  name: cassandra
  namespace: default
  resourceVersion: "857"
  selfLink: /api/v1beta3/namespaces/default/endpoints/cassandra
  uid: 2c7d36bf-e9dd-11e4-a7ed-42010af011dd
subsets:
- addresses:
  - IP: 10.244.3.3
    targetRef:
      kind: Pod
      name: cassandra
      namespace: default
      resourceVersion: "769"
      uid: d185872c-e9dc-11e4-a7ed-42010af011dd
  ports:
  - port: 9042
    protocol: TCP

```

You can see that the _Service_ has found the pod we created in step one.

### Adding replicated nodes
Of course, a single node cluster isn't particularly interesting.  The real power of Kubernetes and Cassandra lies in easily building a replicated, scalable Cassandra cluster.

In Kubernetes a _[Replication Controller](../../docs/replication-controller.md)_ is responsible for replicating sets of identical pods.  Like a _Service_ it has a selector query which identifies the members of it's set.  Unlike a _Service_ it also has a desired number of replicas, and it will create or delete _Pods_ to ensure that the number of _Pods_ matches up with it's desired state.

Replication Controllers will "adopt" existing pods that match their selector query, so let's create a Replication Controller with a single replica to adopt our existing Cassandra Pod.

```yaml
apiVersion: v1beta3
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
              cpu: 1
          env:
            - name: MAX_HEAP_SIZE
              key: MAX_HEAP_SIZE
              value: 512M
            - name: HEAP_NEWSIZE
              key: HEAP_NEWSIZE
              value: 100M
          image: "kubernetes/cassandra:v2"
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

The bulk of the replication controller config is actually identical to the Cassandra pod declaration above, it simply gives the controller a recipe to use when creating new pods.  The other parts are the ```replicaSelector``` which contains the controller's selector query, and the ```replicas``` parameter which specifies the desired number of replicas, in this case 1.

Create this controller:

```sh
$ kubectl create -f cassandra-controller.yaml
```

Now this is actually not that interesting, since we haven't actually done anything new.  Now it will get interesting.

Let's scale our cluster to 2:
```sh
$ kubectl scale rc cassandra --replicas=2
```

Now if you list the pods in your cluster, you should see two cassandra pods:

```sh
$ kubectl get pods
POD                 IP              CONTAINER(S)   IMAGE(S)                 HOST                                    LABELS           STATUS    CREATED      MESSAGE
cassandra           10.244.3.3                                              kubernetes-minion-sft2/104.197.42.181   name=cassandra   Running   7 minutes
                                    cassandra      kubernetes/cassandra:v2                                                           Running   7 minutes
cassandra-gnhk8     10.244.0.5                                              kubernetes-minion-dqz3/104.197.2.71     name=cassandra   Running   About a minute
                                    cassandra      kubernetes/cassandra:v2                                                           Running   51 seconds

```

Notice that one of the pods has the human readable name ```cassandra``` that you specified in your config before, and one has a random string, since it was named by the replication controller.

To prove that this all works, you can use the ```nodetool``` command to examine the status of the cluster, for example:

```sh
$ ssh 104.197.42.181
$ docker exec <cassandra-container-id> nodetool status
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

Examining the status again:
```sh
$ docker exec <cassandra-container-id> nodetool status
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
# create a single cassandra node
kubectl create -f cassandra.yaml

# create a service to track all cassandra nodes
kubectl create -f cassandra-service.yaml

# create a replication controller to replicate cassandra nodes
kubectl create -f cassandra-controller.yaml

# scale up to 2 nodes
kubectl scale rc cassandra --replicas=2

# validate the cluster
docker exec <container-id> nodetool status

# scale up to 4 nodes
kubectl scale rc cassandra --replicas=4
```

### Seed Provider Source

See
[here](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/cassandra/java/src/io/k8s/cassandra/KubernetesSeedProvider.java).

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cassandra/README.md?pixel)]()
