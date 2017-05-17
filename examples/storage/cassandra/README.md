
# Cloud Native Deployments of Cassandra using Kubernetes

## Table of Contents

  - [Prerequisites](#prerequisites)
  - [Cassandra Docker](#cassandra-docker)
  - [Quickstart](#quickstart)
  - [Step 1: Create a Cassandra Headless Service](#step-1-create-a-cassandra-headless-service)
  - [Step 2: Use a StatefulSet to create Cassandra Ring](#step-2-use-a-statefulset-to-create-cassandra-ring)
  - [Step 3: Validate and Modify The Cassandra StatefulSet](#step-3-validate-and-modify-the-cassandra-statefulset)
  - [Step 4: Delete Cassandra StatefulSet](#step-4-delete-cassandra-statefulset)

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
- [_Stateful Sets_](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
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

## Cassandra Image

The pods uses the `quay.io/vorstella/cassandra-k8s:v1.4`
image hosted by quay.io's [container registry](https://quay.io/repository/vorstella/cassandra-k8s).
The docker is based on `ubuntu:slim` and includes OpenJDK 8. This image
includes a standard Cassandra installation from the Apache Debian repo.  Through the use of environment
variables you are able to change values that are inserted into the `cassandra.yaml`.

| ENV VAR       | DEFAULT VALUE  |
| ------------- |:-------------: |
| CASSANDRA_CLUSTER_NAME | 'Test Cluster'  |
| CASSANDRA_NUM_TOKENS  | 32               |
| CASSANDRA_RPC_ADDRESS | 0.0.0.0          |

## Quickstart

If you want to jump straight to the commands we will run,
here are the steps:

```sh
#
# StatefulSet
#

# create a service to track all cassandra statefulset nodes
kubectl create -f examples/storage/cassandra/cassandra-service.yaml

# create a statefulset and storage class
kubectl create -f examples/storage/cassandra/cassandra-statefulset.yaml

# validate the Cassandra cluster. Substitute the name of one of your pods.
kubectl exec -ti cassandra-0 -- nodetool status

# cleanup
grace=$(kubectl get po cassandra-0 --template '{{.spec.terminationGracePeriodSeconds}}') \
  && kubectl delete statefulset,po -l app=cassandra \
  && echo "Sleeping $grace" \
  && sleep $grace \
  && kubectl delete pvc -l app=cassandra

#
# Resource Controller Example
#

# create a replication controller to replicate cassandra nodes
kubectl create -f examples/storage/cassandra/cassandra-controller.yaml

# validate the Cassandra cluster. Substitute the name of one of your pods.
kubectl exec -ti cassandra-xxxxx -- nodetool status

# scale up the Cassandra cluster
kubectl scale rc cassandra --replicas=4

# delete the replication controller
kubectl delete rc cassandra

#
# Create a DaemonSet to place a cassandra node on each kubernetes node
#

kubectl create -f examples/storage/cassandra/cassandra-daemonset.yaml --validate=false

# resource cleanup
kubectl delete service -l app=cassandra
kubectl delete daemonset cassandra
```

## Step 1: Create a Cassandra Headless Service

A Kubernetes _[Service](../../../docs/user-guide/services.md)_ describes a set of
[_Pods_](../../../docs/user-guide/pods.md) that perform the same task. In
Kubernetes, the atomic unit of an application is a Pod: one or more containers
that _must_ be scheduled onto the same host.

The Service is used for DNS lookups between Cassandra Pods, and Cassandra clients
within the Kubernetes Cluster.

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
  clusterIP: None
  ports:
    - port: 9042
  selector:
    app: cassandra
```

[Download example](cassandra-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-service.yaml -->

Create the service for the StatefulSet:


```console
$ kubectl create -f examples/storage/cassandra/cassandra-service.yaml
```

The following command shows if the service has been created.

```console
$ kubectl get svc cassandra
```

The response should be like:

```console
NAME        CLUSTER-IP   EXTERNAL-IP   PORT(S)    AGE
cassandra   None         <none>        9042/TCP   45s
```

If an error is returned the service create failed.

## Step 2: Use a StatefulSet to create Cassandra Ring

StatefulSets (previously PetSets) are a feature that was upgraded to a <i>Beta</i> component in
Kubernetes 1.5.  Deploying stateful distributed applications, like Cassandra, within a clustered
environment can be challenging.  We implemented StatefulSet to greatly simplify this
process.  Multiple StatefulSet features are used within this example, but is out of
scope of this documentation.  [Please refer to the Stateful Set documentation.](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)

The StatefulSet manifest that is included below, creates a Cassandra ring that consists
of three pods.

This example includes using a GCE Storage Class, please update appropriately depending
on the cloud you are working with.

<!-- BEGIN MUNGE: EXAMPLE cassandra-statefulset.yaml -->

```yaml
apiVersion: "apps/v1beta1"
kind: StatefulSet
metadata:
  name: cassandra
spec:
  serviceName: cassandra
  replicas: 2
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
      - name: cassandra
        image: quay.io/vorstella/cassandra-k8s:v1.4
        imagePullPolicy: Always
        ports:
        - containerPort: 7000
          name: intra-node
        - containerPort: 7001
          name: tls-intra-node
        - containerPort: 7199
          name: jmx
        - containerPort: 9042
          name: cql
        resources:
          limits:
            cpu: "500m"
            memory: 1Gi
          requests:
           cpu: "500m"
           memory: 1Gi
        securityContext:
          capabilities:
            add:
              - IPC_LOCK
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "PID=$(pidof java) && kill $PID && while ps -p $PID > /dev/null; do sleep 1; done"]
        env:
          - name: MAX_HEAP_SIZE
            value: 512M
          - name: HEAP_NEWSIZE
            value: 100M
          - name: CASSANDRA_SEEDS
            value: "cassandra-0.cassandra.default.svc.cluster.local,cassandra-1.cassandra.default.svc.cluster.local"
          - name: CASSANDRA_CLUSTER_NAME
            value: "K8Demo"
          - name: CASSANDRA_DC
            value: "DC1-K8Demo"
          - name: CASSANDRA_RACK
            value: "Rack1-K8Demo"
          - name: CASSANDRA_AUTO_BOOTSTRAP
            value: "false"
          - name: POD_IP
            valueFrom:
              fieldRef:
                fieldPath: status.podIP
          - name: POD_NAMESPACE
            valueFrom:
              fieldRef:
                fieldPath: metadata.namespace
        readinessProbe:
          exec:
            command:
            - /bin/bash
            - -c
            - /ready-probe.sh
          initialDelaySeconds: 15
          timeoutSeconds: 5
        # These volume mounts are persistent. They are like inline claims,
        # but not exactly because the names need to match exactly one of
        # the stateful pod volumes.
        volumeMounts:
        - name: cassandra-data
          mountPath: /cassandra_data
  # These are converted to volume claims by the controller
  # and mounted at the paths mentioned above.
  volumeClaimTemplates:
  - metadata:
      name: cassandra-data
      annotations:
        volume.beta.kubernetes.io/storage-class: fast
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
---
#
# StorageClass for GCE
#
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: fast
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
#
# StorageClass for EC2
#
#kind: StorageClass
#apiVersion: storage.k8s.io/v1
#metadata:
#  name: fast
#provisioner: kubernetes.io/aws-ebs
#parameters:
# type: gp2
```

[Download example](cassandra-statefulset.yaml?raw=true)
<!-- END MUNGE: EXAMPLE cassandra-statefulset.yaml -->

Create the Cassandra StatefulSet as follows:

```console
$ kubectl create -f examples/storage/cassandra/cassandra-statefulset.yaml
```

## Step 3: Validate and Modify The Cassandra StatefulSet

Deploying this StatefulSet shows off two of the new features that StatefulSets provides.

1. The pod names are known
2. The pods deploy in incremental order

First validate that the StatefulSet has deployed, by running `kubectl` command below.

```console
$ kubectl get statefulset cassandra
```

The command should respond like:

```console
NAME        DESIRED   CURRENT   AGE
cassandra   3         3         13s
```

Next watch the Cassandra pods deploy, one after another.  The StatefulSet resource
deploys pods in a number fashion: 1, 2, 3, etc.  If you execute the following
command before the pods deploy you are able to see the ordered creation.

```console
$ kubectl get pods -l="app=cassandra"
NAME          READY     STATUS              RESTARTS   AGE
cassandra-0   1/1       Running             0          1m
cassandra-1   0/1       ContainerCreating   0          8s
```

The above example shows two of the three pods in the Cassandra StatefulSet deployed.
Once all of the pods are deployed the same command will respond with the full
StatefulSet.

```console
$ kubectl get pods -l="app=cassandra"
NAME          READY     STATUS    RESTARTS   AGE
cassandra-0   1/1       Running   0          10m
cassandra-1   1/1       Running   0          9m
cassandra-2   1/1       Running   0          8m
```

Running the Cassandra utility `nodetool` will display the status of the ring.

```console
$ kubectl exec cassandra-0 -- nodetool status
Datacenter: DC1-K8Demo
======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address   Load       Tokens       Owns (effective)  Host ID                               Rack
UN  10.4.2.4  65.26 KiB  32           63.7%             a9d27f81-6783-461d-8583-87de2589133e  Rack1-K8Demo
UN  10.4.0.4  102.04 KiB  32           66.7%             5559a58c-8b03-47ad-bc32-c621708dc2e4  Rack1-K8Demo
UN  10.4.1.4  83.06 KiB  32           69.6%             9dce943c-581d-4c0e-9543-f519969cc805  Rack1-K8Demo
```

You can also run `cqlsh` to describe the keyspaces in the cluster.

```console
$ kubectl exec cassandra-0 -- cqlsh -e 'desc keyspaces'

system_traces  system_schema  system_auth  system  system_distributed
```

In order to increase or decrease the size of the Cassandra StatefulSet, you case use
`kubectl scale`.

Use the following command to scale the StatefulSet.

```console
$ kubectl scale statefulset cassandra --replicas=5
```

The StatefulSet will now contain five pods.

```console
$ kubectl get statefulset cassandra
```

The command should respond like:

```console
NAME        DESIRED   CURRENT   AGE
cassandra   5         5         36m
```

## Step 4: Delete Cassandra StatefulSet

Deleting and/or scaling a StatefulSet down will not delete the volumes associated with the StatefulSet. This is done to ensure safety first, your data is more valuable than an auto purge of all related StatefulSet resources. Deleting the Persistent Volume Claims may result in a deletion of the associated volumes, depending on the storage class and reclaim policy. You should never assume ability to access a volume after claim deletion.

Use the following commands to delete the StatefulSet.

```console
$ grace=$(kubectl get po cassandra-0 --template '{{.spec.terminationGracePeriodSeconds}}') \
  && kubectl delete statefulset -l app=cassandra \
  && echo "Sleeping $grace" \
  && sleep $grace \
  && kubectl delete pvc -l app=cassandra \
  && kubectl delete svc cassandra
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/cassandra/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
