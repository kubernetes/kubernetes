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

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/gang-scheduling.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Gang scheduling proposal

## Abstract

A proposal for scheduling a number of pods all at once, a.k.a. gang scheduling

## Motivation

Currently the kubernetes scheduler is a pod scheduler that schedules pods one
by one, it works great in most cases and would be continually improved by
adding new features like preeption, reservation. But in some particular use
cases such like when there're dependencies among the pods in a gang, the
pod scheduler can't handle very well and this is where gang scheduling comes
into play.

Gang scheduling is widely needed in industry and efforts have been made by some
stake holders, an example is [IBM PRS holistic scheduler plugin to openstack]
(http://www.slideshare.net/JarekMiszczyk/practical-advice-on-deployment-and-management-of-enterprise-workloads/7),
of which the solution is not that straight forward.

Based on such needs, this proposal was raised to extend kubernetes pod
scheduler to being more generic allows user to leverage gang scheduling
algorithms gracefully.

## Goals

1. Abstract out the basic behaviors of gang scheduling
2. Describe a basic design

# Non Goals

1. Scheduler algorithm is not the focus of this proposal
2. This proposal does not address the problem of handling the conflicts
   introduced by multiple schedulers or horizontal scaling schedulers

## Breaking down

### Interfaces

1. Introduce new API kind: "kind: Gang"
2. A gang contains the specs of Pods and ReplicationController

   ```yaml
apiVersion: extensions/v1beta1 
kind: Gang
metadata:
  name: mygang
  labels:
    name: mygang
spec:
  replicationControllers:
    - name: myapp
      labels:
        name: myapp
      replicas: 2
      selector:
        name: myapp
      template:
        metadata:
          labels:
            name: myapp
        spec:
          containers:
          - name: myapp1
            image: myapp1
            ports:
            - containerPort: 6222
          - name: myapp2
            image: myapp2
            ports:
            - containerPort: 6223
  pods:
    - name: mydb
      labels:
        name: mydb
      containers:
        - name: mydb
          image: mydb
          ports:
            - containerPort: 6224
   ```

3. Similar to the ReplicationController, gang status will be maintained and
   can be queried:

   ```bash
$ kubectl get gang
GANG         CONTAINER(S)   IMAGE(S)           SELECTOR          REPLICAS 
mygang       myapp1         myapp1             name=myapp        2
             myapp2         myapp2             name=myapp        2
             mydb           mydb
   ```

4. (Better to have) Creating a gang from the exsting ReplicationController and
   Pod description files through CLI argument "--gang=NAME"

   ```bash
kubectl create --gang=mygang -f ./mypod1.yaml -f ./mypod2.yaml -f ./myrc.yaml
   ```

### Scheduler

1. There should be no different behavior for the existing pods scheduler
   before and after gang scheduling was introduced
2. The existing pod scheduler can be extended to support gang scheduling
3. Gang schedulers can be running with the existing sequential schedulers
   simultaneously when the multiple schedulers(#17197) is enabled
4. Gang scheduler watches gang resources and queues them in a gang queue

   ```go
// ConfigFactory knows how to fill out a scheduler config with its support functions.
type ConfigFactory struct {
	Client *client.Client
	// queue for pods that need scheduling
	PodQueue *cache.FIFO
	// queue for gangs that need scheduling
	GangQueue *cache.FIFO
    ...
}

type Config struct {
    ...
	// NextPod should be a function that blocks until the next pod
	// is available. We don't use a channel for this, because scheduling
	// a pod may take some amount of time and we don't want pods to get
	// stale while they sit in a channel.
	NextPod func() *api.Pod
	NextGang func() *api.Gang
    ...
}

func (factory *ConfigFactory) createUnassignedGangLW() *cache.ListWatch {...}
   ```

5. The algorithm interface will be extended by adding the method for scheduling
   pods which accepts an array of pods as input and returns pod machine pairs

   ```go
type Pair struct {
    Pod *api.Pod
    Machine string
}

type ScheduleAlgorithm interface {
    Schedule(*api.Pod, NodeLister) (selectedMachine string, err error)
    SchedulePods([]*api.Pod, NodeLister) (placement []Pair, err error)
}
   ```

### Binding

1. After successfully scheduled, gang scheduler sends all bindings to the API
   server over the POST RPC, New binding API will be created to handle a list
   of the bindings

   ```
POST /api/v1/namespaces/{namespace}/bindingslists

{
  bindings: [
    {
      "kind": "string",
      "apiVersion": "string",
      "metadata": {
        "name": "string",
        "generateName": "string",
        "namespace": "string",
        "selfLink": "string",
        "uid": "string",
        "resourceVersion": "string",
        "generation": 0,
        "creationTimestamp": "string",
        "deletionTimestamp": "string",
        "deletionGracePeriodSeconds": 0
      },
      "target": {
        "kind": "string",
        "namespace": "string",
        "name": "string",
        "uid": "string",
        "apiVersion": "string",
        "resourceVersion": "string",
        "fieldPath": "string"
      }
    }
  ]
}
   ```

2. Will need the transaction support of etcd(#2904) to ensure the integrity of
   the commit to etcd.

   ```go
type Binder interface {
	Bind(binding *api.Binding) error
	BindMultiple(bindings []*api.Binding) error
}
   ```

### Termination

1. When deleting a gang, all Pods and ReplicationControllers belong to the gang
   should be deleted


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/gang-scheduling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
