# Kubernetes Proposal: ResourceController

**Related PR:** 

| Topic | Link |
| ----- | ---- |
| Admission Control Proposal | https://github.com/GoogleCloudPlatform/kubernetes/pull/2501 |
| Separate validation from RESTStorage | https://github.com/GoogleCloudPlatform/kubernetes/issues/2977 |

## Background

This document proposes a system for enforcing resource limits as part of admission control.

## Model Changes

A new resource, **ResourceController**, is introduced to enumerate resource usage constraints scoped to a Kubernetes namespace.

Authorized users are able to set the **ResourceController.Spec** fields to enumerate desired constraints.

```
// ResourceController is an enumerated set of resources constraints enforced as part admission control plug-in
type ResourceController struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`
  // Spec represents the imposed constraints for allowed resources
  Spec ResourceControllerSpec `json:"spec,omitempty"`
  // Status represents the observed allocated resources to inform constraints
  Status ResourceControllerStatus `json:"status,omitempty"`
}

type ResourceControllerSpec struct {
  // Allowed represents the available resources allowed in a quota
  Allowed ResourceList `json:"allowed,omitempty"`
}

type ResourceControllerStatus struct {
  // Allowed represents the available resources allowed in a quota
  Allowed ResourceList `json:"allowed,omitempty"`
  // Allocated represents the allocated resources leveraged against your quota
  Allocated ResourceList `json:"allocated,omitempty"`
}

// ResourceControllerList is a collection of resource controllers.
type ResourceControllerList struct {
  TypeMeta `json:",inline"`
  ListMeta `json:"metadata,omitempty"`
  Items    []ResourceController `json:"items"`
}
```

Authorized users are able to provide a **ResourceObservation** to control a **ResourceController.Status**.

```
// ResourceObservation is written by a resource-controller to update ResourceController.Status
type ResourceObservation struct {
  TypeMeta   `json:",inline"`
  ObjectMeta `json:"metadata,omitempty"`

  // Status represents the observed allocated resources to inform constraints
  Status ResourceControllerStatus `json:"status,omitempty"`
}
```

## AdmissionControl plugin: ResourceLimits

The **ResourceLimits** plug-in introspects all incoming admission requests. 

It makes decisions by introspecting the incoming object, current status, and enumerated constraints on **ResourceController**.

The following constraints are proposed as enforceable:

| Key | Type | Description |
| ------ | -------- | -------- |
| kubernetes.io/namespace/pods | int | Maximum number of pods per namespace |
| kubernetes.io/namespace/replicationControllers | int | Maximum number of replicationControllers per namespace |
| kubernetes.io/namespace/services | int | Maximum number of services per namespace |
| kubernetes.io/pods/containers | int | Maximum number of containers per pod |
| kubernetes.io/pods/containers/memory/max | int | Maximum amount of memory per container in a pod |
| kubernetes.io/pods/containers/memory/min | int | Minimum amount of memory per container in a pod |
| kubernetes.io/pods/containers/cpu/max | int | Maximum amount of CPU per container in a pod |
| kubernetes.io/pods/containers/cpu/min | int | Minimum amount of CPU per container in a pod |
| kubernetes.io/pods/cpu/max | int | Maximum CPU usage across all containers per pod |
| kubernetes.io/pods/cpu/min | int | Minimum CPU usage across all containers per pod |
| kubernetes.io/pods/memory/max | int | Maximum memory usage across all containers in pod |
| kubernetes.io/pods/memory/min | int | Minimum memory usage across all containers in pod |
| kubernetes.io/replicationController/replicas | int | Maximum number of replicas per replication controller |

If the incoming resource would cause a violation of the enumerated constraints, the request is denied with a set of
messages explaining what constraints were the source of the denial.

If a constraint is not enumerated by a **ResourceController** it is not tracked.

If a constraint spans resources, for example, it tracks the total number of some **kind** in a **namespace**,
the plug-in will post a **ResourceObservation** with the new incremented **Allocated*** usage for that constraint
using a compare-and-swap to ensure transactional integrity.  It is possible that the allocated usage will be persisted 
on a create operation, but the create can fail later in the request flow for some other unknown reason.  For this scenario,
the allocated usage will appear greater than the actual usage, the **kube-resource-controller** is responsible for 
synchronizing the observed allocated usage with actual usage. For delete requests, we will not decrement usage right away,
and will always rely on the **kube-resource-controller** to bring the observed value in line.  This is needed until
etcd supports atomic transactions across multiple resources.

## kube-apiserver

The server is updated to be aware of **ResourceController** and **ResourceObservation** objects.

The constraints are only enforced if the kube-apiserver is started as follows:

```
$ kube-apiserver -admission_control=ResourceLimits
```

## kube-resource-controller

This is a new daemon that observes **ResourceController** objects in the cluster, and updates their status with current cluster state.

The daemon runs a synchronization loop to do the following:

For each resource controller, perform the following steps:

  1. Reconcile **ResourceController.Status.Allowed** with **ResourceController.Spec.Allowed**
  2. Reconcile **ResourceController.Status.Allocated** with constraints enumerated in **ResourceController.Status.Allowed**
  3. If there was a change, atomically update **ResourceObservation** to force an update to **ResourceController.Status**

At step 1, allow the **kube-resource-controller** to support an administrator supplied override to enforce that what the
set of constraints desired to not conflict with any configured global constraints.  For example, do not let 
a **kubernetes.io/pods/memory/max** for any pod in any namespace exceed 8GB.  These global constraints could be supplied
via an alternate location in **etcd**, for example, a **ResourceController** in an **infra** namespace that is populated on
bootstrap.

At step 2, for fields that track total number of {kind} in a namespace, we query the cluster to ensure that the observed status
is in-line with the actual tracked status.  This is a stop-gap to etcd supporting transactions across resource updates to ensure
that when a resource is deleted, we can update the observed status.

## kubectl

kubectl is modified to support the **ResourceController** resource.

```kubectl describe``` provides a human-readable output of current constraints and usage in the namespace.

For example,

```
$ kubectl namespace myspace
$ kubectl create -f examples/resource-controller/resource-controller.json
$ kubectl get resourceControllers
NAME        LABELS
limits      <none>
$ kubectl describe resourceController limits
Name:       limits
Key                         Enforced  Allocated
----                        -----     ----
Max pods                    15        13
Max replication controllers 2         2
Max services                5         0
Max containers per pod      2         0
Max replica size            10        0
...
```

## Scenario: How this works in practice

Admin user wants to impose resource constraints in namespace ```dev``` to enforce the following:

1. A pod cannot use more than 8GB of RAM
2. The namespace cannot run more than 100 pods at a time.

To enforce this constraint, the Admin does the following:

```
$ cat resource-controller.json
{
  "id": "limits",
  "kind": "ResourceController",
  "apiVersion": "v1beta1",
  "spec": {
    "allowed": {
      "kubernetes.io/namespace/pods": 100,
      "kubernetes.io/pods/memory/max": 8000,
    }
  },
  "labels": {}
}
$ kubectl namespace dev
$ kubectl create -f resource-controller.json
```

The **kube-resource-controller** sees that a new **ResourceController** resource was created, and updates its
status with the current observations in the namespace.

The Admin describes the resource controller to see the current status:

```
$ kubectl describe resourceController limits
Name:       limits
Key                         Enforced  Allocated
----                        -----     ----
Max pods                    100       50
Max memory per pod          8000      4000
````

The Admin sees that the current ```dev``` namespace is using 50 pods, and the largest pod consumes 4GB of RAM.

The Developer that uses this namespace uses the system until he discovers he has exceeded his limits:

```
$ kubectl namespace dev
$ kubectl create -f pod.json
Unable to create pod.  You have exceeded your max pods in the namespace of 100.
```

or 

```
$ kubectl namespace dev
$ kubectl create -f pod.json
Unable to create pod.  It exceeds the max memory usage per pod of 8000 MB.
```

The Developer can observe his constraints as appropriate:
```
$ kubectl describe resourceController limits
Name:       limits
Key                         Enforced  Allocated
----                        -----     ----
Max pods                    100       100
Max memory per pod          8000      4000
````

And as a consequence reduce his current number of running pods, or memory requirements of the pod to proceed.
Or he could contact the Admin for his namespace to allocate him more resources.

