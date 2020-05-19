{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- A Kubernetes API has 2 parts - a Resource Type and a Controller
- Resources are objects declared as json or yaml and written to a cluster
- Controllers asynchronously actuate Resources after they are stored
{% endpanel %}

# Kubernetes Resources and Controllers Overview

This section provides background on the Kubernetes Resource model.  This information
is also available at the [kubernetes.io](https://kubernetes.io/docs/home/) docs site.

For more information on Kubernetes Resources see: [kubernetes.io Concepts](https://kubernetes.io/docs/concepts/).

## Resources

Instances of Kubernetes objects (e.g. Deployment, Services, Namespaces, etc)
are called **Resources**.

Resources which run containers are referred to as **Workloads**.

Examples of Workloads:

- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/)
- [CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [DaemonSets](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) 


**Users work with Resource APIs by declaring them in files which are then Applied to a Kubernetes
cluster.  These declarative files are called Resource Config.**

Resource Config is *Applied* (declarative Create/Update/Delete) to a Kubernetes cluster using
tools such as Kubectl, and then actuated by a *Controller*.

Resources are uniquely identified:

- **apiVersion** (API Type Group and Version)
- **kind** (API Type Name)
- **metadata.namespace** (Instance namespace)
- **metadata.name** (Instance name)

{% panel style="warning", title="Default Namespace" %}
If namespace is omitted from the Resource Config, the *default* namespace is used.  Users
should almost always explicitly specify the namespace for their Application using a
`kustomization.yaml`.
{% endpanel %}

{% method %}

### Resources Structure

Resources have the following components.

**TypeMeta:** Resource Type **apiVersion** and **kind**.

**ObjectMeta:** Resource **name** and **namespace** + other metadata (labels, annotations, etc).

**Spec:** the desired state of the Resource - intended state the user provides to the cluster.

**Status:** the observed state of the object - recorded state the cluster provides to the user.

Resource Config written by the user omits the Status field.

**Example Deployment Resource Config**
{% sample lang="yaml" %}

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.15.4
```
{% endmethod %}

{% panel style="info", title="Spec and Status" %}
Resources such as ConfigMaps and Secrets do not have a Status,
and as a result their Spec is implicit (i.e. they don't have a spec field).
{% endpanel %}

## Controllers

Controllers actuate Kubernetes APIs.  They observe the state of the system and look for
changes either to desired state of Resources (create, update, delete) or the system
(Pod or Node dies).

Controllers then make changes to the cluster to fulfill the intent specified by the user
(e.g. in Resource Config) or automation (e.g. changes from Autoscalers).

**Example:** After a user creates a Deployment, the Deployment Controller will see
that the Deployment exists and verify that the corresponding ReplicaSet it expects
to find exists.  The Controller will see that the ReplicaSet does not exist and will
create one.

{% panel style="warning", title="Asynchronous Actuation" %}
Because Controllers run asynchronously, issues such as a bad
Container Image or unschedulable Pods will not be present in the CRUD response.
Tooling must facilitate processes for watching the state of the system until changes are
completely actuated by Controllers.  Once the changes have been fully actuated such
that the desired state matches the observed state, the Resource is considered *Settled*.
{% endpanel %}

### Controller Structure

**Reconcile**

Controllers actuate Resources by reading the Resource they are Reconciling + related Resources,
such as those that they create and delete.

**Controllers *do not* Reconcile events, rather they Reconcile the expected
cluster state to the observed cluster state at the time Reconcile is run.**

1. Deployment Controller creates/deletes ReplicaSets
1. ReplicaSet Controller creates/deletes Pods
1. Scheduler (Controller) writes Nodes to Pods
1. Node (Controller) runs Containers specified in Pods on the Node

**Watch**

Controllers actuate Resources *after* they are written by Watching Resource Types, and then
triggering Reconciles from Events.  After a Resource is created/updated/deleted, Controllers
Watching the Resource Type will receive a notification that the Resource has been changed,
and they will read the state of the system to see what has changed (instead of relying on
the Event for this information).

- Deployment Controller watches Deployments + ReplicaSets (+ Pods)
- ReplicaSet Controller watches ReplicaSets + Pods
- Scheduler (Controller) watches Pods
- Node (Controller) watches Pods (+ Secrets + ConfigMaps)

{% panel style="info", title="Level vs Edge Based Reconciliation" %}
Because Controllers don't respond to individual Events, but instead Reconcile the state
of the system at the time that Reconcile is run, **changes from several different events may be observed
and Reconciled together.**  This is referred to as a *Level Based* system, whereas a system that
responds to each event individually would be referred to as an *Edge Based* system.
{% endpanel %}

## Overview of Kubernetes Resource APIs

### Pods

Containers are run in [*Pods*](https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/) which are
scheduled to run on *Nodes* (i.e. worker machines) in a cluster.

Pods run a *single replica* of an Application and provide:

- Compute Resources (cpu, memory, disk)
- Environment Variables
- Readiness and Health Checking
- Network (IP address shared by containers in the Pod)
- Mounting Shared Configuration and Secrets
- Mounting Storage Volumes
- Initialization

{% panel style="warning", title="Multi Container Pods" %}
Multiple replicas of an Application should be created using a Workload API to manage
creation and deletion of Pod replicas using a PodTemplate.

In some cases a Pod may contain multiple Containers forming a single instance of an Application.  These
containers may coordinate with one another through shared network (IP) and storage.
{% endpanel %}

### Workloads

Pods are typically managed by higher level abstractions that handle concerns such as
replication, identity, persistent storage, custom scheduling, rolling updates, etc.

The most common out-of-the-box Workload APIs (manage Pods) are:

- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) (Stateless Applications)
  - replication + rollouts
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/) (Stateful Applications)
  - replication + rollouts + persistent storage + identity
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/) (Batch Work)
  - run to completion
- [CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/) (Scheduled Batch Work)
  - scheduled run to completion
- [DaemonSets](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/) (Per-Machine)
  - per-Node scheduling

{% panel style="success", title="API Abstraction Layers" %}
High-level Workload APIs may manage lower-level Workload APIs instead of directly managing Pods
(e.g. Deployments manage ReplicaSets).
{% endpanel %}

### Service Discovery and Load Balancing

Service discovery and Load Balancing may be managed by a *Service* object.  Services provide a single
virtual IP address and dns name load balanced to a collection of Pods matching Labels.

{% panel style="info", title="Internal vs External Services" %}
- [Services Resources](https://kubernetes.io/docs/concepts/services-networking/service/)
  (L4) may expose Pods internally within a cluster or externally through an HA proxy.
- [Ingress Resources](https://kubernetes.io/docs/concepts/services-networking/ingress/) (L7)
  may expose URI endpoints and route them to Services.
{% endpanel %}

### Configuration and Secrets

Shared Configuration and Secret data may be provided by ConfigMaps and Secrets.  This allows
Environment Variables, Command Line Arguments and Files to be loosely injected into
the Pods and Containers that consume them.

{% panel style="info", title="ConfigMaps vs Secrets" %}
- [ConfigMaps](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/)
  are for providing non-sensitive data to Pods.
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
  are for providing sensitive data to Pods.
{% endpanel %}
