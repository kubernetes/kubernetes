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
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/overview.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Overview

Kubernetes is an open-source system for managing containerized applications across multiple hosts in a cluster. Kubernetes is intended to make deploying containerized/microservice-based applications easy but powerful.

Kubernetes provides mechanisms for application deployment, scheduling, updating, maintenance, and scaling. A key feature of Kubernetes is that it actively manages the containers to ensure that the state of the cluster continually matches the user's intentions. An operations user should be able to launch a micro-service, letting the scheduler find the right placement. We also want to improve the tools and experience for how users can roll-out applications through patterns like canary deployments.

Kubernetes supports [Docker](http://www.docker.io) and [Rocket](https://coreos.com/blog/rocket/) containers, and other container image formats and container runtimes will be supported in the future.

While Kubernetes currently focuses on continuously-running stateless (e.g. web server or in-memory object cache) and "cloud native" stateful applications (e.g. NoSQL datastores), in the near future it will support all the other workload types commonly found in production cluster environments, such as batch, stream processing, and traditional databases.

In Kubernetes, all containers run inside [pods](pods.md). A pod can host a single container, or multiple cooperating containers; in the latter case, the containers in the pod are guaranteed to be co-located on the same machine and can share resources. A pod can also contain zero or more [volumes](volumes.md), which are directories that are private to a container or shared across containers in a pod. For each pod the user creates, the system finds a machine that is healthy and that has sufficient available capacity, and starts up the corresponding container(s) there. If a container fails it can be automatically restarted by Kubernetes' node agent, called the Kubelet. But if the pod or its machine fails, it is not automatically moved or restarted unless the user also defines a [replication controller](replication-controller.md), which we discuss next.

Users can create and manage pods themselves, but Kubernetes drastically simplifies system management by allowing users to delegate two common pod-related activities: deploying multiple pod replicas based on the same pod configuration, and creating replacement pods when a pod or its machine fails. The Kubernetes API object that manages these behaviors is called a [replication controller](replication-controller.md). It defines a pod in terms of a template, that the system then instantiates as some number of pods (specified by the user). The replicated set of pods might constitute an entire application, a micro-service, or one layer in a multi-tier application. Once the pods are created, the system continually monitors their health and that of the machines they are running on; if a pod fails due to a software problem or machine failure, the replication controller automatically creates a new pod on a healthy machine, to maintain the set of pods at the desired replication level. Multiple pods from the same or different applications can share the same machine. Note that a replication controller is needed even in the case of a single non-replicated pod if the user wants it to be re-created when it or its machine fails.

Frequently it is useful to refer to a set of pods, for example to limit the set of pods on which a mutating operation should be performed, or that should be queried for status. As a general mechanism, users can attach to most Kubernetes API objects arbitrary key-value pairs called [labels](labels.md), and then use a set of label selectors (key-value queries over labels) to constrain the target of API operations. Each resource also has a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object, called [annotations](annotations.md).

Kubernetes supports a unique [networking model](../admin/networking.md). Kubernetes encourages a flat address space and does not dynamically allocate ports, instead allowing users to select whichever ports are convenient for them. To achieve this, it allocates an IP address for each pod.

Modern Internet applications are commonly built by layering micro-services, for example a set of web front-ends talking to a distributed in-memory key-value store talking to a replicated storage service. To facilitate this architecture, Kubernetes offers the [service](services.md) abstraction, which provides a stable IP address and [DNS name](../admin/dns.md) that corresponds to a dynamic set of pods such as the set of pods constituting a micro-service. The set is defined using a label selector and thus can refer to any set of pods. When a container running in a Kubernetes pod connects to this address, the connection is forwarded by a local agent (called the kube proxy) running on the source machine, to one of the corresponding back-end containers. The exact back-end is chosen using a round-robin policy to balance load. The kube proxy takes care of tracking the dynamic set of back-ends as pods are replaced by new pods on new hosts, so that the service IP address (and DNS name) never changes.

Every resource in Kubernetes, such as a pod, is identified by a URI and has a UID. Important components of the URI are the kind of object (e.g. pod), the object’s name, and the object’s [namespace](namespaces.md). For a certain object kind, every name is unique within its namespace. In contexts where an object name is provided without a namespace, it is assumed to be in the default namespace. UID is unique across time and space.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/overview.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
