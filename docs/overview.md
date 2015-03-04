# Kubernetes User Documentation

Kubernetes is an open source orchestration system for containers (currently Docker containers). It schedules containers onto nodes in a compute cluster and actively manages these containers to ensure that the state of the cluster continually matches the user's intentions.

In Kubernetes, containers run inside [pods](pods.md), which are an abstraction used to group one or more cooperating containers. Users can create a singleton pod, but the usual deployment pattern in Kubernetes is to define a pod and then to ask the system to create multiple identical replicas of that pod. For each pod the system finds a machine that is healthy and has sufficiently availability capacity, and starts up the corresponding container(s) there.
The replicated set of pods might constitute an entire application, a micro-service, or one layer in a multi-tier application. 

Kubernetes supports a unique [networking model](networking.md). Kubernetes encourages a flat address and does not dynamically allocate ports, instead allowing users to select whichever ports are convenient for them. To achieve this, it allocates an IP address for each pod.

Every resource in Kubernetes, such as a pod, is identified by a URI and has a UID. Important components of the URI are the kind of object (e.g. pod), the object’s name, and the object’s [namespace](namespaces.md). Every name is unique within its namespace, and in contexts where an object name is provided without a namespace, it is assumed to be in the default namespace. UID is unique across time and space.

Frequently it is useful to refer to a set of pods, for example to limit the set of pods on which a mutating operation should be performed, or that should be queried for status. As a general mechanism, users can attach to most Kubernetes API objects arbitrary key-value pairs called [labels](labels.md), and then use a set of label selectors (key-value queries over labels) to constrain the target of API operations. Each resource also has a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object, called [annotations](annotations.md). 

Although users can create and manage pods directly, system management is drastically simplified by delegating that responsibility to [replication controllers](replication-controller.md). A replication controller is an API object that defines a pod in terms of a template that is automatically instantiated some number of times (specified by the user). Once the pods are created, the system continually monitors their health and that of the machines they are running on; if a pod fails due to a software problem or machine failure, the replication controller automatically creates a new pod on a healthy machine, to maintain the set of pods at the desired replication level.

Modern Internet applications are commonly built by layering micro-services, for example a set of web front-ends talking to a distributed in-memory key-value store talking to a replicated storage service. To facilitate this architecture, Kubernetes offers the [service](services.md) abstraction, which provides a stable IP address and [DNS name](dns.md) that corresponds to a dynamic set of pods such as the set of pods constituting a micro-service. The set is defined using a label selector and thus can refer to any set of pods. When a container running in a Kubernetes pod connects to this address, the connection is forwarded by a local agent (called the kube proxy) running on the source machine, to one of the corresponding back-end containers. The exact back-end is chosen using a round-robin policy to balance load. The kube proxy takes care of tracking the dynamic set of back-ends as pods are replaced by new pods on new hosts, so that the service IP address (and DNS name) never changes.

Other details:

* [API](api-conventions.md)
* [Client libraries](client-libraries.md)
* [Command-line interface](kubectl.md)
* [UI](ux.md)
* [Images and registries](images.md)
* [Container environment](container-environment.md)
* [Logging](logging.md)
* Monitoring using [CAdvisor](https://github.com/google/cadvisor) and [Heapster](https://github.com/GoogleCloudPlatform/heapster)

