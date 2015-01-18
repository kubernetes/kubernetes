# Kubernetes User Documentation

The Kubernetes API currently manages 3 main resources:
* [pods](pods.md)
* [replication controllers](replication-controller.md)
* [services](services.md)

In Kubernetes, rather than individual containers, _pods_ are the smallest deployable units that can be created, scheduled, and managed. Singleton pods can be created directly, and sets of pods may created, maintained, and scaled using replication controllers.  Services create load-balanced targets for sets of pods.

Kubernetes supports a unique [networking model](networking.md). Kubernetes encourages a flat address and does not dynamically allocate ports, instead allowing users to select whichever ports are convenient for them. To achieve this, it allocates an IP address for each pod and each service. Services provide stable addresses and [DNS names](dns.md) for clients to connect to, even as serving pods are replaced by new pods on new hosts.

Each resource has a map of key-value [labels](labels.md). Individual labels are used to specify identifying metadata that can be used to define sets of resources by specifying required labels. 

Each resource also has a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object, called [annotations](annotations.md).

Each resource is created within a specific [namespace](namespaces.md), a default one if unspecified.

Other details:

* [API](api-conventions.md)
* [Client libraries](client-libraries.md)
* [Command-line interface](cli.md)
* [UI](ux.md)
* [Images and registries](images.md)
* [Container environment](container-environment.md)
* [Logging](logging.md)
* Monitoring using [CAdvisor](https://github.com/google/cadvisor) and [Heapster](https://github.com/GoogleCloudPlatform/heapster)
