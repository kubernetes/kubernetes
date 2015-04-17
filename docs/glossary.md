
# Glossary and Concept Index

**Authorization**
:Kubernetes does not currently have an authorization system.  Anyone with the cluster password can do anything.  We plan
to add sophisticated authorization, and to make it pluggable.  See the [access control design doc](./design/access.md) and
[this issue](https://github.com/GoogleCloudPlatform/kubernetes/issues/1430).

**Annotation**
: A key/value pair that can hold large (compared to a Label), and possibly not human-readable data.  Intended to store
non-identifying metadata associated with an object, such as provenance information.  Not indexed.

**Image**
: A [Docker Image](https://docs.docker.com/userguide/dockerimages/).  See [images](./images.md).

**Label**
: A key/value pair conveying user-defined identifying attributes of an object, and used to form sets of related objects, such as
pods which are replicas in a load-balanced service.  Not intended to hold large or non-human-readable data.  See [labels](./labels.md).

**Name**
: A user-provided name for an object.  See [identifiers](identifiers.md).

**Namespace**
: A namespace is like a prefix to the name of an object.  You can configure your client to use a particular namespace,
so you do not have to type it all the time. Namespaces allow multiple projects to prevent naming collisions between unrelated teams.

**Pod**
: A collection of containers which will be scheduled onto the same node, which share and an IP and port space, and which
can be created/destroyed together.  See [pods](./pods.md).

**Replication Controller**
: A _replication controller_ ensures that a specified number of pod "replicas" are running at any one time. Both allows
for easy scaling of replicated systems, and handles restarting of a Pod when the machine it is on reboots or otherwise fails.

**Resource**
: CPU, memory, and other things that a pod can request.   See [resources](resources.md).

**Secret**
: An object containing sensitive information, such as authentication tokens, which can be made available to containers upon request. See [secrets](secrets.md).

**Selector**
: An expression that matches Labels.  Can identify related objects, such as pods which are replicas in a load-balanced
service.  See [labels](labels.md).

**Service**
: A load-balanced set of `pods` which can be accessed via a single stable IP address.  See [services](./services.md).

**UID**
: An identifier on all Kubernetes objects that is set by the Kubernetes API server.  Can be used to distinguish between historical
occurrences of same-Name objects.  See [identifiers](identifiers.md).

**Volume**
: A directory, possibly with some data in it, which is accessible to a Container as part of its filesystem.  Kubernetes
Volumes build upon [Docker Volumes](https://docs.docker.com/userguide/dockervolumes/), adding provisioning of the Volume
directory and/or device.  See [volumes](volumes.md).
