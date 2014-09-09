# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers in Kubernetes.  Described in [GitHub issue #199](https://github.com/GoogleCloudPlatform/kubernetes/issues/199).


## Definitions

UID
: A non-empty, opaque, system-generated value guaranteed to be unique in time and space; intended to distinguish between historical occurrences of similar entities.

Name
: A non-empty string guaranteed to be unique within a given scope at a particular time; used in resource URLs; provided by clients at creation time and encouraged to be human friendly; intended to facilitate creation idempotence and space-uniqueness of singleton objects, distinguish distinct entities, and reference particular entities across operations.

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) label (DNS_LABEL)
: An alphanumeric (a-z, A-Z, and 0-9) string, with a maximum length of 63 characters, with the '-' character allowed anywhere except the first or last character, suitable for use as a hostname or segment in a domain name

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) subdomain (DNS_SUBDOMAIN)
: One or more rfc1035/rfc1123 labels separated by '.' with a maximum length of 253 characters

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) universally unique identifier (UUID)
: A 128 bit generated value that is extremely unlikely to collide across time and space and requires no central coordination


## Objectives for names and UIDs

1. Uniquely identify (via a UID) an object across space and time

2. Uniquely name (via a name) an object across space

3. Provide human-friendly names in API operations and/or configuration files

4. Allow idempotent creation of API resources (#148) and enforcement of space-uniqueness of singleton objects

5. Allow DNS names to be automatically generated for some objects


## General design

1. When an object is created via an api, a Name string (a DNS_SUBDOMAIN) must be provided.
   1. must be non-empty and unique within the apiserver
   2. enables idempotent and space-unique creation
      1. generating random names will defeat idempotentcy
   3. parts of the system (e.g. replication controller) may join strings (e.g. a base name and a random suffix) to create a unique Name
   Example: "guestbook.user"
   Example: "backend-x4eb1"

FIXME: final debate on having master default a name. Alternative: set "autosetName"=true

2. Upon acceptance of an object via an api, the object is assigned a UID (a UUID).
   1. must be non-empty and unique across space and time
   Example: "01234567-89ab-cdef-0123-456789abcdef"


## Case study: Scheduling a pod

Pods can be placed onto a particular node in a number of ways.  This case
study demonstrates how the above design can be applied to satisfy the
objectives.

### A pod scheduled by a user through the apiserver

1. A user submits a pod named "guestbook" to the apiserver.

2. The apiserver validates the input.
   1. The pod name must be space-unique within the apiserver.
   2. Each container within the pod has a name which must be space-unique within the pod.

3. The pod is accepted and a UID is assigned.

4. The pod is bound to a node.
   1. The kubelet on the node is passed the pod's UID and name.

5. Kubelet validates the input.

6. Kubelet further namespaces the pod name with information about the source of the pod.
   1. E.g. Namespace="api.k8s.example.com"

7. Kubelet runs the pod.
   1. Each container is started up with enough metadata to distinguish the pod from whence it came.
   2. Each attempt to run a container is assigned a UID (a string) that is unique across time.
      1. This may correspond to Docker's container ID.

### A pod placed by a config file on the node

1. A config file is stored on the node, containing a pod named "cadvisor" with no UID.

2. Kubelet validates the input.
   1. Since UID is not provided, kubelet generates one.

3. Kubelet further namespaces the pod name with information about the source of the pod.
   1. The generated namespace should be deterministic and cluster-unique for the source.
      1. E.g. a hash of the hostname and file path.
   2. E.g. Namespace="file-f4231812554558a718a01ca942782d81"

4. Kubelet runs the pod.
   1. Each container is started up with enough metadata to distinguish the pod from whence it came.
   2. Each attempt to run a container is assigned a UID (a string) that is unique across time.
      1. This may correspond to Docker's container ID.
