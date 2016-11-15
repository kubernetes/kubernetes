# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers in Kubernetes.
Described in GitHub issue [#199](http://issue.k8s.io/199).


## Definitions

`UID`: A non-empty, opaque, system-generated value guaranteed to be unique in time
and space; intended to distinguish between historical occurrences of similar
entities.

`Name`: A non-empty string guaranteed to be unique within a given scope at a
particular time; used in resource URLs; provided by clients at creation time and
encouraged to be human friendly; intended to facilitate creation idempotence and
space-uniqueness of singleton objects, distinguish distinct entities, and
reference particular entities across operations.

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) `label` (DNS_LABEL):
An alphanumeric (a-z, and 0-9) string, with a maximum length of 63 characters,
with the '-' character allowed anywhere except the first or last character,
suitable for use as a hostname or segment in a domain name.

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) `subdomain` (DNS_SUBDOMAIN):
One or more lowercase rfc1035/rfc1123 labels separated by '.' with a maximum
length of 253 characters.

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) `universally unique identifier` (UUID):
A 128 bit generated value that is extremely unlikely to collide across time and
space and requires no central coordination.

[rfc6335](https://tools.ietf.org/rfc/rfc6335.txt) `port name` (IANA_SVC_NAME):
An alphanumeric (a-z, and 0-9) string, with a maximum length of 15 characters,
with the '-' character allowed anywhere except the first or the last character
or adjacent to another '-' character, it must contain at least a (a-z)
character.

## Objectives for names and UIDs

1. Uniquely identify (via a UID) an object across space and time.
2. Uniquely name (via a name) an object across space.
3. Provide human-friendly names in API operations and/or configuration files.
4. Allow idempotent creation of API resources (#148) and enforcement of
space-uniqueness of singleton objects.
5. Allow DNS names to be automatically generated for some objects.


## General design

1. When an object is created via an API, a Name string (a DNS_SUBDOMAIN) must
be specified. Name must be non-empty and unique within the apiserver. This
enables idempotent and space-unique creation operations. Parts of the system
(e.g. replication controller) may join strings (e.g. a base name and a random
suffix) to create a unique Name. For situations where generating a name is
impractical, some or all objects may support a param to auto-generate a name.
Generating random names will defeat idempotency.
   * Examples: "guestbook.user", "backend-x4eb1"
2. When an object is created via an API, a Namespace string (a DNS_SUBDOMAIN?
format TBD via #1114) may be specified. Depending on the API receiver,
namespaces might be validated (e.g. apiserver might ensure that the namespace
actually exists). If a namespace is not specified, one will be assigned by the
API receiver. This assignment policy might vary across API receivers (e.g.
apiserver might have a default, kubelet might generate something semi-random).
   * Example: "api.k8s.example.com"
3. Upon acceptance of an object via an API, the object is assigned a UID
(a UUID). UID must be non-empty and unique across space and time.
   * Example: "01234567-89ab-cdef-0123-456789abcdef"

## Case study: Scheduling a pod

Pods can be placed onto a particular node in a number of ways. This case study
demonstrates how the above design can be applied to satisfy the objectives.

### A pod scheduled by a user through the apiserver

1. A user submits a pod with Namespace="" and Name="guestbook" to the apiserver.
2. The apiserver validates the input.
   1. A default Namespace is assigned.
   2. The pod name must be space-unique within the Namespace.
   3. Each container within the pod has a name which must be space-unique within
the pod.
3. The pod is accepted.
   1. A new UID is assigned.
4. The pod is bound to a node.
   1. The kubelet on the node is passed the pod's UID, Namespace, and Name.
5. Kubelet validates the input.
6. Kubelet runs the pod.
   1. Each container is started up with enough metadata to distinguish the pod
from whence it came.
   2. Each attempt to run a container is assigned a UID (a string) that is
unique across time. * This may correspond to Docker's container ID.

### A pod placed by a config file on the node

1. A config file is stored on the node, containing a pod with UID="",
Namespace="", and Name="cadvisor".
2. Kubelet validates the input.
   1. Since UID is not provided, kubelet generates one.
   2. Since Namespace is not provided, kubelet generates one.
      1. The generated namespace should be deterministic and cluster-unique for
the source, such as a hash of the hostname and file path.
         * E.g. Namespace="file-f4231812554558a718a01ca942782d81"
3. Kubelet runs the pod.
   1. Each container is started up with enough metadata to distinguish the pod
from whence it came.
   2. Each attempt to run a container is assigned a UID (a string) that is
unique across time.
      1. This may correspond to Docker's container ID.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/identifiers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
