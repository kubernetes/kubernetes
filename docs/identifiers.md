# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers in Kubernetes.  Described in [GitHub issue #199](https://github.com/GoogleCloudPlatform/kubernetes/issues/199).


## Definitions

uid
: An opaque system-generated value guaranteed to be unique in time and space; intended to distinguish between historical occurrences of similar entities.

name
: A string guaranteed to be unique within a given scope at a particular time; used in resource URLs; provided by clients at creation time and encouraged to be human friendly; intended to facilitate creation idempotence and space-uniqueness of singleton objects, distinguish distinct entities, and reference particular entities across operations.

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) label (DNS_LABEL)
: An alphanumeric (a-z, A-Z, and 0-9) string, with a maximum length of 63 characters, with the '-' character allowed anywhere except the first or last character, suitable for use as a hostname or segment in a domain name

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) subdomain (DNS_SUBDOMAIN)
: One or more rfc1035/rfc1123 labels separated by '.' with a maximum length of 253 characters

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) universally unique identifier (UUID)
: A 128 bit generated value that is extremely unlikely to collide across time and space and requires no central coordination


## Objectives for names and uids

1) Uniquely identify (via a uid) an object across space and time

2) Uniquely name (via a Name) an object across space

3) Provide human-friendly names in API operations and/or configuration files

4) Allow idempotent creation of API resources (#148) and enforcement of space-uniqueness of singleton objects

5) Allow DNS names to be automatically generated for some objects


FIXME: Should this be more agnostic to resource type, and talk about pod as a particular case?
## Design

1) When an object is created on an apiserver, a Name string (a DNS_SUBDOMAIN) must be provided.
   1) must be non-empty and unique within the apiserver
   2) enables idempotent and space-unique creation
      1) generating random names will defeat idempotentcy
   3) other parts of the system (e.g. replication controller) may join strings (e.g. a base name and a random suffic) to create a unique Name
   Example: "guestbook.user"
   Example: "backend-x4eb1"

FIXME: final debate on having master default a name. Alternative: set "autosetName"=true

2) Upon acceptance at the apiserver, a pod is assigned a uid (a UUID).
   1) must be non-empty and unique across space and time
   Example: "01234567-89ab-cdef-0123-456789abcdef"

3) Each container within a pod must have a Name string (a DNS_LABEL).
   1) must be non-empty and unique within the pod
   Example: "frontend"

4) When a pod is bound to a node, the node is told the pod's uid.
   1) if not provided, the kubelet will generate one
   2) provides for pods from node-local config files

6) When a pod is bound to a node, the node is told the pod's Name.
   1) kubelet will namespace pods from distinct sources (e.g. files vs apiserver)
   2) namespaces must be deterministic
   3) provides a cluster-wide space-unique name
   Example: Namespace="k8s.example.com" Name="guestbook.user"
   Example: Namespace="k8s.example.com" Name="backend-x4eb1"
   Example: Namespace="file-f4231812554558a718a01ca942782d81" Name="cadvisor"

7) Each run of a container within a pod will be assigned an AttemptID (string) that is unique across time.
   1) corresponds to Docker's container ID
   Example: "77af4d6b9913e693e8d0b4b294fa62ade6054e6b2f1ffb617ac955dd63fb0182"
