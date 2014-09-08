# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers and names in Kubernetes.  Described in [GitHub issue #199](https://github.com/GoogleCloudPlatform/kubernetes/issues/199).

FIXME: Should this be more agnostic to resource type, and talk about pod as a particular case?


## Definitions

identifier
: An opaque machine generated value guaranteed to be unique in time and space

name
: A human-friendly string intended to help an end user distinguish between similar but distinct entities

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) label (DNS_LABEL)
: An alphanumeric (a-z, A-Z, and 0-9) string, with a maximum length of 63 characters, with the '-' character allowed anywhere except the first or last character, suitable for use as a hostname or segment in a domain name

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) subdomain (DNS_SUBDOMAIN)
: One or more rfc1035/rfc1123 labels separated by '.' with a maximum length of 253 characters

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) universally unique identifier (UUID)
: A 128 bit generated value that is extremely unlikely to collide across time and space and requires no central coordination


## Objectives for names and identifiers

1) Uniquely identify an instance of a pod

2) Uniquely identify a container within a pod

3) Uniquely identify a single execution of a container

4) The structure of a pod specification should stay largely the same throughout the entire system

5) Provide human-friendly, memorable, semantically meaningful, names in the API

6) Provide predictable container and pod names in operations and/or configuration files

7) Allow idempotent creation of API resources (#148)


## Design

1) Each apiserver must be assigned a Namespace string (a DNS_SUBDOMAIN).
   1) must be non-empty and unique across all apiservers that share minions
   Example: "k8s.example.com"

2) When a pod is created on an apiserver, a Name string (a DNS_SUBDOMAIN) must be provided.
   1) must be non-empty and unique within the apiserver's Namespace
   2) provides idempotent creation
   3) other parts of the system (e.g. replication controller) may append to Name
   Example: "guestbook.user"
   Example: "foobar-x4eb1"

FIXME: final debate on having master default a name. Alternative: set "autosetName"=true
FIXME: how long can <name>+<namespace> be?  We previously had FullName, making it the apiserver's problem to truncate long names to DNS_DOMAIN len.

3) Upon acceptance at the apiserver, a pod is assigned an ID (a UUID).
   1) must be non-empty and unique across space and time
   Example: "01234567-89ab-cdef-0123-456789abcdef"

4) Each container within a pod must have a Name string (a DNS_LABEL).
   1) must be non-empty and unique within the pod
   Example: "frontend"

5) When a pod is bound to a node, the node is told the pod's ID.
   1) if not provided, the kubelet will generate one
   2) provides for pods from config files

6) When a pod is bound to a node, the node is told the pod's Namespace, and Name.
   1) if Namespace is not provided, the kubelet will generate one
   2) generated Namespaces must be deterministic
   3) provides a cluster-wide space-unique name
   Example: Namespace="k8s.example.com" Name="guestbook.user"
   Example: Namespace="file-f4231812554558a718a01ca942782d81" Name="cadvisor"

7) Each run of a container within a pod will be assigned an AttemptID (string) that is unique across time.
   1) corresponds to Docker's container ID
   Example: "77af4d6b9913e693e8d0b4b294fa62ade6054e6b2f1ffb617ac955dd63fb0182"
