# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers and names in Kubernetes.  Described in [GitHub issue #199](https://github.com/GoogleCloudPlatform/kubernetes/issues/199).

## Definitions

identifier
: An opaque machine generated value guaranteed to be unique in a certain space

name
: A human readable string intended to help an end user distinguish between similar but distinct entities

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) label (DNS_LABEL)
: An alphanumeric (a-z, A-Z, and 0-9) string, with a maximum length of 63 characters, with the '-' character allowed anywhere except the first or last character, suitable for use as a hostname or segment in a domain name

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) subdomain (DNS_SUBDOMAIN)
: One or more rfc1035/rfc1123 labels separated by '.' with a maximum length of 253 characters

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) universally unique identifier (UUID)
: A 128 bit generated value that is extremely unlikely to collide across time and space and requires no central coordination

## Objectives for names and identifiers

1) Uniquely identify an instance of a pod on the apiserver and on the kubelet

2) Uniquely identify an instance of a container within a pod on the apiserver and on the kubelet

3) Uniquely identify a single execution of a container in time for logging or reporting

4) The structure of a pod specification should stay largely the same throughout the entire system

5) Provide human-friendly, memorable, semantically meaningful, short-ish references in container and pod operations

6) Provide predictable container and pod references in operations and/or configuration files

7) Allow idempotent creation of API resources (#148)

8) Allow DNS names to be automatically generated for individual containers or pods (#146)


## Design

1) Each apiserver has a Namespace string (a DNS_SUBDOMAIN) that is unique across all apiservers that share its configured minions.
   Example: "k8s.example.com"

2) Each pod instance on an apiserver has a PodName string (a DNS_SUBDOMAIN) which is and unique within the Namespace.
   1) If not specified by the client, the apiserver will assign this identifier
   Example: "guestbook.user"

3) Each pod instance on an apiserver has a PodFullName (a DNS_SUBDOMAIN) string which is derived from a combination of the Namespace and Name strings.
   1) If the joined Namespace and PodName is too long for a DNS_SUBDOMAIN, the apiserver must transform it to fit, while still being unique
   Example: "guestbook.user.k8s.example.com"

4) Each pod instance on an apiserver has a PodID (a UUID) that is unique across space and time
   1) If not specified by the client, the apiserver will assign this identifier
   2) This identifier will persist for the lifetime of the pod, even if the pod is stopped and started or moved across hosts
   Example: "01234567-89ab-cdef-0123-456789abcdef"

5) Each container within a pod has a ContainerName string (a DNS_LABEL) that is unique within that pod
   1) This name must be specified by the client or the apiserver will reject the pod
   Example: "frontend"

6) Each pod instance on a kubelet has a PodNamespace string (a DNS_SUBDOMAIN)
   1) This corresponds to the apiserver's Namespace string
   2) If not specified, the kubelet will assign this name to a deterministic value which is likely to be unique across all sources on the host
   Example: "k8s.example.com"
   Example: "file-f4231812554558a718a01ca942782d81"

7) Each pod instance on a kubelet has a PodName string (a DNS_SUBDOMAIN) which is unique within the source Namespace
   1) This corresponds to the apiserver's PodName string
   2) If not specified, the kubelet will assign this name to a deterministic value
   Example: "frontend"

8) When starting an instance of a pod on a kubelet, a PodInstanceID (a UUID) will be assigned to that pod instance
   1) If not specified, the kubelet will assign this identifier
   2) If the pod is restarted, it must retain the PodInstanceID it previously had
   3) If the pod is stopped and a new instance with the same PodNamespace and PodName is started, it must be assigned a new PodInstanceID
   4) If the pod is moved across hosts, it must be assigned a new PodInstanceID
   Example: "01234567-89ab-cdef-0123-456789abcdef"

9) The kubelet may use the PodNamespace, PodName, PodID, and PodInstanceID to produce a docker container name (--name)
   Example: "01234567-89ab-cdef-0123-456789abcdef_frontend_k8s.example.com"

10) Each run of a container within a pod will be assigned a ContainerAttemptID (string) that is unique across time.
   1) This corresponds to Docker container IDs
   Example: "77af4d6b9913e693e8d0b4b294fa62ade6054e6b2f1ffb617ac955dd63fb0182"
