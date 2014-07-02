# Identifiers and Names in Kubernetes

A summarization of the goals and recommendations for identifiers and names in Kubernetes.  Described in [GitHub issue #199](https://github.com/GoogleCloudPlatform/kubernetes/issues/199).

## Definitions

identifier
: an opaque machine generated value guaranteed to be unique in a certain space

name
: a human readable string intended to help an end user distinguish between similar but distinct entities

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) label (DNS_LABEL)
: An alphanumeric (a-z, A-Z, and 0-9) string less than 64 characters, with the '-' character allowed anywhere except the first or last character, suitable for use as a hostname or segment in a domain name.

[rfc1035](http://www.ietf.org/rfc/rfc1035.txt)/[rfc1123](http://www.ietf.org/rfc/rfc1123.txt) subdomain (DNS_SUBDOMAIN)
: One or more rfc1035/rfc1123 labels separated by '.' with a maximum length of 255 characters

namespace string (NAMESPACE)
: An rfc1035/rfc1123 subdomain no longer than 191 characters (255-63-1)

source namespace string
: The namespace string of a source of pod definitions on a host

[rfc4122](http://www.ietf.org/rfc/rfc4122.txt) universally unique identifier (UUID)
: A 128 bit generated value that is extremely unilkely to collide across time and space and requires no central coordination

pod unique name
: the combination of a pod's source namespace string and name string on a host

pod unique identifier
: the identifier associated with a single execution of a pod on a host, which changes on each restart.  Must be a UUID


## Objectives for names and identifiers

1) Uniquely identify an instance of a pod on the apiserver and on the kubelet

2) Uniquely identify an instance of a container within a pod on the apiserver and on the kubelet

3) Uniquely identify a single execution of a container in time for logging or reporting

4) The structure of a pod specification should stay largely the same throughout the entire system

5) Provide human-friendly, memorable, semantically meaningful, short-ish references in container and pod operations

6) Provide predictable container and pod references in operations and/or configuration files

7) Allow idempotent creation of API resources (#148)

8) Allow DNS names to be automatically generated for individual containers or pods (#146)


## Implications

1) Each container name within a container manifest must be unique within that manifest

2) Each pod instance on the apiserver must have a unique identifier across space and time (UUID)
   1) The apiserver may set this identifier if not specified by a client
   2) This identifier will persist even if moved across hosts

3) Each pod instance on the apiserver must have a name string which is human-friendly, dns-friendly (DNS_LABEL), and unique in the apiserver space
   1) The apiserver may set this name string if not specified by a client

4) Each apiserver must have a configured namespace string (NAMESPACE) that is unique across all apiservers that share its configured minions

5) Each source of pod configuration to a kubelet must have a source namespace string (NAMESPACE) that is unique across all sources available to that kubelet

6) All pod instances on a host must have a name string which is human-friendly, dns-friendly, and unique per namespace string (DNS_LABEL)

7) The combination of the name string and source namespace string on a kubelet must be unique and is referred to as the pod unique name

8) When starting an instance of a pod on a kubelet the first time, a new pod unique identifier (UUID) should be assigned to that pod instance
   1) If that pod is restarted, it must retain the pod unique identifier it previously had
   2) If the pod is stopped and a new instance with the same pod unique name is started, it must be assigned a new pod unique identifier

9) The kubelet should use the pod unique name and pod unique identifier to produce a Docker container name (--name)

