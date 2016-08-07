<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/design/secrets.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

A proposal for the distribution of [secrets](../user-guide/secrets.md)
(passwords, keys, etc) to the Kubelet and to containers inside Kubernetes using
a custom [volume](../user-guide/volumes.md#secrets) type. See the
[secrets example](../user-guide/secrets/) for more information.

## Motivation

Secrets are needed in containers to access internal resources like the
Kubernetes master or external resources such as git repositories, databases,
etc. Users may also want behaviors in the kubelet that depend on secret data
(credentials for image pull from a docker registry) associated with pods.

Goals of this design:

1.  Describe a secret resource
2.  Define the various challenges attendant to managing secrets on the node
3.  Define a mechanism for consuming secrets in containers without modification

## Constraints and Assumptions

*  This design does not prescribe a method for storing secrets; storage of
secrets should be pluggable to accommodate different use-cases
*  Encryption of secret data and node security are orthogonal concerns
*  It is assumed that node and master are secure and that compromising their
security could also compromise secrets:
   *  If a node is compromised, the only secrets that could potentially be
exposed should be the secrets belonging to containers scheduled onto it
   *  If the master is compromised, all secrets in the cluster may be exposed
*  Secret rotation is an orthogonal concern, but it should be facilitated by
this proposal
*  A user who can consume a secret in a container can know the value of the
secret; secrets must be provisioned judiciously

## Use Cases

1.  As a user, I want to store secret artifacts for my applications and consume
them securely in containers, so that I can keep the configuration for my
applications separate from the images that use them:
    1.  As a cluster operator, I want to allow a pod to access the Kubernetes
master using a custom `.kubeconfig` file, so that I can securely reach the
master
    2.  As a cluster operator, I want to allow a pod to access a Docker registry
using credentials from a `.dockercfg` file, so that containers can push images
    3.  As a cluster operator, I want to allow a pod to access a git repository
using SSH keys, so that I can push to and fetch from the repository
2.  As a user, I want to allow containers to consume supplemental information
about services such as username and password which should be kept secret, so
that I can share secrets about a service amongst the containers in my
application securely
3.  As a user, I want to associate a pod with a `ServiceAccount` that consumes a
secret and have the kubelet implement some reserved behaviors based on the types
of secrets the service account consumes:
    1.  Use credentials for a docker registry to pull the pod's docker image
    2.  Present Kubernetes auth token to the pod or transparently decorate
traffic between the pod and master service
4.  As a user, I want to be able to indicate that a secret expires and for that
secret's value to be rotated once it expires, so that the system can help me
follow good practices

### Use-Case: Configuration artifacts

Many configuration files contain secrets intermixed with other configuration
information. For example, a user's application may contain a properties file
than contains database credentials, SaaS API tokens, etc. Users should be able
to consume configuration artifacts in their containers and be able to control
the path on the container's filesystems where the artifact will be presented.

### Use-Case: Metadata about services

Most pieces of information about how to use a service are secrets. For example,
a service that provides a MySQL database needs to provide the username,
password, and database name to consumers so that they can authenticate and use
the correct database. Containers in pods consuming the MySQL service would also
consume the secrets associated with the MySQL service.

### Use-Case: Secrets associated with service accounts

[Service Accounts](service_accounts.md) are proposed as a mechanism to decouple
capabilities and security contexts from individual human users. A
`ServiceAccount` contains references to some number of secrets. A `Pod` can
specify that it is associated with a `ServiceAccount`. Secrets should have a
`Type` field to allow the Kubelet and other system components to take action
based on the secret's type.

#### Example: service account consumes auth token secret

As an example, the service account proposal discusses service accounts consuming
secrets which contain Kubernetes auth tokens. When a Kubelet starts a pod
associated with a service account which consumes this type of secret, the
Kubelet may take a number of actions:

1.  Expose the secret in a `.kubernetes_auth` file in a well-known location in
the container's file system
2.  Configure that node's `kube-proxy` to decorate HTTP requests from that pod
to the `kubernetes-master` service with the auth token, e. g. by adding a header
to the request (see the [LOAS Daemon](http://issue.k8s.io/2209) proposal)

#### Example: service account consumes docker registry credentials

Another example use case is where a pod is associated with a secret containing
docker registry credentials. The Kubelet could use these credentials for the
docker pull to retrieve the image.

### Use-Case: Secret expiry and rotation

Rotation is considered a good practice for many types of secret data. It should
be possible to express that a secret has an expiry date; this would make it
possible to implement a system component that could regenerate expired secrets.
As an example, consider a component that rotates expired secrets. The rotator
could periodically regenerate the values for expired secrets of common types and
update their expiry dates.

## Deferral: Consuming secrets as environment variables

Some images will expect to receive configuration items as environment variables
instead of files. We should consider what the best way to allow this is; there
are a few different options:

1.  Force the user to adapt files into environment variables. Users can store
secrets that need to be presented as environment variables in a format that is
easy to consume from a shell:

        $ cat /etc/secrets/my-secret.txt
        export MY_SECRET_ENV=MY_SECRET_VALUE

    The user could `source` the file at `/etc/secrets/my-secret` prior to
executing the command for the image either inline in the command or in an init
script.

2.  Give secrets an attribute that allows users to express the intent that the
platform should generate the above syntax in the file used to present a secret.
The user could consume these files in the same manner as the above option.

3.  Give secrets attributes that allow the user to express that the secret
should be presented to the container as an environment variable. The container's
environment would contain the desired values and the software in the container
could use them without accommodation the command or setup script.

For our initial work, we will treat all secrets as files to narrow the problem
space. There will be a future proposal that handles exposing secrets as
environment variables.

## Flow analysis of secret data with respect to the API server

There are two fundamentally different use-cases for access to secrets:

1.  CRUD operations on secrets by their owners
2.  Read-only access to the secrets needed for a particular node by the kubelet

### Use-Case: CRUD operations by owners

In use cases for CRUD operations, the user experience for secrets should be no
different than for other API resources.

#### Data store backing the REST API

The data store backing the REST API should be pluggable because different
cluster operators will have different preferences for the central store of
secret data. Some possibilities for storage:

1.  An etcd collection alongside the storage for other API resources
2.  A collocated [HSM](http://en.wikipedia.org/wiki/Hardware_security_module)
3.  A secrets server like [Vault](https://www.vaultproject.io/) or
[Keywhiz](https://square.github.io/keywhiz/)
4.  An external datastore such as an external etcd, RDBMS, etc.

#### Size limit for secrets

There should be a size limit for secrets in order to:

1.  Prevent DOS attacks against the API server
2.  Allow kubelet implementations that prevent secret data from touching the
node's filesystem

The size limit should satisfy the following conditions:

1.  Large enough to store common artifact types (encryption keypairs,
certificates, small configuration files)
2.  Small enough to avoid large impact on node resource consumption (storage,
RAM for tmpfs, etc)

To begin discussion, we propose an initial value for this size limit of **1MB**.

#### Other limitations on secrets

Defining a policy for limitations on how a secret may be referenced by another
API resource and how constraints should be applied throughout the cluster is
tricky due to the number of variables involved:

1.  Should there be a maximum number of secrets a pod can reference via a
volume?
2.  Should there be a maximum number of secrets a service account can reference?
3.  Should there be a total maximum number of secrets a pod can reference via
its own spec and its associated service account?
4.  Should there be a total size limit on the amount of secret data consumed by
a pod?
5.  How will cluster operators want to be able to configure these limits?
6.  How will these limits impact API server validations?
7.  How will these limits affect scheduling?

For now, we will not implement validations around these limits.  Cluster
operators will decide how much node storage is allocated to secrets. It will be
the operator's responsibility to ensure that the allocated storage is sufficient
for the workload scheduled onto a node.

For now, kubelets will only attach secrets to api-sourced pods, and not file-
or http-sourced ones.  Doing so would:
  - confuse the secrets admission controller in the case of mirror pods.
  - create an apiserver-liveness dependency -- avoiding this dependency is a
main reason to use non-api-source pods.

### Use-Case: Kubelet read of secrets for node

The use-case where the kubelet reads secrets has several additional requirements:

1.  Kubelets should only be able to receive secret data which is required by
pods scheduled onto the kubelet's node
2.  Kubelets should have read-only access to secret data
3.  Secret data should not be transmitted over the wire insecurely
4.  Kubelets must ensure pods do not have access to each other's secrets

#### Read of secret data by the Kubelet

The Kubelet should only be allowed to read secrets which are consumed by pods
scheduled onto that Kubelet's node and their associated service accounts.
Authorization of the Kubelet to read this data would be delegated to an
authorization plugin and associated policy rule.

#### Secret data on the node: data at rest

Consideration must be given to whether secret data should be allowed to be at
rest on the node:

1.  If secret data is not allowed to be at rest, the size of secret data becomes
another draw on the node's RAM - should it affect scheduling?
2.  If secret data is allowed to be at rest, should it be encrypted?
    1.  If so, how should be this be done?
    2.  If not, what threats exist?  What types of secret are appropriate to
store this way?

For the sake of limiting complexity, we propose that initially secret data
should not be allowed to be at rest on a node; secret data should be stored on a
node-level tmpfs filesystem. This filesystem can be subdivided into directories
for use by the kubelet and by the volume plugin.

#### Secret data on the node: resource consumption

The Kubelet will be responsible for creating the per-node tmpfs file system for
secret storage. It is hard to make a prescriptive declaration about how much
storage is appropriate to reserve for secrets because different installations
will vary widely in available resources, desired pod to node density, overcommit
policy, and other operation dimensions. That being the case, we propose for
simplicity that the amount of secret storage be controlled by a new parameter to
the kubelet with a default value of **64MB**. It is the cluster operator's
responsibility to handle choosing the right storage size for their installation
and configuring their Kubelets correctly.

Configuring each Kubelet is not the ideal story for operator experience; it is
more intuitive that the cluster-wide storage size be readable from a central
configuration store like the one proposed in [#1553](http://issue.k8s.io/1553).
When such a store exists, the Kubelet could be modified to read this
configuration item from the store.

When the Kubelet is modified to advertise node resources (as proposed in
[#4441](http://issue.k8s.io/4441)), the capacity calculation
for available memory should factor in the potential size of the node-level tmpfs
in order to avoid memory overcommit on the node.

#### Secret data on the node: isolation

Every pod will have a [security context](security_context.md).
Secret data on the node should be isolated according to the security context of
the container. The Kubelet volume plugin API will be changed so that a volume
plugin receives the security context of a volume along with the volume spec.
This will allow volume plugins to implement setting the security context of
volumes they manage.

## Community work

Several proposals / upstream patches are notable as background for this
proposal:

1.  [Docker vault proposal](https://github.com/docker/docker/issues/10310)
2.  [Specification for image/container standardization based on volumes](https://github.com/docker/docker/issues/9277)
3.  [Kubernetes service account proposal](service_accounts.md)
4.  [Secrets proposal for docker (1)](https://github.com/docker/docker/pull/6075)
5.  [Secrets proposal for docker (2)](https://github.com/docker/docker/pull/6697)

## Proposed Design

We propose a new `Secret` resource which is mounted into containers with a new
volume type. Secret volumes will be handled by a volume plugin that does the
actual work of fetching the secret and storing it. Secrets contain multiple
pieces of data that are presented as different files within the secret volume
(example: SSH key pair).

In order to remove the burden from the end user in specifying every file that a
secret consists of, it should be possible to mount all files provided by a
secret with a single `VolumeMount` entry in the container specification.

### Secret API Resource

A new resource for secrets will be added to the API:

```go
type Secret struct {
    TypeMeta
    ObjectMeta

    // Data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN.
    // The serialized form of the secret data is a base64 encoded string,
    // representing the arbitrary (possibly non-string) data value here.
    Data map[string][]byte `json:"data,omitempty"`

    // Used to facilitate programmatic handling of secret data.
    Type SecretType `json:"type,omitempty"`
}

type SecretType string

const (
    SecretTypeOpaque              SecretType = "Opaque"                                 // Opaque (arbitrary data; default)
    SecretTypeServiceAccountToken SecretType = "kubernetes.io/service-account-token"    // Kubernetes auth token
    SecretTypeDockercfg           SecretType = "kubernetes.io/dockercfg"                // Docker registry auth
    // FUTURE: other type values
)

const MaxSecretSize = 1 * 1024 * 1024
```

A Secret can declare a type in order to provide type information to system
components that work with secrets. The default type is `opaque`, which
represents arbitrary user-owned data.

Secrets are validated against `MaxSecretSize`. The keys in the `Data` field must
be valid DNS subdomains.

A new REST API and registry interface will be added to accompany the `Secret`
resource. The default implementation of the registry will store `Secret`
information in etcd. Future registry implementations could store the `TypeMeta`
and `ObjectMeta` fields in etcd and store the secret data in another data store
entirely, or store the whole object in another data store.

#### Other validations related to secrets

Initially there will be no validations for the number of secrets a pod
references, or the number of secrets that can be associated with a service
account. These may be added in the future as the finer points of secrets and
resource allocation are fleshed out.

### Secret Volume Source

A new `SecretSource` type of volume source will be added to the `VolumeSource`
struct in the API:

```go
type VolumeSource struct {
    // Other fields omitted

    // SecretSource represents a secret that should be presented in a volume
    SecretSource *SecretSource `json:"secret"`
}

type SecretSource struct {
    Target ObjectReference
}
```

Secret volume sources are validated to ensure that the specified object
reference actually points to an object of type `Secret`.

In the future, the `SecretSource` will be extended to allow:

1.  Fine-grained control over which pieces of secret data are exposed in the
volume
2.  The paths and filenames for how secret data are exposed

### Secret Volume Plugin

A new Kubelet volume plugin will be added to handle volumes with a secret
source. This plugin will require access to the API server to retrieve secret
data and therefore the volume `Host` interface will have to change to expose a
client interface:

```go
type Host interface {
    // Other methods omitted

    // GetKubeClient returns a client interface
    GetKubeClient() client.Interface
}
```

The secret volume plugin will be responsible for:

1.  Returning a `volume.Mounter` implementation from `NewMounter` that:
    1.  Retrieves the secret data for the volume from the API server
    2.  Places the secret data onto the container's filesystem
    3.  Sets the correct security attributes for the volume based on the pod's
`SecurityContext`
2.  Returning a `volume.Unmounter` implementation from `NewUnmounter` that
cleans the volume from the container's filesystem

### Kubelet: Node-level secret storage

The Kubelet must be modified to accept a new parameter for the secret storage
size and to create a tmpfs file system of that size to store secret data. Rough
accounting of specific changes:

1.  The Kubelet should have a new field added called `secretStorageSize`; units
are megabytes
2.  `NewMainKubelet` should accept a value for secret storage size
3.  The Kubelet server should have a new flag added for secret storage size
4.  The Kubelet's `setupDataDirs` method should be changed to create the secret
storage

### Kubelet: New behaviors for secrets associated with service accounts

For use-cases where the Kubelet's behavior is affected by the secrets associated
with a pod's `ServiceAccount`, the Kubelet will need to be changed. For example,
if secrets of type `docker-reg-auth` affect how the pod's images are pulled, the
Kubelet will need to be changed to accommodate this. Subsequent proposals can
address this on a type-by-type basis.

## Examples

For clarity, let's examine some detailed examples of some common use-cases in
terms of the suggested changes. All of these examples are assumed to be created
in a namespace called `example`.

### Use-Case: Pod with ssh keys

To create a pod that uses an ssh key stored as a secret, we first need to create
a secret:

```json
{
  "kind": "Secret",
  "apiVersion": "v1",
  "metadata": {
    "name": "ssh-key-secret"
  },
  "data": {
    "id-rsa": "dmFsdWUtMg0KDQo=",
    "id-rsa.pub": "dmFsdWUtMQ0K"
  }
}
```

**Note:** The serialized JSON and YAML values of secret data are encoded as
base64 strings.  Newlines are not valid within these strings and must be
omitted.

Now we can create a pod which references the secret with the ssh key and
consumes it in a volume:

```json
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "secret-test-pod",
    "labels": {
      "name": "secret-test"
    }
  },
  "spec": {
    "volumes": [
      {
        "name": "secret-volume",
        "secret": {
          "secretName": "ssh-key-secret"
        }
      }
    ],
    "containers": [
      {
        "name": "ssh-test-container",
        "image": "mySshImage",
        "volumeMounts": [
          {
            "name": "secret-volume",
            "readOnly": true,
            "mountPath": "/etc/secret-volume"
          }
        ]
      }
    ]
  }
}
```

When the container's command runs, the pieces of the key will be available in:

    /etc/secret-volume/id-rsa.pub
    /etc/secret-volume/id-rsa

The container is then free to use the secret data to establish an ssh
connection.

### Use-Case: Pods with prod / test credentials

This example illustrates a pod which consumes a secret containing prod
credentials and another pod which consumes a secret with test environment
credentials.

The secrets:

```json
{
  "apiVersion": "v1",
  "kind": "List",
  "items":
  [{
    "kind": "Secret",
    "apiVersion": "v1",
    "metadata": {
      "name": "prod-db-secret"
    },
    "data": {
      "password": "dmFsdWUtMg0KDQo=",
      "username": "dmFsdWUtMQ0K"
    }
  },
  {
    "kind": "Secret",
    "apiVersion": "v1",
    "metadata": {
      "name": "test-db-secret"
    },
    "data": {
      "password": "dmFsdWUtMg0KDQo=",
      "username": "dmFsdWUtMQ0K"
    }
  }]
}
```

The pods:

```json
{
  "apiVersion": "v1",
  "kind": "List",
  "items":
  [{
    "kind": "Pod",
    "apiVersion": "v1",
    "metadata": {
      "name": "prod-db-client-pod",
      "labels": {
        "name": "prod-db-client"
      }
    },
    "spec": {
      "volumes": [
        {
          "name": "secret-volume",
          "secret": {
            "secretName": "prod-db-secret"
          }
        }
      ],
      "containers": [
        {
          "name": "db-client-container",
          "image": "myClientImage",
          "volumeMounts": [
            {
              "name": "secret-volume",
              "readOnly": true,
              "mountPath": "/etc/secret-volume"
            }
          ]
        }
      ]
    }
  },
  {
    "kind": "Pod",
    "apiVersion": "v1",
    "metadata": {
      "name": "test-db-client-pod",
      "labels": {
        "name": "test-db-client"
      }
    },
    "spec": {
      "volumes": [
        {
          "name": "secret-volume",
          "secret": {
            "secretName": "test-db-secret"
          }
        }
      ],
      "containers": [
        {
          "name": "db-client-container",
          "image": "myClientImage",
          "volumeMounts": [
            {
              "name": "secret-volume",
              "readOnly": true,
              "mountPath": "/etc/secret-volume"
            }
          ]
        }
      ]
    }
  }]
}
```

The specs for the two pods differ only in the value of the object referred to by
the secret volume source. Both containers will have the following files present
on their filesystems:

    /etc/secret-volume/username
    /etc/secret-volume/password


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/secrets.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
