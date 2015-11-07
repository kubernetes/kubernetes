<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/secrets.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Secrets

Objects of type `secret` are intended to hold sensitive information, such as
passwords, OAuth tokens, and ssh keys.  Putting this information in a `secret`
is safer and more flexible than putting it verbatim in a `pod` definition or in
a docker image. See [Secrets design document](../design/secrets.md) for more information.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Secrets](#secrets)
  - [Overview of Secrets](#overview-of-secrets)
    - [Service Accounts Automatically Create and Attach Secrets with API Credentials](#service-accounts-automatically-create-and-attach-secrets-with-api-credentials)
    - [Creating a Secret Manually](#creating-a-secret-manually)
    - [Manually specifying a Secret to be Mounted on a Pod](#manually-specifying-a-secret-to-be-mounted-on-a-pod)
    - [Manually specifying an imagePullSecret](#manually-specifying-an-imagepullsecret)
    - [Arranging for imagePullSecrets to be Automatically Attached](#arranging-for-imagepullsecrets-to-be-automatically-attached)
    - [Automatic Mounting of Manually Created Secrets](#automatic-mounting-of-manually-created-secrets)
  - [Details](#details)
    - [Restrictions](#restrictions)
    - [Consuming Secret Values](#consuming-secret-values)
    - [Secret and Pod Lifetime interaction](#secret-and-pod-lifetime-interaction)
  - [Use cases](#use-cases)
    - [Use-Case: Pod with ssh keys](#use-case-pod-with-ssh-keys)
    - [Use-Case: Pods with prod / test credentials](#use-case-pods-with-prod--test-credentials)
    - [Use-case: Secret visible to one container in a pod](#use-case-secret-visible-to-one-container-in-a-pod)
  - [Security Properties](#security-properties)
    - [Protections](#protections)
    - [Risks](#risks)

<!-- END MUNGE: GENERATED_TOC -->

## Overview of Secrets

A Secret is an object that contains a small amount of sensitive data such as
a password, a token, or a key.  Such information might otherwise be put in a
Pod specification or in an image; putting it in a Secret object allows for
more control over how it is used, and reduces the risk of accidental exposure.

Users can create secrets, and the system also creates some secrets.

To use a secret, a pod needs to reference the secret.
A secret can be used with a pod in two ways: either as files in a [volume](volumes.md) mounted on one or more of
its containers, or used by kubelet when pulling images for the pod.

### Service Accounts Automatically Create and Attach Secrets with API Credentials

Kubernetes automatically creates secrets which contain credentials for
accessing the API and it automatically modifies your pods to use this type of
secret.

The automatic creation and use of API credentials can be disabled or overridden
if desired.  However, if all you need to do is securely access the apiserver,
this is the recommended workflow.

See the [Service Account](service-accounts.md) documentation for more
information on how Service Accounts work.

### Creating a Secret Manually

This is an example of a simple secret, in yaml format:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: dmFsdWUtMg0K
  username: dmFsdWUtMQ0K
```

The data field is a map.  Its keys must match
[`DNS_SUBDOMAIN`](../design/identifiers.md), except that leading dots are also
allowed.  The values are arbitrary data, encoded using base64. The values of
username and password in the example above, before base64 encoding,
are `value-1` and `value-2`, respectively, with carriage return and newline characters at the end.

Create the secret using [`kubectl create`](kubectl/kubectl_create.md).

Once the secret is created, you can need to modify your pod to specify
that it should use the secret.

### Manually specifying a Secret to be Mounted on a Pod

This is an example of a pod that mounts a secret in a volume:

```json
{
 "apiVersion": "v1",
 "kind": "Pod",
  "metadata": {
    "name": "mypod",
    "namespace": "myns"
  },
  "spec": {
    "containers": [{
      "name": "mypod",
      "image": "redis",
      "volumeMounts": [{
        "name": "foo",
        "mountPath": "/etc/foo",
        "readOnly": true
      }]
    }],
    "volumes": [{
      "name": "foo",
      "secret": {
        "secretName": "mysecret"
      }
    }]
  }
}
```

Each secret you want to use needs its own `spec.volumes`.

If there are multiple containers in the pod, then each container needs its
own `volumeMounts` block, but only one `spec.volumes` is needed per secret.

You can package many files into one secret, or use many secrets,
whichever is convenient.

See another example of creating a secret and a pod that consumes that secret in a volume [here](secrets/).

### Manually specifying an imagePullSecret

Use of imagePullSecrets is described in the [images documentation](images.md#specifying-imagepullsecrets-on-a-pod)

### Arranging for imagePullSecrets to be Automatically Attached

You can manually create an imagePullSecret, and reference it from
a serviceAccount.  Any pods created with that serviceAccount
or that default to use that serviceAccount, will get their imagePullSecret
field set to that of the service account.
See [here](service-accounts.md#adding-imagepullsecrets-to-a-service-account)
 for a detailed explanation of that process.


### Automatic Mounting of Manually Created Secrets

We plan to extend the service account behavior so that manually created
secrets (e.g. one containing a token for accessing a github account)
can be automatically attached to pods based on their service account.
*This is not implemented yet.  See [issue 9902](http://issue.k8s.io/9902).*

## Details

### Restrictions

Secret volume sources are validated to ensure that the specified object
reference actually points to an object of type `Secret`.  Therefore, a secret
needs to be created before any pods that depend on it.

Secret API objects reside in a namespace.   They can only be referenced by pods
in that same namespace.

Individual secrets are limited to 1MB in size.  This is to discourage creation
of very large secrets which would exhaust apiserver and kubelet memory.
However, creation of many smaller secrets could also exhaust memory.  More
comprehensive limits on memory usage due to secrets is a planned feature.

Kubelet only supports use of secrets for Pods it gets from the API server.
This includes any pods created using kubectl, or indirectly via a replication
controller.  It does not include pods created via the kubelets
`--manifest-url` flag, its `--config` flag, or its REST API (these are
not common ways to create pods.)

### Consuming Secret Values

Inside the container that mounts a secret volume, the secret keys appear as
files and the secret values are base-64 decoded and stored inside these files.
This is the result of commands
executed inside the container from the example above:

```console
$ ls /etc/foo/
username
password
$ cat /etc/foo/username
value-1
$ cat /etc/foo/password
value-2
```

The program in a container is responsible for reading the secret(s) from the
files.  Currently, if a program expects a secret to be stored in an environment
variable, then the user needs to modify the image to populate the environment
variable from the file as an step before running the main program.  Future
versions of Kubernetes are expected to provide more automation for populating
environment variables from files.


### Secret and Pod Lifetime interaction

When a pod is created via the API, there is no check whether a referenced
secret exists.  Once a pod is scheduled, the kubelet will try to fetch the
secret value.  If the secret cannot be fetched because it does not exist or
because of a temporary lack of connection to the API server, kubelet will
periodically retry.  It will report an event about the pod explaining the
reason it is not started yet.  Once the a secret is fetched, the kubelet will
create and mount a volume containing it.  None of the pod's containers will
start until all the pod's volumes are mounted.

Once the kubelet has started a pod's containers, its secret volumes will not
change, even if the secret resource is modified.  To change the secret used,
the original pod must be deleted, and a new pod (perhaps with an identical
`PodSpec`) must be created.  Therefore, updating a secret follows the same
workflow as deploying a new container image.  The `kubectl rolling-update`
command can be used ([man page](kubectl/kubectl_rolling-update.md)).

The [`resourceVersion`](../devel/api-conventions.md#concurrency-control-and-consistency)
of the secret is not specified when it is referenced.
Therefore, if a secret is updated at about the same time as pods are starting,
then it is not defined which version of the secret will be used for the pod. It
is not possible currently to check what resource version of a secret object was
used when a pod was created.  It is planned that pods will report this
information, so that a replication controller restarts ones using an old
`resourceVersion`.  In the interim, if this is a concern, it is recommended to not
update the data of existing secrets, but to create new ones with distinct names.

## Use cases

### Use-Case: Pod with ssh keys

To create a pod that uses an ssh key stored as a secret, we first need to create a secret:

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

The container is then free to use the secret data to establish an ssh connection.

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

Both containers will have the following files present on their filesystems:

```console
    /etc/secret-volume/username
    /etc/secret-volume/password
```

Note how the specs for the two pods differ only in one field;  this facilitates
creating pods with different capabilities from a common pod config template.

You could further simplify the base pod specification by using two Service Accounts:
one called, say, `prod-user` with the `prod-db-secret`, and one called, say,
`test-user` with the `test-db-secret`.  Then, the pod spec can be shortened to, for example:

```json
{
"kind": "Pod",
"apiVersion": "v1",
"metadata": {
  "name": "prod-db-client-pod",
  "labels": {
    "name": "prod-db-client"
  }
},
"spec": {
  "serviceAccount": "prod-db-client",
  "containers": [
    {
      "name": "db-client-container",
      "image": "myClientImage",
    }
  ]
}
```

### Use-case: Secret visible to one container in a pod

<a name="use-case-two-containers"></a>

Consider a program that needs to handle HTTP requests, do some complex business
logic, and then sign some messages with an HMAC.  Because it has complex
application logic, there might be an unnoticed remote file reading exploit in
the server, which could expose the private key to an attacker.

This could be divided into two processes in two containers: a frontend container
which handles user interaction and business logic, but which cannot see the
private key; and a signer container that can see the private key, and responds
to simple signing requests from the frontend (e.g. over localhost networking).

With this partitioned approach, an attacker now has to trick the application
server into doing something rather arbitrary, which may be harder than getting
it to read a file.

<!-- TODO: explain how to do this while still using automation. -->

## Security Properties

### Protections

Because `secret` objects can be created independently of the `pods` that use
them, there is less risk of the secret being exposed during the workflow of
creating, viewing, and editing pods.  The system can also take additional
precautions with `secret` objects, such as avoiding writing them to disk where
possible.

A secret is only sent to a node if a pod on that node requires it.  It is not
written to disk.  It is stored in a tmpfs.  It is deleted once the pod that
depends on it is deleted.

On most Kubernetes-project-maintained distributions, communication between user
to the apiserver, and from apiserver to the kubelets, is protected by SSL/TLS.
Secrets are protected when transmitted over these channels.

Secret data on nodes is stored in tmpfs volumes and thus does not come to rest
on the node.

There may be secrets for several pods on the same node.  However, only the
secrets that a pod requests are potentially visible within its containers.
Therefore, one Pod does not have access to the secrets of another pod.

There may be several containers in a pod.  However, each container in a pod has
to request the secret volume in its `volumeMounts` for it to be visible within
the container.  This can be used to construct useful [security partitions at the
Pod level](#use-case-two-containers).

### Risks

 - In the API server secret data is stored as plaintext in etcd; therefore:
   - Administrators should limit access to etcd to admin users
   - Secret data in the API server is at rest on the disk that etcd uses; admins may want to wipe/shred disks
     used by etcd when no longer in use
 - Applications still need to protect the value of secret after reading it from the volume,
   such as not accidentally logging it or transmitting it to an untrusted party.
 - A user who can create a pod that uses a secret can also see the value of that secret.  Even
   if apiserver policy does not allow that user to read the secret object, the user could
   run a pod which exposes the secret.
 - If multiple replicas of etcd are run, then the secrets will be shared between them.
   By default, etcd does not secure peer-to-peer communication with SSL/TLS, though this can be configured.
 - It is not possible currently to control which users of a Kubernetes cluster can
   access a secret.  Support for this is planned.
 - Currently, anyone with root on any node can read any secret from the apiserver,
   by impersonating the kubelet.  It is a planned feature to only send secrets to
   nodes that actually require them, to restrict the impact of a root exploit on a
   single node.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/secrets.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
