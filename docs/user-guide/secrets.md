<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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
    - [Built-in Secrets](#built-in-secrets)
      - [Service Accounts Automatically Create and Attach Secrets with API Credentials](#service-accounts-automatically-create-and-attach-secrets-with-api-credentials)
    - [Creating your own Secrets](#creating-your-own-secrets)
      - [Creating a Secret Using kubectl create secret](#creating-a-secret-using-kubectl-create-secret)
      - [Creating a Secret Manually](#creating-a-secret-manually)
      - [Decoding a Secret](#decoding-a-secret)
    - [Using Secrets](#using-secrets)
      - [Using Secrets as Files from a Pod](#using-secrets-as-files-from-a-pod)
        - [Consuming Secret Values from Volumes](#consuming-secret-values-from-volumes)
      - [Using Secrets as Environment Variables](#using-secrets-as-environment-variables)
        - [Consuming Secret Values from Environment Variables](#consuming-secret-values-from-environment-variables)
      - [Using imagePullSecrets](#using-imagepullsecrets)
        - [Manually specifying an imagePullSecret](#manually-specifying-an-imagepullsecret)
        - [Arranging for imagePullSecrets to be Automatically Attached](#arranging-for-imagepullsecrets-to-be-automatically-attached)
      - [Automatic Mounting of Manually Created Secrets](#automatic-mounting-of-manually-created-secrets)
  - [Details](#details)
    - [Restrictions](#restrictions)
    - [Secret and Pod Lifetime interaction](#secret-and-pod-lifetime-interaction)
  - [Use cases](#use-cases)
    - [Use-Case: Pod with ssh keys](#use-case-pod-with-ssh-keys)
    - [Use-Case: Pods with prod / test credentials](#use-case-pods-with-prod--test-credentials)
    - [Use-case: Dotfiles in secret volume](#use-case-dotfiles-in-secret-volume)
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
A secret can be used with a pod in two ways: as files in a [volume](volumes.md) mounted on one or more of
its containers, in environment variables, or used by kubelet when pulling images for the pod.

### Built-in Secrets

#### Service Accounts Automatically Create and Attach Secrets with API Credentials

Kubernetes automatically creates secrets which contain credentials for
accessing the API and it automatically modifies your pods to use this type of
secret.

The automatic creation and use of API credentials can be disabled or overridden
if desired.  However, if all you need to do is securely access the apiserver,
this is the recommended workflow.

See the [Service Account](service-accounts.md) documentation for more
information on how Service Accounts work.

### Creating your own Secrets

#### Creating a Secret Using kubectl create secret

Say that some pods need to access a database.  The
username and password that the pods should use is in the files
`./username.txt` and `./password.txt` on your local machine.

```console
# Create files needed for rest of example.
$ echo "admin" > ./username.txt
$ echo "1f2d1e2e67df" > ./password.txt
```

The `kubectl create secret` command
packages these files into a Secret and creates
the object on the Apiserver.

```console
$ kubectl create secret generic db-user-pass --from-file=./username.txt --from-file=./password.txt
secret "db-user-pass" created
```

You can check that the secret was created like this:

```console
$ kubectl get secrets
NAME                  TYPE                                  DATA      AGE
db-user-pass          Opaque                                2         51s
$ kubectl describe secrets/db-user-pass
Name:		db-user-pass
Namespace:	default
Labels:		<none>
Annotations:	<none>

Type:	Opaque

Data
====
password.txt:	13 bytes
username.txt:	6 bytes
```

Note that neither `get` nor `describe` shows the contents of the file by default.
This is to protect the secret from being exposed accidentally to someone looking
or from being stored in a terminal log.

See [decoding a secret](#decoding-a-secret) for how to see the contents.

#### Creating a Secret Manually

You can also create a secret object in a file first,
in json or yaml format, and then create that object.

Each item must be base64 encoded:

```console
$ echo "admin" | base64
YWRtaW4K
$ echo "1f2d1e2e67df" | base64
MWYyZDFlMmU2N2RmCg==
```

Now write a secret object that looks like this:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: MWYyZDFlMmU2N2RmCg==
  username: YWRtaW4K
```

The data field is a map.  Its keys must match
[`DNS_SUBDOMAIN`](../design/identifiers.md), except that leading dots are also
allowed.  The values are arbitrary data, encoded using base64.

Create the secret using [`kubectl create`](kubectl/kubectl_create.md):

```console
$ kubectl create -f ./secret.yaml
secret "mysecret" created
```

**Encoding Note:** The serialized JSON and YAML values of secret data are encoded as
base64 strings.  Newlines are not valid within these strings and must be
omitted (i.e. do not use `-b` option of `base64` which breaks long lines.)

#### Decoding a Secret

Get back the secret created in the previous section:

```console
$ kubectl get secret mysecret -o yaml
apiVersion: v1
data:
  password: MWYyZDFlMmU2N2RmCg==
  username: YWRtaW4K
kind: Secret
metadata:
  creationTimestamp: 2016-01-22T18:41:56Z
  name: mysecret
  namespace: default
  resourceVersion: "164619"
  selfLink: /api/v1/namespaces/default/secrets/mysecret
  uid: cfee02d6-c137-11e5-8d73-42010af00002
type: Opaque
```

Decode the password field:

```console
$ echo "MWYyZDFlMmU2N2RmCg==" | base64 -D
1f2d1e2e67df
```

### Using Secrets

Secrets can be mounted as data volumes or be exposed as environment variables to
be used by a container in a pod.  They can also be used by other parts of the
system, without being directly exposed to the pod.  For example, they can hold
credentials that other parts of the system should use to interact with external
systems on your behalf.

#### Using Secrets as Files from a Pod

To consume a Secret in a volume in a Pod:

1. Create a secret or use an existing one.  Multiple pods can reference the same secret.
1. Modify your Pod definition to add a volume under `spec.volumes[]`.  Name the volume anything, and have a `spec.volumes[].secret.secretName` field equal to the name of the secret object.
1. Add a `spec.containers[].volumeMounts[]` to each container that needs the secret.  Specify `spec.containers[].volumeMounts[].readOnly = true` and `spec.containers[].volumeMounts[].mountPath` to an unused directory name where you would like the secrets to appear.
1. Modify your image and/or command line so that the the program looks for files in that directory.  Each key in the secret `data` map becomes the filename under `mountPath`.

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

Each secret you want to use needs to be referred to in `spec.volumes`.

If there are multiple containers in the pod, then each container needs its
own `volumeMounts` block, but only one `spec.volumes` is needed per secret.

You can package many files into one secret, or use many secrets, whichever is convenient.

See another example of creating a secret and a pod that consumes that secret in a volume [here](secrets/).

##### Consuming Secret Values from Volumes

Inside the container that mounts a secret volume, the secret keys appear as
files and the secret values are base-64 decoded and stored inside these files.
This is the result of commands
executed inside the container from the example above:

```console
$ ls /etc/foo/
username
password
$ cat /etc/foo/username
admin
$ cat /etc/foo/password
1f2d1e2e67df
```

The program in a container is responsible for reading the secret(s) from the
files.

#### Using Secrets as Environment Variables

To use a secret in an environment variable in a pod:

1. Create a secret or use an existing one.  Multiple pods can reference the same secret.
1. Modify your Pod definition in each container that you wish to consume the value of a secret key to add an environment variable for each secret key you wish to consume.  The environment variable that consumes the secret key should populate the secret's name and key in `env[x].valueFrom.secretKeyRef`.
1. Modify your image and/or command line so that the the program looks for values in the specified environment variabless

This is an example of a pod that mounts a secret in a volume:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secret-env-pod
spec:
  containers:
    - name: mycontainer
      image: redis
      env:
        - name: SECRET_USERNAME
          valueFrom:
            secretKeyRef:
              name: mysecret
              key: username
        - name: SECRET_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysecret
              key: password
  restartPolicy: Never
```

##### Consuming Secret Values from Environment Variables

Inside a container that consumes a secret in an environment variables, the secret keys appear as
normal environment variables containing the base-64 decoded values of the secret data.
This is the result of commands executed inside the container from the example above:

```console
$ echo $SECRET_USERNAME
admin
$ cat /etc/foo/password
1f2d1e2e67df
```

#### Using imagePullSecrets

An imagePullSecret is a way to pass a secret that contains a Docker (or other) image registry
password to the Kubelet so it can pull a private image on behalf of your Pod.

##### Manually specifying an imagePullSecret

Use of imagePullSecrets is described in the [images documentation](images.md#specifying-imagepullsecrets-on-a-pod)

##### Arranging for imagePullSecrets to be Automatically Attached

You can manually create an imagePullSecret, and reference it from
a serviceAccount.  Any pods created with that serviceAccount
or that default to use that serviceAccount, will get their imagePullSecret
field set to that of the service account.
See [here](service-accounts.md#adding-imagepullsecrets-to-a-service-account)
 for a detailed explanation of that process.

#### Automatic Mounting of Manually Created Secrets

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

Create a secret containing some ssh keys:

```console
$ kubectl create secret generic my-secret --from-file=ssh-privatekey=/path/to/.ssh/id_rsa --from-file=ssh-publickey=/path/to/.ssh/id_rsa.pub
```

**Security Note:** think carefully before sending your own ssh keys: other users of the cluster may have access to the secret.  Use a service account which you want to have accessible to all the users with whom you share the kubernetes cluster, and can revoke if they are compromised.


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

Make the secrets:

```console
$ kubectl create secret generic prod-db-password --from-literal=user=produser --from-literal=password=Y4nys7f11
secret "prod-db-password" created
$ kubectl create secret generic test-db-password --from-literal=user=testuser --from-literal=password=iluvtests
secret "test-db-password" created
```

Now make the pods:

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
      "image": "myClientImage"
    }
  ]
}
```

### Use-case: Dotfiles in secret volume

In order to make piece of data 'hidden' (ie, in a file whose name begins with a dot character), simply
make that key begin with a dot.  For example, when the following secret secret is mounted into a volume:

```json
{
  "kind": "Secret",
  "apiVersion": "v1",
  "metadata": {
    "name": "dotfile-secret"
  },
  "data": {
    ".secret-file": "dmFsdWUtMg0KDQo=",
  }
}

{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "secret-dotfiles-pod",
  },
  "spec": {
    "volumes": [
      {
        "name": "secret-volume",
        "secret": {
          "secretName": "dotfile-secret"
        }
      }
    ],
    "containers": [
      {
        "name": "dotfile-test-container",
        "image": "gcr.io/google_containers/busybox",
        "command": "ls -l /etc/secret-volume"
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


The `secret-volume` will contain a single file, called `.secret-file`, and
the `dotfile-test-container` will have this file present at the path
`/etc/secret-volume/.secret-file`.

**NOTE**

Files beginning with dot characters are hidden from the output of  `ls -l`;
you must use `ls -la` to see them when listing directory contents.


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




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/secrets.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
