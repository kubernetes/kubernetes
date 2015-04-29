# Secrets example

Following this example, you will create a secret and a pod that consumes that secret in a volume.

## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and running, and that you have
installed the ```kubectl``` command line tool somewhere in your path. Please see the [getting
started](../../docs/getting-started-guides) for installation instructions for your platform.

## Step One: Create the secret

A secret contains a set of named byte arrays.

Use the `examples/secrets/secret.yaml` file to create a secret:

```shell
$ kubectl create -f examples/secrets/secret.yaml
```

You can use `kubectl` to see information about the secret:

```shell
$ kubectl get secrets
NAME          TYPE      DATA
test-secret   Opaque    2
```

## Step Two: Create a pod that consumes a secret

Pods consume secrets in volumes.  Now that you have created a secret, you can create a pod that
consumes it.

Use the `examples/secrets/secret-pod.yaml` file to create a Pod that consumes the secret.

```shell
$ kubectl create -f examples/secrets/secret-pod.yaml
```

This pod runs a binary that displays the content of one of the pieces of secret data in the secret
volume: 

```shell
$ kubectl log secret-test-pod
2015-04-29T21:17:24.712206409Z content of file "/etc/secret-volume/data-1": value-1
```
