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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# ConfigMap example



## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and running, and that you have
installed the `kubectl` command line tool somewhere in your path. Please see the [getting
started](../../../docs/getting-started-guides/) for installation instructions for your platform.

## Step One: Create the ConfigMap

A ConfigMap contains a set of named strings.

Use the [`examples/configmap/configmap.yaml`](configmap.yaml) file to create a ConfigMap:

```console
$ kubectl create -f docs/user-guide/configmap/configmap.yaml
```

You can use `kubectl` to see information about the ConfigMap:

```console
$ kubectl get configmap
NAME          DATA
test-secret   2

$ kubectl describe configMap test-configmap
Name:          test-configmap
Labels:        <none>
Annotations:   <none>

Data
====
data-1: 7 bytes
data-2: 7 bytes
```

View the values of the keys with `kubectl get`:

```console
$ cluster/kubectl.sh get configmaps test-configmap -o yaml
apiVersion: v1
data:
  data-1: value-1
  data-2: value-2
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T20:28:50Z
  name: test-configmap
  namespace: default
  resourceVersion: "1090"
  selfLink: /api/v1/namespaces/default/configmaps/test-configmap
  uid: 384bd365-d67e-11e5-8cd0-68f728db1985
```

## Step Two: Create a pod that consumes a configMap in environment variables

Use the [`examples/configmap/env-pod.yaml`](env-pod.yaml) file to create a Pod that consumes the
ConfigMap in environment variables.

```console
$ kubectl create -f docs/user-guide/configmap/env-pod.yaml
```

This pod runs the `env` command to display the environment of the container:

```console
$ kubectl logs secret-test-pod
KUBE_CONFIG_1=value-1
KUBE_CONFIG_2=value-2
```

## Step Three: Create a pod that sets the command line using ConfigMap

Use the [`examples/configmap/command-pod.yaml`](env-pod.yaml) file to create a Pod with a container
whose command is injected with the keys of a ConfigMap

```console
$ kubectl create -f docs/user-guide/configmap/env-pod.yaml
```

This pod runs an `echo` command to display the keys:

```console
value-1 value-2
```

## Step Four: Create a pod that consumes a configMap in a volume

Pods can also consume ConfigMaps in volumes.  Use the [`examples/configmap/volume-pod.yaml`](volume-pod.yaml) file to create a Pod that consume the ConfigMap in a volume.

```console
$ kubectl create -f docs/user-guide/configmap/volume-pod.yaml
```

This pod runs a `cat` command to print the value of one of the keys in the volume:

```console
value-1
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/configmap/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
