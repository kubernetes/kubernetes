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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/simple-yaml.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Getting started with config files.

In addition to the imperative style commands described [elsewhere](simple-nginx.md), Kubernetes
supports declarative YAML or JSON configuration files.  Often times config files are preferable
to imperative commands, since they can be checked into version control and changes to the files
can be code reviewed, producing a more robust, reliable and archival system.

### Running a container from a pod configuration file

```console
$ cd kubernetes
$ kubectl create -f ./pod.yaml
```

Where pod.yaml contains something like:

<!-- BEGIN MUNGE: EXAMPLE pod.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

[Download example](pod.yaml)
<!-- END MUNGE: EXAMPLE pod.yaml -->

You can see your cluster's pods:

```console
$ kubectl get pods
```

and delete the pod you just created:

```console
$ kubectl delete pods nginx
```

### Running a replicated set of containers from a configuration file

To run replicated containers, you need a [Replication Controller](replication-controller.md).
A replication controller is responsible for ensuring that a specific number of pods exist in the
cluster.

```console
$ cd kubernetes
$ kubectl create -f ./replication.yaml
```

Where `replication.yaml` contains:

<!-- BEGIN MUNGE: EXAMPLE replication.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

[Download example](replication.yaml)
<!-- END MUNGE: EXAMPLE replication.yaml -->

To delete the replication controller (and the pods it created):

```console
$ kubectl delete rc nginx
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/simple-yaml.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
