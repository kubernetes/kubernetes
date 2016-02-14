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

# Notes on Different UX with rkt container runtime

### Doesn't support ENTRYPOINT + CMD feature

To run a Docker image, rkt will convert it into [App Container Image (ACI) format](https://github.com/appc/spec/blob/master/SPEC.md) first.
However, during the conversion, the `ENTRYPOINT` and `CMD` are concatentated to construct ACI's `Exec` field.
This means after the conversion, we are not able to replace only `ENTRYPOINT` or `CMD` without touching the other part.
So for now, users are recommended to specify the **executable path** in `Command` and **arguments** in `Args`.
(This has the same effect if users specify the **executable path + arguments** in `Command` or `Args` alone).

For example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

The above pod yaml file is valid as it's not specifying `Command` or `Args`, so the default `ENTRYPOINT` and `CMD` of the image will be used.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  labels:
    name: busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command:
    - /bin/sleep
    - 1000
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  labels:
    name: busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command:
    - /bin/sleep
    args:
    - 1000
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  labels:
    name: busybox
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - /bin/sleep
    - 1000
```

All the three examples above are valid as they contain both the executable path and the arguments.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  labels:
    name: busybox
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - 1000
```

The last example is invalid, as we cannot override just the `CMD` of the image alone.





<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/rkt/notes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
