<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Command Execution Example

Because most Kubernetes providers do not expose pod or service IPs outside of the cluster (depending on network configuration), it's often necessary to execute commands within a pod, from inside the cluster. This is especially useful for debugging.

One way to do this is to create a busybox pod that sleeps forever (technically an hour, restarted until deleted).

For example, here is a sleep-pod definition ([sleep-pod.yaml](sleep-pod.yaml)):

```
apiVersion: v1
kind: Pod
metadata:
  name: sleep-pod
spec:
  containers:
  - name: busybox
    image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
  restartPolicy: Always
```

Create the sleep-pod ([sleep-pod.yaml](sleep-pod.yaml)):

```sh
kubectl create -f examples/cmd-exec/sleep-pod.yaml
```

Execute a command remotely:

```sh
kubectl exec sleep-pod -- echo hello
```

Delete the sleep-pod:

```sh
kubectl delete pod sleep-pod
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cmd-exec/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
