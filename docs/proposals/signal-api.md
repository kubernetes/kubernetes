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

# Signal API

This document describes signal API effort in Kubernetes.

## Abstract

The goal is to provide a possibility to send signals to container entrypoint application. We are going to introduce
a concept of *notification* in the container's definition inside pod's spec, which will provide a possibility of
implementing the *notification types*. One of these types will be signals. Container will have a list of notifications
with the user-defined name (and also type-specific intormations), then the API resource will allow to use that
notification.

In case of signal type, the information which will be provided to this notification list is a POSIX signal name.
The signal notification will be in fact a mapping from the notification name to the actual POSIX name. In example,
user may want to define notification called *bump*, give it a *signal* type and point it to the SIGHUP signal.
After creating such a pod, he would be able to send it a *bump* notification, which will result.

The idea of the notifications allows to create many types, but only implementing the signal type is in the scope
of this proposal.

Once me make agreement on the API part, the document will be extended with the implementation details.

One thing that needs to be noticed is that restarting scenarios are not a goad of this proposal. The only goal here
is to send single signals, no more, no less. More complicated restart scenarios (like sending SIGTERM, then waiting
for graceful termination, then sending SIGKILL after reaching timeout) may need another resource in Kubernetes API,
which will not expose any POSIX terminology to the user.

## Motivation

Many Kubernetes users and operators would like to send custom signals to the processes running in containers. That
may be motivated both by default behaviour that signals are bringing and the custom signal handling which the
program is doing.

The more concrete use-cases are described below.

## Use-Cases

#### Re-reading configuration

Many applications follow the convention to re-read configuration files when handling the SIGHUP signal. Ability
to send SIGHUP signal will allow us to reconfigure pods without updating their definition or removing them. Also,
we can consider using this API in the watch mechanism in kubelet to reconfigure a pod automatically on the ConfigMap
change.

## Analysis

The system needs to be able to:

1. Model correctly which signals are supported
2. Allow to define what signals can be used for each pod (and whether they can be used at all)
3. Provide custom names for the chosen signals in pod

## Proposed design

Signal API will be a part of pod API. There will be no separate API group for that.

#### Schema

The proposed schema is as follow.

There will be a type Notification being a part of the Container (inside the PodSpec):

```go
type Notification struct {
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`
	Type string `json:"type" protobuf:"bytes,2,opt,name=type"`
	Signal string `json:"signal" protobuf:"bytes,3,opt,name=signal"`
}
```

The *signal* parameter will be able to take any name of POSIX signal and that will be a criterion to validate that
field.

And there will be an API subresource of `pod` called `notify`, with the options:
- name

#### Endpoints

There will be PUT and POST endpoint, rooted at /api/v1/, with the parameters as listed:
- /namespaces/{namespace}/pods/{pod}/notify
  - name  - name of the notification to send

#### ContainerRuntime interface

The following method will be defined in the ContainerRuntime interface:

```go
Kill(id, signal string) error
```

Translating the notification name to the POSIX signal name will happen before using any container runtime, on the
kubelet API level.

In case of Docker, the `kill` API resource will be called.

In rkt, the only way to signal containers is to use machinectl. Implementing the good way to signal containers in rkt
seems to be out of scope of this proposal.

## Examples

Let's assume that we have a pod defined like:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: signal-example
spec:
  containers:
    - name: my-container
      image: myusername/my-program-which-handles-signals
      notifications:
        - name: reconfigure
          type: signal
          signal: SIGHUP
        - name: stop
          type: signal
          signal: SIGTERM
```

The names of notifications should be unique across all containers in the pod.

The contents of the *notifications* section will allow us to do the requests to
`/api/v1/namespaces/default/pods/signal-example/notify` with data like `{"name": "reconfigure"}` or
`{"name": "stop"}` etc. This API isn't pointing the concrete container in any way - that's why we aim to keep the
cross-container name uniqueness.

And finally, there will be a possibility to do that with kubectl:

```
kubectl notify signal-example reconfigure
kubectl notify signal-example stop
```

## Further improvements

Depending on the further requirements the following features may be added:
- support for Windows messages
- keeping notification requests in etcd and send them asynchronously, in case of kube-apiserver or kubelet failover

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/signal-api.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
