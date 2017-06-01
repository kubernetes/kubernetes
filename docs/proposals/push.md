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
[here](http://releases.k8s.io/release-1.3/docs/proposals/push.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Proposal: commit a container and push it to registry
----

## Motivation

There are some use case to commit a container, and push its resulting image to
a registry, for example:

1. For a run-once pod whose result represents an image that should be pushed to
a registry for consumption by other actors in the system

2. There is someone desire to be able to modify a running container and then
commit/push the change so they could, for example, make a debug change and then
scale up the deployment to test it in a clustered mode, without having to start
from scratch with building a new image.

The goal of this proposal is to add the commit/push capability, which will allow
user commit and push their containers manually, or automically before/after
container exits.

## Design

### Manually commit and push a container

######1. create a ThirdPartyResource

```
metadata:
  name: push.k8s.io
apiVersion: extensions/v1beta1
kind: ThirdPartyResource
description: "Allow user to commit a container and push it into registry"
versions:
- name: v1
```

###### 2. create a `Push` resource
user can then create a `Push` resource claimming that he wants commit a
container and push it into registry. Example `Push` resource is as follows:

```
kind: Push
apiVersion: k8s.io/v1
metadata:
  namespace: ns1
  name: name1
  labels:
    kubepush.alpha.kubernetes.io/nodename: i-jgganq
spec:
  podName: master-nginx-i-94uugzhjm
  containerName: master-nginx
  image: cargo.caicloud.io/liangmq/nginx: 1.2
  imagePushSecrets:
  - name: cargo.caicloud.io
status:
  phase: Succeeded
  message: push succeeded
```

Note:

* Why we need a `kubepush.alpha.kubernetes.io/nodename` label: this label
  indicate which node the container user whant commit and push is running on


######3. kubepush daemonset
There is a kubepush daemonst running on every node, list&watch `Push` resource
and

```
apiVersion: extensions/v1beta1
kind: DaemonSet
metadata:
  labels:
    app: kubepush
  name: kubepush
  namespace: kube-system
spec:
  template:
    metadata:
      labels:
        app: kubepush
    spec:
      containers:
        - name: kubepush
          image: index.caicloud.io/caicloud/kubepush:latest
          imagePullPolicy: Always
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          volumeMounts:
            - name: run
              mountPath: /var/run/docker.sock
      volumes:
        - name: run
          hostPath:
              path: /var/run/docker.sock
```

Note:

* mount `/var/run/docker.sock` of node into kubepush Pod, so that kubepush Pod
  could access docker daemon on host
* The kubepush Pod running on node i-node-a just need list&watch Push resource
  with `kubepush.alpha.kubernetes.io/nodename: i-node-a` label, if kubepush can
  not found such a container on node, report failure
* Why use label: third party resource doesn't support filter by field


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/push.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
