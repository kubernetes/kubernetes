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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/docker.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Running Kubernetes locally via Docker
-------------------------------------

**Table of Contents**

- [Overview](#setting-up-a-cluster)
- [Prerequisites](#prerequisites)
- [Step One: Run etcd](#step-one-run-etcd)
- [Step Two: Run the master](#step-two-run-the-master)
- [Step Three: Run the service proxy](#step-three-run-the-service-proxy)
- [Test it out](#test-it-out)
- [Run an application](#run-an-application)
- [Expose it as a service:](#expose-it-as-a-service)
- [A note on turning down your cluster](#a-note-on-turning-down-your-cluster)

### Overview

The following instructions show you how to set up a simple, single node Kubernetes cluster using Docker.

Here's a diagram of what the final result will look like:
![Kubernetes Single Node on Docker](k8s-singlenode-docker.png)

### Prerequisites

1. You need to have docker installed on one machine.

### Step One: Run etcd

```sh
docker run --net=host -d gcr.io/google_containers/etcd:2.0.12 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

### Step Two: Run the master

```sh
docker run --net=host --privileged -d -v /sys:/sys:ro -v /var/run/docker.sock:/var/run/docker.sock  gcr.io/google_containers/hyperkube:v1.0.1 /hyperkube kubelet --api-servers=http://localhost:8080 --v=2 --insecure-bind-address=0.0.0.0 --enable-server --hostname-override=127.0.0.1 --config=/etc/kubernetes/manifests
```

This actually runs the kubelet, which in turn runs a [pod](../user-guide/pods.md) that contains the other master components.

### Step Three: Run the service proxy

*Note, this could be combined with master above, but it requires --privileged for iptables manipulation*

```sh
docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v1.0.1 /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

### Test it out

At this point you should have a running Kubernetes cluster.  You can test this by downloading the kubectl
binary
([OS X](https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/darwin/amd64/kubectl))
([linux](https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/linux/amd64/kubectl))

*Note:*
On OS/X you will need to set up port forwarding via ssh:

```sh
boot2docker ssh -L8080:localhost:8080
```

List the nodes in your cluster by running::

```sh
kubectl get nodes
```

This should print:

```console
NAME        LABELS    STATUS
127.0.0.1   <none>    Ready
```

If you are running different Kubernetes clusters, you may need to specify `-s http://localhost:8080` to select the local cluster.

### Run an application

```sh
kubectl -s http://localhost:8080 run nginx --image=nginx --port=80
```

now run `docker ps` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service

```sh
kubectl expose rc nginx --port=80
```

This should print:

```console
NAME      LABELS    SELECTOR              IP          PORT(S)
nginx     run=nginx run=nginx             <ip-addr>   80/TCP
```

If ip-addr is blank run the following command to obtain it. Know issue #10836

```sh
kubectl get svc nginx
```

Hit the webserver:

```sh
curl <insert-ip-from-above-here>
```

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### A note on turning down your cluster

Many of these containers run under the management of the `kubelet` binary, which attempts to keep containers running, even if they fail.  So, in order to turn down
the cluster, you need to first kill the kubelet container, and then any other containers.

You may use `docker ps -a | awk '{print $1}' | xargs docker kill`, note this removes _all_ containers running under Docker, so use with caution.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
