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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/docker.md).

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
- [Expose it as a service](#expose-it-as-a-service)
- [A note on turning down your cluster](#a-note-on-turning-down-your-cluster)

### Overview

The following instructions show you how to set up a simple, single node Kubernetes cluster using Docker.

Here's a diagram of what the final result will look like:
![Kubernetes Single Node on Docker](k8s-singlenode-docker.png)

### Prerequisites

1. You need to have docker installed on one machine.
2. Your kernel should support memory and swap accounting. Ensure that the
following configs are turned on in your linux kernel:

    ```console
    CONFIG_RESOURCE_COUNTERS=y
    CONFIG_MEMCG=y
    CONFIG_MEMCG_SWAP=y
    CONFIG_MEMCG_SWAP_ENABLED=y
    CONFIG_MEMCG_KMEM=y
    ```

3. Enable the memory and swap accounting in the kernel, at boot, as command line
parameters as follows:

    ```console
    GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"
    ```

    NOTE: The above is specifically for GRUB2.
    You can check the command line parameters passed to your kernel by looking at the
    output of /proc/cmdline:

    ```console
    $cat /proc/cmdline
    BOOT_IMAGE=/boot/vmlinuz-3.18.4-aufs root=/dev/sda5 ro cgroup_enable=memory
    swapaccount=1
    ```

4. Decide what Kubernetes version to use.  Set the `${K8S_VERSION}` variable to
   a value such as "1.0.7".

### Step One: Run etcd

```sh
docker run --net=host -d gcr.io/google_containers/etcd:2.2.1 /usr/local/bin/etcd --addr=127.0.0.1:4001 --bind-addr=0.0.0.0:4001 --data-dir=/var/etcd/data
```

### Step Two: Run the master

```sh
docker run \
    --volume=/:/rootfs:ro \
    --volume=/sys:/sys:ro \
    --volume=/dev:/dev \
    --volume=/var/lib/docker/:/var/lib/docker:rw \
    --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
    --volume=/var/run:/var/run:rw \
    --net=host \
    --pid=host \
    --privileged=true \
    -d \
    gcr.io/google_containers/hyperkube:v${K8S_VERSION} \
    /hyperkube kubelet --containerized --hostname-override="127.0.0.1" --address="0.0.0.0" --api-servers=http://localhost:8080 --config=/etc/kubernetes/manifests
```

This actually runs the kubelet, which in turn runs a [pod](../user-guide/pods.md) that contains the other master components.

### Step Three: Run the service proxy

```sh
docker run -d --net=host --privileged gcr.io/google_containers/hyperkube:v${K8S_VERSION} /hyperkube proxy --master=http://127.0.0.1:8080 --v=2
```

### Test it out

At this point you should have a running Kubernetes cluster.  You can test this
by downloading the kubectl binary for `${K8S_VERSION}` (look at the URL in the
following links) and make it available by editing your PATH environment
variable.
([OS X](http://storage.googleapis.com/kubernetes-release/release/v1.0.7/bin/darwin/amd64/kubectl))
([linux](http://storage.googleapis.com/kubernetes-release/release/v1.0.7/bin/linux/amd64/kubectl))

For example, OS X:

```console
$ wget http://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/darwin/amd64/kubectl
$ chmod 755 kubectl
$ PATH=$PATH:`pwd`
```

Linux:

```console
$ wget http://storage.googleapis.com/kubernetes-release/release/v${K8S_VERSION}/bin/linux/amd64/kubectl
$ chmod 755 kubectl
$ PATH=$PATH:`pwd`
```

<hr>

**Note for OS/X users:**
You will need to set up port forwarding via ssh. For users still using boot2docker directly, it is enough to run the command:

```sh
boot2docker ssh -L8080:localhost:8080
```

Since the recent deprecation of boot2docker/osx-installer, the correct way to solve the problem is to issue

```sh
docker-machine ssh default -L 8080:localhost:8080
```

However, this solution works only from docker-machine version 0.5. For older versions of docker-machine, a workaround is the
following:

```sh
docker-machine env default
ssh -f -T -N -L8080:localhost:8080 -l docker $(echo $DOCKER_HOST | cut -d ':' -f 2 | tr -d '/')
```

Type `tcuser` as the password.

<hr>

List the nodes in your cluster by running:

```sh
kubectl get nodes
```

This should print:

```console
NAME        LABELS                             STATUS
127.0.0.1   kubernetes.io/hostname=127.0.0.1   Ready
```

If you are running different Kubernetes clusters, you may need to specify `-s http://localhost:8080` to select the local cluster.

### Run an application

```sh
kubectl -s http://localhost:8080 run nginx --image=nginx --port=80
```

Now run `docker ps` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service

```sh
kubectl expose rc nginx --port=80
```

Run the following command to obtain the IP of this service we just created. There are two IPs, the first one is internal (CLUSTER_IP), and the second one is the external load-balanced IP.

```sh
kubectl get svc nginx
```

Alternatively, you can obtain only the first IP (CLUSTER_IP) by running:

```sh
kubectl get svc nginx --template={{.spec.clusterIP}}
```

Hit the webserver with the first IP (CLUSTER_IP):

```sh
curl <insert-cluster-ip-here>
```

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### A note on turning down your cluster

Many of these containers run under the management of the `kubelet` binary, which attempts to keep containers running, even if they fail.  So, in order to turn down
the cluster, you need to first kill the kubelet container, and then any other containers.

You may use `docker kill $(docker ps -aq)`, note this removes _all_ containers running under Docker, so use with caution.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
