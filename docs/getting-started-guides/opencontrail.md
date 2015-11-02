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
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/opencontrail.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Opencontrail Networking for Kubernetes - (Deployment in GCE)
------------------------------------------------------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Starting a Cluster](#starting-a-cluster)
    - [Set up working directory](#set-up-working-directory)
    - [Configure and start the kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
    - [Test](#test)
    - [Deploy addons](#deploy-addons)
    - [Trouble shooting](#trouble-shooting)

## Introduction

Opencontrail is carrier grade open-source network virtualization solution for cloud.
More details on opencontrail can be found @ http://www.opencontrail.org/

This document describes how to deploy opencontrail networking in kubernetes.
1 master and 3 nodes and 1 opencontrail gateway will be deployed by default.
You can scale to **any number of nodes** by changing some env vars.

## Prerequisites

1. GKE account and gcloud SDK setup on host should be complete
2. Verify gcloud api for authentication, authorization and access to GKE
3. Create compute instances successfuly from host using gcloud.
   For details please check - https://cloud.google.com/sdk/gcloud/reference/compute/instances/create
4. Environment variable - NETWORK_PROVIDER=opencontrail needs to be set before invoking kube-up.sh
5. Environment variable - OPENCONTRAIL_KUBERNETES_TAG and OPENCONTRAIL_TAG needs to be set
   Ex: export OPENCONTRAIL_KUBERNETES_TAG=vrouter_manifest
       export OPENCONTRAIL_TAG=R2.20
   These variables provide the opencontrail and opencontrail-kubernetes branch info to the salt provisioning
   scripts for picking up the specific version


## Starting a Cluster

### Set up working directory

Clone the Juniper kubernetes forked github repo locally

``` console
$ git clone -b opencontrail-integration https://github.com/Juniper/kubernetes/kubernetes.git
```

Note: Upstreaming of the integration code is in progress and PR is issued for this.

#### Building Kubernetes

Please follow instructions on building kubernetes @ https://github.com/kubernetes/kubernetes/tree/master/build
Note: On a Mac OS X Yosemite you need to have boot2docker

#### Deploy Kubernetes cluster

Deploying kubernetes cluster is as simple as the following:

1. Make sure to unset KUBERNETES_PROVIDER environment variable
   By default KUBERNETES_PROVIDER is set to GCE. You may also choose
   to set the variable exlicitly to KUBERNETES_PROVIDER=GCE
2. Run ~/cluster/kube-up.sh
  Ex: `$ KUBERNETES_PROVIDER=GCE ./kube-up.sh`

Above step should complete successfully and you should have a working kubernetes cluster in GCE with opencontrail
networking.

An example cluster is listed below:

| IP Address   |   Role   |
|--------------|----------|
|104.10.103.223|   node   |
|104.10.103.162|   node   |
|104.10.103.164|   node   |
|130.10.103.250|   master |
|104.10.103.250|   gateway|

__*Kubernetes-Master*__:

Opencontrail controller modules in the container are deployed and configured on the master.
Contrail contnaiers on the master are:

`root@kubernetes-master:~# docker ps |grep contrail | grep -v pause` <br />
`8d41e850e55b  opencontrail/kube-network-manager` <br />
`f5cd41ff1503  opencontrail/web:2.20` <br />
`8aa226fbc877  opencontrail/config:2.20` <br />
`bd9cbb8b5cb5  opencontrail/control:2.20` <br />
`b95a13b61d40  opencontrail/ifmap-server:2.20` <br />
`57c4596c9481  opencontrail/config:2.20` <br />
`e68d1eaec682  cassandra:2.2.0` <br />
`4c55916f2455  opencontrail/analytics:2.20` <br />
`bb42ad2c4c49  opencontrail/analytics:2.20` <br />
`3de472ba306d  opencontrail/analytics:2.20` <br />

Details on kube-network-manager and kubernetes on opencontrail can be found @ https://pedrormarques.wordpress.com/2015/07/14/kubernetes-networking-with-opencontrail/

__*Kubernetes-minion*__:

In addition to the kubelet and docker managed by kubelet, opencontrail plugin is deployed on the minion.

Kubelet on minion provisioned with opencontrail will have the opencontrail plugin configured as shown below

`DAEMON_ARGS="$DAEMON_ARGS --api-servers=https://kubernetes-master --enable-debugging-handlers=true  --cloud-provider=gce --config=/etc/kubernetes/manifests  --allow-privileged=True --v=2 --cluster-dns=10.0.0.10 --cluster-domain=cluster.local   --configure-cbr0=true --cgroup-root=/ --system-container=/system    --network_plugin=opencontrail "`

The additional argument passed to kubelet is __--network_plugin=opencontrail__ which will start the opencontrail kubelet plugin. The plugin adds container that belings to a pod to vrouter agent.

`root@kubernetes-minion-4bfu:~# ps -ef|grep opencontrail` <br />
`root     12419 14312  1 21:38 ?        00:00:00 /usr/bin/python /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail/opencontrail`

Opencontrail vrouter agent in container and kernel module (vrouter.ko) is deployed on the kubernetes cluster nodes.

`root@kubernetes-minion-4bfu:~# docker ps |grep contrail | grep -v pause` <br />
`497a0d6bd096    opencontrail/vrouter-agent:2.20` <br />

root@kubernetes-minion-4bfu:~# lsmod |grep vrouter <br />
vrouter               235766  1 <br />

Opencontrail vrouter agent manages the forwarding path for data. Please find more details on this @ https://github.com/Juniper/contrail-controller/blob/master/src/vnsw/agent/README

**Kubernetes-Opencontrail-gateway:

Opencontrail gateway provides gateway fucntion for any external (north-south) traffic going towards the internal kubernetes pods.
Details on this functionality can be found @ https://github.com/Juniper/contrail-controller/wiki/Simple-Gateway

Opencontrail vrouter agent in container and kernel module (vrouter.ko) is deployed on kubernetes-opencontrail-gateway

`root@kubernetes-opencontrail-gateway:~# lsmod | grep vrouter` <br />
`vrouter               235766  1 ` <br />
`root@kubernetes-opencontrail-gateway:~# docker ps | grep contrail | grep -v pause` <br />
`f88f474628fa        opencontrail/vrouter-agent:2.20` <br />

### Test

You can use `kubectl` command to check if the newly created cluster is working correctly.
The `kubectl` binary is under the `cluster/ubuntu/binaries` directory.
You can make it available via PATH, then you can use the below command smoothly.

For example, use `$ kubectl get nodes` to see if all of your nodes are ready.

```console
$ kubectl get nodes
NAME            LABELS                                 STATUS
104.10.103.162   kubernetes.io/hostname=104.10.103.162   Ready
104.10.103.223   kubernetes.io/hostname=104.10.103.223   Ready
104.10.103.250   kubernetes.io/hostname=104.10.103.250   Ready
```

Deploy k8petstore by following README for k8petstore and check the functionaility for it on opencontrail

You can run Kubernetes [guest-example](../../examples/guestbook-go/) to build a redis backend cluster talking to redis-master
and a frontend guestbook app. Few changes are required for this app to work in opencontrail environment.
Please apply the patch from @ https://github.com/Juniper/contrail-kubernetes/blob/vrouter-manifest/cluster/patch_guest_book

Follow steps below to apply patch:

1. cd to kubernetes base directory
2. wget https://raw.githubusercontent.com/Juniper/contrail-kubernetes/vrouter-manifest/cluster/patch_guest_book
3. from kubernetes base direcrtory run the commands:<br />
   3a. `git apply --stat patch_guest_book` <br />
   3b. `git apply --check patch_guest_book` <br />
   If 3a and 3b are sucessfull and have no errors, please apply the patch with command below:

4. `git apply patch_guest_book`

5. Copy the guestbook-go app to the master where you have access to kubectl and deploy the app.
6. Follow instructions from https://github.com/kubernetes/kubernetes/blob/master/examples/guestbook-go/README.md

Please note the required steps for deploying are:

`kubectl create -f guestbook-go/redis-master-controller.json` <br />
`kubectl create -f guestbook-go/redis-master-service.json` <br />

`kubectl create -f guestbook-go/redis-slave-controller.json` <br />
`kubectl create -f guestbook-go/redis-slave-service.json` <br />

`kubectl create -f guestbook-go/guestbook-controller.json` <br />
`kubectl create -f guestbook-go/guestbook-service.json` <br />


After some time, you can use `$ kubectl get pods --namespace=kube-system` to see the redis master, redis slave and the guestbook controller pods running

You can then create SSH tunnel to port 3000 and access guestbook app from browser and check the functionality

### TODO

We are working on the following features which we'd like to let everybody know:

1. External route to kube-ui
2. Upstreaming the code to kubernetes

### Trouble shooting

Salt provisioning for opencontrail follows this model:

1. kube-up calls salt and based on the role salt provisions modules.
   The following new modules are added to support opencontrail:
   - opencontrail-networking-master
   - opencontrail-kubelet-plugin
   - opencontrail-vrouter-kernel
   - opencontrail-networking-minion
   - opencontrail-networking-gateway

2. Initialization of these modules will execute script that works it thru high state
3. log files for the scripts can be found at /var/log/contrail
4. Successful runs of the provisioning scripts for opencontrail will
   have a corresponding .ok file generated in /etc/contrail
   If this file is not found then there could be issues in provisioning.

   Provisioning log provides details of the errors during provisioning
5. In case of erors and salt-call exists, you can re-trigger salt for the specific module
   Ex: salt-call --local state.sls docker,kubelet,kube-apiserver concurrent=True

   Above call will re-provision kube-apiserver.
   Please note concurrent salt-calls are supported only from Salt 2014.7 (Helium)

   In case you find a need to re-run salt and re-provision the cluster, ofcourse with the fix,
   please use the following command from either master, node or gateway.

   `$ salt-call --local state.highstate`

6. Please report the problem and any help required to sanjua@juniper.net, anantha@juniper.net for support

To simply tear down the cluster and re-create it use:

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-down.sh
$ KUBERNETES_PROVIDER=ubuntu ./kube-up.sh
```

kube-down.sh will clean up all the resources (disks, external-ips, routes, etc) used by kubernetes cluster and kubernetes cluster instances.

## Upgrading a Cluster

Testing for upgrade is work in progress. However it will fall in the framework of using kube-push to upgrade the opencontrail components.

```console
$ KUBERNETES_PROVIDER=ubuntu ./kube-push.sh
```

Will post the details on upgrade soon.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/opencontrail.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
