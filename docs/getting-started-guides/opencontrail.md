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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/ubuntu.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Opencontrail Networking for Kubernetes Deployment in GCE
--------------------------------------------------------

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Starting a Cluster](#starting-a-cluster)
    - [Set up working directory](#set-up-working-directory)
    - [Configure and start the kubernetes cluster](#configure-and-start-the-kubernetes-cluster)
    - [Test it out](#test-it-out)
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

Kubernetes-Master:

Opencontrail controller modules in the container are deployed and configured on the master. 
Contrail contnaiers on the master are:

root@kubernetes-master:~# docker ps |grep contrail | grep -v pause
8d41e850e55b        opencontrail/kube-network-manager                                                   "/go/kube-network-man"   56 minutes ago      Up 56 minutes                           k8s_kube-network-manager.932351ab_kube-network-manager-kubernetes-master_default_d2be274a13c05a0d7e8a683587e18586_8515e494
f5cd41ff1503        opencontrail/web:2.20                                                               "/usr/bin/contrail-we"   57 minutes ago      Up 57 minutes                           k8s_contrail-web.c14fc93d_contrail-web-kubernetes-master_default_e217b52ab60719f2394e4b1403511132_1a383cfc
8aa226fbc877        opencontrail/config:2.20                                                            "/usr/bin/contrail-sc"   57 minutes ago      Up 57 minutes                           k8s_contrail-schema.bceb4ca0_contrail-schema-kubernetes-master_default_c3a3c58883dc5e3cd17d1432f0d4208e_905491ad
bd9cbb8b5cb5        opencontrail/control:2.20                                                           "/usr/bin/contrail-co"   57 minutes ago      Up 56 minutes                           k8s_contrail-control.9d2ce2c5_contrail-control-kubernetes-master_default_a725f6e0bd53a8239a6764761384b60f_1f9c48dc
b95a13b61d40        opencontrail/ifmap-server:2.20                                                      "/entrypoint.sh"         57 minutes ago      Up 57 minutes                           k8s_ifmap-server.12ea28bd_ifmap-server-kubernetes-master_default_c8589a0f26652c384cc462b4ca178910_abea9e3f
57c4596c9481        opencontrail/config:2.20                                                            "/usr/bin/contrail-ap"   57 minutes ago      Up 57 minutes                           k8s_contrail-api.6f2e4b6f_contrail-api-kubernetes-master_default_ce0f7e301d5c1515f055acdcccf579d3_e9446d4c
e68d1eaec682        cassandra:2.2.0                                                                     "/bin/sh -c 'sed -ri "   57 minutes ago      Up 57 minutes                           k8s_opencontrail-config-db.31407271_cassandra-kubernetes-master_default_5b2bd074fdc25365d3856cf7415757c0_5583722c
4c55916f2455        opencontrail/analytics:2.20                                                         "/usr/bin/contrail-co"   57 minutes ago      Up 57 minutes                           k8s_contrail-collector.803551f2_contrail-collector-kubernetes-master_default_23d91f648a03a6b16e8650394bacbebd_7ff480ca
bb42ad2c4c49        opencontrail/analytics:2.20                                                         "/usr/bin/contrail-qu"   57 minutes ago      Up 57 minutes                           k8s_contrail-query-engine.30575520_contrail-query-engine-kubernetes-master_default_e214cb3721ecc1d4430700eaa2090fca_8c572560
3de472ba306d        opencontrail/analytics:2.20                                                         "/usr/bin/contrail-an"   57 minutes ago      Up 57 minutes                           k8s_contrail-analytics-api.b73654b9_contrail-analytics-api-kubernetes-master_default_52820ac14513bce55d15926b90de75cd_745698d7

Details on kube-network-manager and kubernetes on opencontrail can be found @ https://pedrormarques.wordpress.com/2015/07/14/kubernetes-networking-with-opencontrail/ 

Kubernetes-minion:

Opencontrail vrouter agnet in container and kernel module (vrouter.ko) is deployed on the kubernetes cluster nodes.

root@kubernetes-minion-4bfu:~# docker ps |grep contrail | grep -v pause
497a0d6bd096        opencontrail/vrouter-agent:2.20                      "/usr/bin/contrail-vr"   About an hour ago   Up About an hour                        k8s_contrail-vrouter-agent.d97cd63f_contrail-vrouter-agent-kubernetes-minion-4bfu_default_91106db29e22a244cf51102991662f2f_bfb3b90c

root@kubernetes-minion-4bfu:~# lsmod |grep vrouter
vrouter               235766  1 

Opencontrail vrouter agent manages the forwarding path for data. Please find more details on this @ https://github.com/Juniper/contrail-controller/blob/master/src/vnsw/agent/README

Kubernetes-Opencontrail-gateway:

Opencontrail gateway provides gateway fucntion for any external (north-south) traffic going towards the internal kubernetes pods.
Details on this functionality can be found @ https://github.com/Juniper/contrail-controller/wiki/Simple-Gateway

### Test it out

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

Also you can run Kubernetes [guest-example](../../examples/guestbook-go/) to build a redis backend cluster．


### Deploy addons

Assuming you have a starting cluster now, this section will tell you how to deploy addons like DNS
and UI onto the existing cluster.

The configuration of DNS is configured in cluster/ubuntu/config-default.sh.

```sh
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"

DNS_SERVER_IP="10.0.0.10"

DNS_DOMAIN="cluster.local"

DNS_REPLICAS=1
```

The `DNS_SERVER_IP` is defining the ip of dns server which must be in the `SERVICE_CLUSTER_IP_RANGE`.
The `DNS_REPLICAS` describes how many dns pod running in the cluster.

By default, kube-dns addon is deployed with networking provided by opencontrail. Every Pod has connectivity to the DNS pod.


After some time, you can use `$ kubectl get pods --namespace=kube-system` to see the DNS and UI pods are running in the cluster.

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

### Test it out

You can use the `kubectl` command to check if the newly upgraded kubernetes cluster is working correctly.

You can deploy k8petstore by following README for k8petstore and check the functionaility for it on opencontrail

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
