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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Simplified Federation Control Plane Deployment

**Author**: Madhusudan C.S. <madhusudancs@google.com>  
**Last updated**: 10/07/2016  
**Status**: **Draft**|Approved|Abandoned|Obsolete


**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Simplified Federation Control Plane Deployment](#simplified-federation-control-plane-deployment)
  - [Background](#background)
  - [Goals](#goals)
  - [User Experience](#user-experience)
        - [Federation Control Plane Deployment Experience](#federation-control-plane-deployment-experience)
        - [Cluster Registration/Deregistration Experience](#cluster-registrationderegistration-experience)
  - [Design](#design)
        - [Federation Control Plane Deployment](#federation-control-plane-deployment)
        - [Cluster Registration/Deregistration](#cluster-registrationderegistration)
  - [Timeline](#timeline)
  - [Maturity Level](#maturity-level)

<!-- END MUNGE: GENERATED_TOC -->

## Background

Kubernetes Cluster Federation has its own control plane which is similar
to the control plane of a Kubernetes cluster. Like Kubernetes, a
federation control plane is composed an API server and a controller
manager that bundles all the core controllers. While that looks simple
on paper, deploying a federation control plane comes with its own set of
challenges. Deploying a federation control plane not just involves
deploying a federation API server and a federation controller manager,
but also other components that support the control plane. This includes:

1. A persistent store to store the state of the federation - `etcd`.
2. A persistent device to store `etcd` data.
3. A secret store to store the federation API server credentials.

And these components must be setup in such a way that they can
appropriately coordinate with each other. While this is already starting
to look complicated, this is only one part of the problem.

Kubernetes Cluster Federation operates only on the clusters that it is
responsible for. So once a federation control plane is deployed, for it
to be useful, users should be able to register and deregister Kubernetes
clusters with/from their federation. In order to register a cluster,
users should supply the cluster's credentials and its API server
endpoint.


## Goals

Today, we ship bash scripts to facilitate the federation control plane
deployment process and the registration/deregistration process is mostly
manual. Both these processes involve multiple coordinated steps and are
error prone. For example, to register a Kubernetes cluster with
federation users have to write a yaml file describing a cluster
resource. They have to specify the cluster's endpoint in the right
format. That field is too restrictive and hard to get it right. They
also have to correctly specify where the credentials for the cluster is
stored.

Therefore, the goal here is to simplify both the federation control
plane deployment experience and the cluster registration and
deregistration process. Each of these processes will be reduced to a
single command with additional flags for customization.


## User Experience

Kubernetes v1.4 shipped a new command line tool called
[`kubeadm`](http://kubernetes.io/docs/getting-started-guides/kubeadm/).
`kubeadm` simplifies Kubernetes control plane deployment. Federation
control plane deployment builds on top of this tool. Cluster
registration and deregistration will be facilitated through
[`kubectl`](http://kubernetes.io/docs/user-guide/kubectl-overview/).


##### Federation Control Plane Deployment Experience

1. User provisions their cloud/compute resources: a VM or a physical
   machine.
2. Downloads the
   [`kubeadm`](http://kubernetes.io/docs/getting-started-guides/kubeadm/)
   executable on to each of those machines, virtual or physical.
3. Runs:

    ```shell
    $ kubeadm init federation
    ```

   to set up a Kubernetes control plane and a federation control plane
   on that machine.


##### Cluster Registration/Deregistration Experience

1. User downloads the
   [`kubectl`](http://kubernetes.io/docs/getting-started-guides/kubeadm/)
   executable on their local computer, i.e. the computer they use for
   their day-to-day activities.
2. Runs:

    ```shell
    $ kubectl register mycluster
    ```

   to register their cluster named `mycluster`.


## Design


##### Federation Control Plane Deployment




##### Cluster Registration/Deregistration


## Test Plan



## Timeline

Both the federation control plane deployment functionality via `kubeadm`
and the cluster registration/deregistration functionality via `kubectl`
are planned to be implemented in the v1.5 timeframe and are considered a
P0 feature for v1.5.


## Maturity Level

Both functionalities will be labelled "alpha" in v1.5.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/simplified_federation_deployment.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
