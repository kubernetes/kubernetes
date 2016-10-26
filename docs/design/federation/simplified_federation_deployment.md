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
**Last updated**: 10/26/2016  
**Status**: **Draft**|Approved|Abandoned|Obsolete


**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Simplified Federation Control Plane Deployment](#simplified-federation-control-plane-deployment)
  - [Background](#background)
  - [Goals](#goals)
  - [User Experience](#user-experience)
      - [Introducing `kubefed`](#introducing-kubefed)
      - [Federation Control Plane Deployment Experience](#federation-control-plane-deployment-experience)
        - [Deploying to an existing cluster](#deploying-to-an-existing-cluster)
        - [Deploying to a new cluster at the time of cluster bootstrap](#deploying-to-a-new-cluster-at-the-time-of-cluster-bootstrap)
      - [Cluster Registration Experience](#cluster-registration-experience)
      - [Cluster Deregistration Experience](#cluster-deregistration-experience)
    - [Points for Discussion:](#points-for-discussion)
  - [Design](#design)
      - [Federation Control Plane Bootstrap](#federation-control-plane-bootstrap)
        - [Additional enhancements](#additional-enhancements)
        - [Caveats](#caveats)
      - [Cluster Registration/Deregistration](#cluster-registrationderegistration)
        - [Registration](#registration)
        - [Deregistration](#deregistration)
  - [Test Plan](#test-plan)
      - [Federation Control Plane Bootstrap](#federation-control-plane-bootstrap-1)
      - [Cluster Registration/Deregistration](#cluster-registrationderegistration-1)
  - [Timeline](#timeline)
  - [Maturity Level](#maturity-level)

<!-- END MUNGE: GENERATED_TOC -->

## Background

Kubernetes Cluster Federation has its own control plane which is similar
to the control plane of a Kubernetes cluster. Like Kubernetes, a
federation control plane is composed of an API server and a controller
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

**Side note**: It is a mouthful to always prefix "federation" to each of
its control plane components. But it is necessary to disambiguate these
components from the almost identical components in a Kubernetes cluster.

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

#### Introducing `kubefed`

In the wake of the discussion in
[Issue #35226](https://github.com/kubernetes/kubernetes/issues/35226#issuecomment-255207042),
we are introducing a new command line tool called `kubefed` that allows
users to perform federation-specific actions. `kubefed` stands for
"kubernetes federatate". In the first implementation, users will be
able to deploy federation control plane onto exiting clusters and
register/deregister clusters with/from federation with `kubefed`.


#### Federation Control Plane Deployment Experience

Users might want to either deploy a federation control plane onto
existing Kubernetes clusters or deploy it while bootstrapping a new
cluster. We want to provide mechanisms for both these cases.


##### Deploying to an existing cluster

When a user already has one or more Kubernetes clusters running and
would like to deploy federation control plane onto one of those
clusters, they use the `kubefed` CLI to achieve that.

1. User downloads the `kubefed` executable on to their machine.
2. Runs:

    ```shell
    $ kubefed init myfederation --host=mycluster
    ```

  to deploy a federation control plane onto the cluster whose kubeconfig
  context name is "mycluster".

**HA Note:** Since the question of how this experience could be extended
to an HA federation control plane deployment has come up in the review a
number of times, I am just leaving a note here. For HA deployments,
`--host` flag will be extended into an array flag that could be
specified multiple times on the command line to deploy the federation
control plane onto all the listed clusters. More details to come later
in a design doc that is specific to HA federation control plane
deployments.

**Note:** Users can download `kubefed` and run these command on any
machine. This could either be their local computer which they use for
their day-to- day activities, a physical machine or a VM where they run
their Kubernetes cluster or any other machine. The only requirement is a
kubeconfig file containing the credentials of the cluster on to which
they are deploying the federation control plane, also otherwise called
as the "host cluster".


##### Deploying to a new cluster at the time of cluster bootstrap

Kubernetes v1.4 shipped a new command line tool called
[`kubeadm`](http://kubernetes.io/docs/getting-started-guides/kubeadm/).
`kubeadm` simplifies Kubernetes control plane deployment. Federation
control plane deployment builds on top of this tool for the case where
users prefer to deploy the federation control plane while bootstrapping
a new cluster.

1. User provisions their cloud/compute resources: a VM or a physical
   machine.
2. Downloads the
   [`kubeadm`](http://kubernetes.io/docs/getting-started-guides/kubeadm/)
   executable on to each of those machines, virtual or physical.
3. Runs:

    ```shell
    $ kubeadm federation init myfederation
    ```

   to set up a Kubernetes control plane and a federation control plane
   on that machine.


#### Cluster Registration Experience

Cluster registration and deregistration will be facilitated through
`kubefed`.

1. User downloads the `kubefed` executable on to their local computer,
   i.e. the computer they use for their day-to-day activities.
2. Runs:

    ```shell
    $ kubefed join mycluster
    ```

   to register their cluster named `mycluster`, where `mycluster` is the
   name of the cluster's context in the local kubeconfig.

**Note:** Like in the case of federation control plane deployment above,
users can download `kubefed` and run these commands on any machine as
long as they have a kubeconfig file with the required credentials.


#### Cluster Deregistration Experience

1. To deregister a cluster, user runs:

    ```shell
    $ kubefed unjoin mycluster
    ```

### Points for Discussion:

1. Is `unregister` a better verb than `deregister`?

A. This document previously proposed `register/deregister` as the verb
   pair for cluster registration/deregistration and that has been
   replaced by `join/unjoin` now. See the next question.

2. Does the verb pair `join/unjoin` make more sense in this context?

A. After quite a bit of debate and deliberation, we, the federation team
   have come to a conclusion that `join/unjoin` is a better verb pair
   than register/deregister.


## Design


#### Federation Control Plane Bootstrap

Federation control plane bootstrap works the same way whether it is
initiated through `kubefed` or `kubeadm`. The only difference is that
`kubeadm init` initializes a Kubernetes control plane. In order to do
that, it generates the manifests for the various control plane
components and starts them. It also generates the certificates and the
credentials required for those components to interact with the API
server.

Federation control plane deployment builds on top of this functionality.
Running `kubeadm federation init` first performs all the operations
involved in bootstrapping a Kubernetes control plane, as running
`kubeadm init` would. We henceforth call this Kubernetes cluster for
which the control plane was initialized, the "host cluster".

To bootstrap a federation control plane, we perform the following
operations:

**1. Create a namespace for federation system components**

* Call this namespace `federation-system` to keep it consistent with
  Kubernetes.
* The namespace is created in the bootstrap cluster.

**2. Expose a network endpoint for the federation API server**

* Create a "Loadbalancer" type Kubernetes service in the
  `federation-system` namespace in the bootstrap cluster to expose the
  yet to be created federation API server.
* Wait until a load balancer IP address is allocated to the service.

**3. Generate TLS certificates and credentials**

* Using `kubeadm`'s PKI infrastructure, generate TLS certificates for a
  new CA, yet to be created federation API server and federation
  controller manager.
* Generate credentials for the federation API server.

**4. Create a kubeconfig secret**

* Using the cluster-local DNS name of the federation API server and
  the generated certificates and credentials, generate a kubeconfig
  file for the federation controller manager.
* Create a secret in the bootstrap cluster's `federation-system`
  namespace and populate it with the contents of the generated
  kubeconfig file.

**5. Create a persistent volume and a claim to store the federation API server's state**

* Create a persistent volume claim in the `federation-system` namespace.
* Also request a dynamically provisioned persistent storage device for
  the PVC.

**6. Create federation API server**

* Create a deployment for the federation API server in the `federation-
  system` namespace.
* The pod template spec is composed of two containers: one for `etcd`
  and the other one for the federation API server.
* The pod template spec references the PVC created in the previous step
  as a volume and the `etcd` container mounts that volume to store
  `etcd`'s data.
* The federation API server container uses the official `hyperkube`
  release image.
* Since the API server is stateless, it needs no persistent storage.
  However, the certificates and the credentials are made available to
  the API server by mounting them as volumes inside the API server
  container.

**7. Create federation controller manager**

* Create a deployment for the federation controller manager in the
  `federation-system` namespace.
* The pod template spec consists of a single container for the
  federation controller manager and it also uses the same official
  `hyperkube` release image as the federation API server.
* Kubeconfig required by the federation controller manager to discover
  and authenticate with the federation API server is referenced as a
  secret in the pod template spec and mounted as a volume inside the
  container.
* The mounted kubeconfig file is passed as an argument to the federation
  controller manager's `--kubeconfig` flag.

##### Additional enhancements

Users can pass `--join` as an additional flag to
`kubeadm federation init` to also register the cluster with federation.
See the
[Cluster Registration/Deregistration](#cluster-registrationderegistration)
section below for details on how this is done.

##### Caveats

* In the first phase of this implementation, we only create a single
  instance of `etcd`. This neither makes `etcd` nor the federation
  control plane HA. HA federation control plane is on the road map and
  will arrive in one of the future releases.
* Federation control plane components aren't started until additional
  nodes join the bootstrap cluster. In other words, these components
  aren't pinned to the bootstrap cluster's master node. We just create
  their manifests at this stage and they are started when additional
  schedulable nodes join the bootstrap cluster.


#### Cluster Registration/Deregistration

`kubectl` is Kubernetes' client-side CLI used for ongoing day-to-day
interactions with a set of clusters. `kubectl` is already being extended
to enable interactions with federation. Users can also already use
`kubectl` to register clusters with and deregister clusters from
federation. As already described, this process is manual, tedious and
error-prone. Hence, we aim to reduce these operations into a single
command each using the `kubefed` tool.

##### Registration

When a user runs `kubefed join <cluster-context-name> --host=<host-
cluster-context>`, we perform the following operations assuming that the
current kubeconfig context is a federation endpoint:

**1. Create a kubeconfig secret for the cluster in federation**

* Read the kubeconfig fields: `cluster`, `user` and `context`
  corresponding to the context `<cluster-context-name>` into a separate
  kubeconfig and create a secret for that in the the host cluster
  indicated by the `--host` flag.

**2. Retrieve the cluster endpoint**

* Read the API server endpoint for `<cluster-context-name>` from the
  `cluster` field in the local kubeconfig file.

**3. Create the cluster resource in federation**

* Using the `<cluster-context-name>`'s API server endpoint and the
  kubeconfig secret name, create the cluster API resource in federation.

##### Deregistration

When a user runs `kubefed unjoin <cluster-context-name> --host =<host-
cluster-context>`, we perform the following operations, again assuming
that the current kubeconfig context is a federation endpoint:

**1. Remove the cluster resource from federation**

* Remove the cluster API resource corresponding to `<cluster-context-
name>` from federation.

**2. Remove the credentials secret**

* Remove the kubeconfig secret that we created during the registration
  process from the host cluster.

**Note:** `--host` flag is only required because federation still
doesn't support selecting specific clusters in federated secrets. It
copies them to all the underlying clusters. It is unnecessary to copy
the credentials in clusters other than the host cluster. And since these
are secrets that include cluster credentials, we want to err on the side
of being too conservative. This flag can be deprecated once federated
secrets gain cluster selection support.

## Test Plan

#### Federation Control Plane Bootstrap

End-to-end testing of `kubeadm` functionality is somewhat tricky because
`kubeadm init` and hence `kubeadm federation init` prints out a token to
STDOUT that must be used in all the `kubeadm join` invocations on the
non-master Kubernetes nodes. Given that, here is an overview of how we
plan to test the federation bootstrap functionality.

**1. kubernetes-anywhere for resource provisioning**

* We use
  [kubernetes-anywhere](https://github.com/kubernetes/kubernetes-anywhere)
  to provision cloud resources, i.e. VMs, Networking etc.

**2. Provide a customized VM startup script**

* We provide a customized `configure-vm.sh` script to run during VM
  startup.
* On the master VM, this script runs `kubeadm federation init` which
  prints out a token.
* The token is written to an object store (GCS on GCP and S3 on AWS).
* On the non-master nodes, `configure-vm.sh` script polls for the token
  object in the pre-configured storage bucket.
* Once that appears, it uses the token to run `kubeadm join <token>`.

**3. Verify that the federation control plane is up and running**

* We verify that the federation control plane pods are up and are in
  "Running" state.
* We verify that the request to a discovery endpoint on the federation
  API server returns a valid response.

#### Cluster Registration/Deregistration

This should be as simple as running `kubefed join <cluster-context>`
and verifying that the registered cluster reports "Ready" status.

Running `kubectl get clusters` should not list the unregistered cluster.


## Timeline

Both the federation control plane deployment functionality and the
cluster registration/deregistration functionality via `kubefed` are
planned to be implemented in the v1.5 time frame and are considered
a P0 feature for v1.5.


## Maturity Level

Both functionalities will be labeled "alpha" in v1.5.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/federation/simplified_federation_deployment.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
