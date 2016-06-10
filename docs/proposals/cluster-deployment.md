<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Objective

Simplify the cluster provisioning process for a cluster with one master and multiple worker nodes.
It should be secured with SSL and have all the default add-ons. There should not be significant
differences in the provisioning process across deployment targets (cloud provider + OS distribution)
once machines meet the node specification.

# Overview

Cluster provisioning can be broken into a number of phases, each with their own exit criteria.
In some cases, multiple phases will be combined together to more seamlessly automate the cluster setup,
but in all cases the phases can be run sequentially to provision a functional cluster.

It is possible that for some platforms we will provide an optimized flow that combines some of the steps
together, but that is out of scope of this document.

# Deployment flow

**Note**: _Exit critieria_ in the following sections are not intended to list all tests that should pass,
rather list those that must pass.

## Step 1: Provision cluster

**Objective**: Create a set of machines (master + nodes) where we will deploy Kubernetes.

For this phase to be completed successfully, the following requirements must be completed for all nodes:
- Basic connectivity between nodes (i.e. nodes can all ping each other)
- Docker installed (and in production setups should be monitored to be always running)
- One of the supported OS

We will provide a node specification conformance test that will verify if provisioning has been successful.

This step is provider specific and will be implemented for each cloud provider + OS distribution separately
using provider specific technology (cloud formation, deployment manager, PXE boot, etc).
Some OS distributions may meet the provisioning criteria without needing to run any post-boot steps as they
ship with all of the requirements for the node specification by default.

**Substeps** (on the GCE example):

1. Create network
2. Create firewall rules to allow communication inside the cluster
3. Create firewall rule to allow ```ssh``` to all machines
4. Create firewall rule to allow ```https``` to master
5. Create persistent disk for master
6. Create static IP address for master
7. Create master machine
8. Create node machines
9. Install docker on all machines

**Exit critera**:

1. Can ```ssh``` to all machines and run a test docker image
2. Can ```ssh``` to master and nodes and ping other machines

## Step 2: Generate certificates

**Objective**: Generate security certificates used to configure secure communication between client, master and nodes

TODO: Enumerate ceritificates which have to be generated.

## Step 3: Deploy master

**Objective**: Run kubelet and all the required components (e.g. etcd, apiserver, scheduler, controllers) on the master machine.

**Substeps**:

1. copy certificates
2. copy manifests for static pods:
	1. etcd
	2. apiserver, controller manager, scheduler
3. run kubelet in docker container (configuration is read from apiserver Config object)
4. run kubelet-checker in docker container

**v1.2 simplifications**:

1. kubelet-runner.sh - we will provide a custom docker image to run kubelet; it will contain
kubelet binary and will run it using ```nsenter``` to workaround problem with mount propagation
1. kubelet config file - we will read kubelet configuration file from disk instead of apiserver; it will
be generated locally and copied to all nodes.

**Exit criteria**:

1. Can run basic API calls (e.g. create, list and delete pods) from the client side (e.g. replication
controller works - user can create RC object and RC manager can create pods based on that)
2. Critical master components works:
  1. scheduler
  2. controller manager

## Step 4: Deploy nodes

**Objective**: Start kubelet on all nodes and configure kubernetes network.
Each node can be deployed separately and the implementation should make it ~impossible to change this assumption.

### Step 4.1: Run kubelet

**Substeps**:

1. copy certificates
2. run kubelet in docker container (configuration is read from apiserver Config object)
3. run kubelet-checker in docker container

**v1.2 simplifications**:

1. kubelet config file - we will read kubelet configuration file from disk instead of apiserver; it will
be generated locally and copied to all nodes.

**Exit critera**:

1. All nodes are registered, but not ready due to lack of kubernetes networking.

### Step 4.2: Setup kubernetes networking

**Objective**: Configure the Kubernetes networking to allow routing requests to pods and services.

To keep default setup consistent across open source deployments we will use Flannel to configure
kubernetes networking. However, implementation of this step will allow to easily plug in different
network solutions.

**Substeps**:

1. copy manifest for flannel server to master machine
2. create a daemonset with flannel daemon (it will read assigned CIDR and configure network appropriately).

**v1.2 simplifications**:

1. flannel daemon will run as a standalone binary (not in docker container)
2. flannel server will assign CIDRs to nodes outside of kubernetes; this will require restarting kubelet
after reconfiguring network bridge on local machine; this will also require running master nad node differently
(```--configure-cbr0=false``` on node and ```--allocate-node-cidrs=false``` on master), which breaks encapsulation
between nodes

**Exit criteria**:

1. Pods correctly created, scheduled, run and accessible from all nodes.

## Step 5: Add daemons

**Objective:** Start all system daemons (e.g. kube-proxy)

**Substeps:**:

1. Create daemonset for kube-proxy

**Exit criteria**:

1. Services work correctly on all nodes.

## Step 6: Add add-ons

**Objective**: Add default add-ons (e.g. dns, dashboard)

**Substeps:**:

1. Create Deployments (and daemonsets if needed) for all add-ons

## Deployment technology

We will use Ansible as the default technology for deployment orchestration. It has low requirements on the cluster machines
and seems to be popular in kubernetes community which will help us to maintain it.

For simpler UX we will provide simple bash scripts that will wrap all basic commands for deployment (e.g. ```up``` or ```down```)

One disadvantage of using Ansible is that it adds a dependency on a machine which runs deployment scripts. We will workaround
this by distributing deployment scripts via a docker image so that user will run the following command to create a cluster:

```docker run gcr.io/google_containers/deploy_kubernetes:v1.2 up --num-nodes=3 --provider=aws```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/cluster-deployment.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
