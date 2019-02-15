# Kubemark Pre-existing Provider Guide

**Kubemark Master**
- A set of Kubernetes control plane components running in a VM

**Kubernetes Cluster**
- A real Kubernetes Cluster that has master and nodes. The hollow-node pods
  are run in this cluster, but appear as nodes to the Kubemark Master

## Introduction

Every running Kubemark setup looks like the following:
 1) A running Kubernetes cluster pointed to by the local kubeconfig
 2) A separate VM where the Kubemark Master is running
 3) Some hollow-nodes that run on the Kubernetes Cluster from #1
 4) The hollow-nodes are configured to talk with the Kubemark Master at #2

When using the pre-existing provider, the developer is responsible for creating
#1 and #2.  Therefore, the Kubemark scripts will not create any infrastructure
or start a Kubemark Master like in other providers. Instead, the existing
resources provided by the VM at $MASTER_IP will serve as the Kubemark Master.

## Use Case

The goal of the pre-existing provider is to use the Kubemark tools with an
existing kubermark master. It's meant to provide the developer with
additional flexibility to customize the cluster infrastructure and still use
the Kubemark setup tools.  The pre-existing provider is an **advanced** use
case that requires the developer to have knowledge of setting up a Kubemark
Master.

## Requirements

To use the pre-existing provider, the expectation is that there's a Kubemark
Master that is reachable at $MASTER_IP. The machine that the Kubemark Master is
on has to be ssh able from the host that's executing the Kubemark scripts. And
the user on that machine has to be 'kubernetes'.

Requirement checklist:
- Set MASTER_IP to ip address to the Kubemark Master
- The host where you execute the Kubemark scripts must be able to ssh to
  kubernetes@$MASTER_IP

## Example Configuration

_test/kubemark/cloud-provider-config.sh_

```
CLOUD_PROVIDER="pre-existing"
KUBEMARK_IMAGE_MAKE_TARGET="push"
CONTAINER_REGISTRY=docker.io
PROJECT="rthallisey"
MASTER_IP="192.168.121.29:6443"
```
