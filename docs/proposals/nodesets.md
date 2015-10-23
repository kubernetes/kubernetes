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
[here](http://releases.k8s.io/release-1.0/docs/proposals/nodesets.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# NodeSets in Kubernetes

**Author**: Justin Santa Barbara

**Status**: Proposal.

This document presents the design of the Kubernetes NodeSet, describes use cases, and gives an overview of the code.

## Motivation

The official bring-up procedure for a kubernetes cluster is to use kube-up.  This is a pretty complicated bash script
that is largely separate for each provider.  It offers limited functionality: you cannot change the number of nodes
directly, nor change the instance type used.  Many people resort to alternative methods of installation,
to varying degrees of success.

NodeSets put node launching under the control of the master.  They allow the operator to create
and destroy NodeSets dynamically, or to change the number of instances in a NodeSet.  They
allow for easier upgrades by launching nodes with different OS/k8s versions.

This simplifies the Kubernetes installation, because the primary requirement is simply to launch a master.  Then
the installation script can either automatically create a NodeSet, or leave this up to the operator post-creation.

## Use Cases

### Resize number of Nodes through the API

By updating the count on a NodeSet, nodes can be launched or destroyed.  Users will no longer have
to go to the underlying cloud API.  Kubernetes can likely make better decisions about which instance to
terminate than a cloud-implemented auto-scaling group can, because it is more aware of the workloads.

### Consistent auto-scaling across providers

In future we can add auto-scaling functionality at the Kubernetes level (instead of at the cloud level), so
that it works consistently across clouds, and can be based on Kubernetes metrics instead of cloud-metrics
(i.e. it can take account of pods _not_ scheduled).

### Support multiple NodeSets of different instance types

Users may want to run instances of different types in the same cluster, permanently or transiently.
Permanently mixing instance types may require more scheduler support, but for transient usage (changing
instance sizes) this is helpful for changing a cluster.

Users may also want to mix instance types in a way that is not visible to the cluster, for example mixing
on-demand and spot instances on AWS.

### Support upgrading by specifying software versions

By specifying that a new NodeSet should be launched with a different image or kubernetes version,
users can easily perform an online upgrade of their nodes (within compatible versions).

## Functionality

The NodeSet supports standard API features:
- create
  - The spec for NodeSet includes a count of nodes
  - The spec includes a template, which specifies the type of nodes that will be launched as labels.
    We imagine that a cloudprovider actually has a pool of machines of various instance types, images,
    kubernetes versions etc.  We will label each node with the labels describing its instance type, running image,
    kubernetes version etc on creation.  Then the selector acts to specify the information that nodes must have.
    Thus, specifying `kubernetes.io/instanceType=n1-highmem-16` will ensure that nodes are launched with
    `n1-highmem-16` instance type.  Selectors with the `kubernetes.io/` prefix must be known, or it is a
    validation error; selectors without the prefix are treated as normal labels.
    (TODO: Is this too 'cute'?)
  - YAML example:

```YAML
  apiVersion: v1
  kind: NodeSet
  metadata:
    labels:
      bornondate: 2015-10-23
    name: bigmemory-node-pool1
  spec:
    count: 4
    template:
      metadata:
        labels:
          kubernetes.io/instanceType: n1-highmem-16
```

  - commands that get info
    - get (e.g. kubectl get nodesets)
    - describe
  - Modifiers
    - delete
    - label
	- annotate
    - update operations like patch and replace (only allowed to NodeSet metadata.labels and spec.count)
  - In general, for all the supported features like get, describe, update, etc, the NodeSet works in a similar way to the Replication Controller.


## Design

#### Client

- Add support for NodeSet commands to kubectl and the client. Client code will be added to the experimental area of client/unversioned.

#### Apiserver

- Accept, parse, validate client commands
- REST API calls are handled in registry/daemon
  - In particular, the api server will add the object to etcd
  - NodeSetManager listens for updates to etcd (using Framework.informer)
- API objects & validation for NodeSet will be created alongside other experimental types.

#### NodeSetManager

- Creates new nodes and deletes nodes in response to changes as observed.
  Will delegate to the cloud provider for actually creating; initial implementation
  will cover GCE & AWS, will likely initially delegate in turn to GCE MIGs or AWS ASGs.

#### Kubelet

- Will need to be modified to add labels to Nodes on bring-up, so that the consistency of the selector-as-creator
  model is maintained.






<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/nodesets.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
