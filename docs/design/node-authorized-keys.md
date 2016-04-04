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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Accessing Kubernetes Nodes over SSH

Kubernetes clusters are often deployment and upgraded over SSH. This
may include the node operating system, the kublet, and so on. SSH tunnels
are also optionally used for node/proxy communication.

This SSH access information should be tracked and configured along with other
Node information, to allow concepts such as self-registering nodes, clusters
with heterogenous Nodes, common deployment and cluster upgrade code, and so on.

## Authorized Public Keys

The SSH public keys authorized to authenticate on a node are present in the Node resource.
When granted access to modify the Node resource, an API caller can add a public key for
access to an aribrary node.

The sshd_config file can have an AuthorizedKeysCommand option configured to accept the
keys listed in the Node resource to grant access to the node instances. It also caches
the authorized keys it found in the Node structure so that if API access to the Node
resource fails then the last available data is used.

A public key can be set to be valid until a specific time. These are no longer accepted
after that point, and purged periodically.

Resource outline:

```
type NodeSpec struct {
	...
	NodeAuthorizedKeys []NodePublicKey `json:"nodeAuthorizedKeys",omitempty`
	...
}

type NodePublicKey struct {
	// Type starting with 'ssh' is an SSH public key
	Type string `json:"type"`
	PublicKey string `json:"publicKey"`
        Until `json:"until,omitempty"`
}
```

Although the examples are targetted at SSH keys, the data structure above should be future
proof to accept other string encoded keys.

**Proof of Concept**: https://github.com/stefwalter/authorized-kube-keys

## Node Host Keys

In addition to granting access, ensuring that the correct node is reached is important.
The NodeStatus structure contains the known host keys, and this information is updated
by the kubelet.

```
type NodeSystemInfo struct {
	...
	NodeHostKeys []NodePublicKey `json:"nodeHostKeys",omitempty`
	...
}
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/node-authorized-keys.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
