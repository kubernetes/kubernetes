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

# Master <-> Node Communication

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Master <-> Node Communication](#master---node-communication)
  - [Summary](#summary)
  - [Cluster -> Master](#cluster---master)
  - [Master -> Cluster](#master---cluster)
    - [SSH Tunnels](#ssh-tunnels)

<!-- END MUNGE: GENERATED_TOC -->

## Summary

This document catalogs the communication paths between the master (really the
apiserver) and the Kubernetes cluster. The intent is to allow users to
customize their installation to harden the network configuration such that
the cluster can be run on an untrusted network (or on fully public IPs on a
cloud provider).

## Cluster -> Master

All communication paths from the cluster to the master terminate at the
apiserver (none of the other master components are designed to expose remote
services). In a typical deployment, the apiserver is configured to listen for
remote connections on a secure HTTPS port (443) with one or more forms of
client [authentication](authentication.md) enabled.

Nodes should be provisioned with the public root certificate for the cluster
such that they can connect securely to the apiserver along with valid client
credentials. For example, on a default GCE deployment, the client credentials
provided to the kubelet are in the form of a client certificate. Pods that
wish to connect to the apiserver can do so securely by leveraging a service
account so that Kubernetes will automatically inject the public root
certificate and a valid bearer token into the pod when it is instantiated.
The `kubernetes` service (in all namespaces) is configured with a virtual IP
address that is redirected (via kube-proxy) to the HTTPS endpoint on the
apiserver.

The master components communicate with the cluster apiserver over the
insecure (not encrypted or authenticated) port. This port is typically only
exposed on the localhost interface of the master machine, so that the master
components, all running on the same machine, can communicate with the
cluster apiserver. Over time, the master components will be migrated to use
the secure port with authentication and authorization (see
[#13598](https://github.com/kubernetes/kubernetes/issues/13598)).

As a result, the default operating mode for connections from the cluster
(nodes and pods running on the nodes) to the master is secured by default
and can run over untrusted and/or public networks.

## Master -> Cluster

There are two primary communication paths from the master (apiserver) to the
cluster. The first is from the apiserver to the kubelet process which runs on
each node in the cluster. The second is from the apiserver to any node, pod,
or service through the apiserver's proxy functionality.

The connections from the apiserver to the kubelet are used for fetching logs
for pods, attaching (through kubectl) to running pods, and using the kubelet's
port-forwarding functionality. These connections terminate at the kubelet's
HTTPS endpoint, which is typically using a self-signed certificate, and
ignore the certificate presented by the kubelet (although you can override this
behavior by specifying the `--kubelet-certificate-authority`,
`--kubelet-client-certificate`, and `--kubelet-client-key` flags when starting
the cluster apiserver). By default, these connections **are not currently safe**
to run over untrusted and/or public networks as they are subject to
man-in-the-middle attacks.

The connections from the apiserver to a node, pod, or service default to plain
HTTP connections and are therefore neither authenticated nor encrypted. They
can be run over a secure HTTPS connection by prefixing `https:` to the node,
pod, or service name in the API URL, but they will not validate the certificate
provided by the HTTPS endpoint nor provide client credentials so while the
connection will by encrypted, it will not provide any guarantees of integrity.
These connections **are not currently safe** to run over untrusted and/or
public networks.

### SSH Tunnels

[Google Container Engine](https://cloud.google.com/container-engine/docs/) uses
SSH tunnels to protect the Master -> Cluster communication paths. In this
configuration, the apiserver initiates an SSH tunnel to each node in the
cluster (connecting to the ssh server listening on port 22) and passes all
traffic destined for a kubelet, node, pod, or service through the tunnel.
This tunnel ensures that the traffic is not exposed outside of the private
GCE network in which the cluster is running.






<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/master-node-communication.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
