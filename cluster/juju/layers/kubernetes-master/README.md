# Kubernetes-master

[Kubernetes](http://kubernetes.io/) is an open source system for managing 
application containers across a cluster of hosts. The Kubernetes project was
started by Google in 2014, combining the experience of running production 
workloads combined with best practices from the community.

The Kubernetes project defines some new terms that may be unfamiliar to users
or operators. For more information please refer to the concept guide in the 
[getting started guide](https://kubernetes.io/docs/home/).

This charm is an encapsulation of the Kubernetes master processes and the 
operations to run on any cloud for the entire lifecycle of the cluster.

This charm is built from other charm layers using the Juju reactive framework.
The other layers focus on specific subset of operations making this layer 
specific to operations of Kubernetes master processes.

# Deployment

This charm is not fully functional when deployed by itself. It requires other
charms to model a complete Kubernetes cluster. A Kubernetes cluster needs a
distributed key value store such as [Etcd](https://coreos.com/etcd/) and the
kubernetes-worker charm which delivers the Kubernetes node services. A cluster
requires a Software Defined Network (SDN) and Transport Layer Security (TLS) so
the components in a cluster communicate securely. 

Please take a look at the [Canonical Distribution of Kubernetes](https://jujucharms.com/canonical-kubernetes/) 
or the [Kubernetes core](https://jujucharms.com/kubernetes-core/) bundles for 
examples of complete models of Kubernetes clusters.

# Resources

The kubernetes-master charm takes advantage of the [Juju Resources](https://jujucharms.com/docs/2.0/developer-resources) 
feature to deliver the Kubernetes software.

In deployments on public clouds the Charm Store provides the resource to the
charm automatically with no user intervention. Some environments with strict
firewall rules may not be able to contact the Charm Store. In these network
restricted  environments the resource can be uploaded to the model by the Juju
operator.

# Configuration

This charm supports some configuration options to set up a Kubernetes cluster 
that works in your environment:

#### dns_domain

The domain name to use for the Kubernetes cluster for DNS.

#### enable-dashboard-addons

Enables the installation of Kubernetes dashboard, Heapster, Grafana, and
InfluxDB.

#### enable-rbac

Enable RBAC and Node authorisation.

# DNS for the cluster

The DNS add-on allows the pods to have a DNS names in addition to IP addresses.
The Kubernetes cluster DNS server (based off the SkyDNS library) supports 
forward lookups (A records), service lookups (SRV records) and reverse IP 
address lookups (PTR records). More information about the DNS can be obtained
from the [Kubernetes DNS admin guide](http://kubernetes.io/docs/admin/dns/).

# Actions

The kubernetes-master charm models a few one time operations called 
[Juju actions](https://jujucharms.com/docs/stable/actions) that can be run by
Juju users.

#### create-rbd-pv

This action creates RADOS Block Device (RBD) in Ceph and defines a Persistent
Volume in Kubernetes so the containers can use durable storage. This action
requires a relation to the ceph-mon charm before it can create the volume.

#### restart

This action restarts the master processes `kube-apiserver`, 
`kube-controller-manager`, and `kube-scheduler` when the user needs a restart.

# More information

 - [Kubernetes github project](https://github.com/kubernetes/kubernetes)
 - [Kubernetes issue tracker](https://github.com/kubernetes/kubernetes/issues)
 - [Kubernetes documentation](http://kubernetes.io/docs/)
 - [Kubernetes releases](https://github.com/kubernetes/kubernetes/releases)

# Contact

The kubernetes-master charm is free and open source operations created
by the containers team at Canonical. 

Canonical also offers enterprise support and customization services. Please
refer to the [Kubernetes product page](https://www.ubuntu.com/cloud/kubernetes)
for more details.
