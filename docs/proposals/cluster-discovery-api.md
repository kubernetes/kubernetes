# Cluster Discovery API

Author: [@mikedanese](https://github.com/mikedanese)

#### Abstract

This document describes a mechanism for a kubelet to discover, from a trusted source, the
location of the cluster API and the public key of the cluster's local root certificate
authority.

#### Motivation

Setting up a kubernetes cluster is hard. Setting up a secure kubernetes cluster is harder.
Public key infrastructure is an area that most of our users are unfamiliar with and may
not care about for many cluster deployments where network securityis not a concern. The
Kubernetes Discovery API provides a mechanism to enable a high level of security by
default without compromising on the UX of the cluster deployment flow.

#### Overview

The Kubelet TLS Bootstrap API reduces the amount of manual steps required to generate and
distribute cryptographic assets required to setup a secure Kubernetes cluster. Still, in
order to initiate the kubelet TLS bootstrap process, the user is required to distribute
the URLs of the api-server and the root CA public key to each node. Today, there is no
mechanism built into Kubernetes to facilitate this.

We can reduce this configuration to a single 256 bit cluster-id that can be easily passed
into the commandline of the kubelet by hand or otherwise. This cluster-id would be used to
lookup a discovery.Cluster object (documented bellow) in a globally addressable and trusted
service (by default https://discovery.k8s.io, but also internally hostable). The default
service's certificate will be signed by a CA whose certificate is packaged by the OS
distribution (i.e a public and trusted CA).

On first boot, kubelet would accept the cluster-id as configuration and use it to lookup the
discovery.Cluster object. The discovery.Cluster object would have the required information to
establish a trusted first contact with the cluster api and initiate the kubelet TLS bootstrap
flow.

#### API Types

Hosted at:

```
PUT https://${DISCOVERY_SERVICE}/v1alpha1/clusters
GET https://${DISCOVERY_SERVICE}/v1alpha1/clusters/<cluster-id>
```

A user can create and retrieve the cluster object:

```
type Cluster struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`
	RootCAPublicKey      []byte   `json:"rootCAPublicKey"`
	ApiServerURLs        []string `json:"apiServerURLs"`
}
```

Cluster objects are immutable after creation.

#### Validating the Integrity of the discovery.Cluster object

We can make the Cluster object self validating if the cluster-id is generated from a consistent
hashing of a subset of the object. For example the cluster-id colud be generated from a sha256
sum of an object containing the fields `.type`, `.apiVersion`, `.metadata.annotations`,
`.rootCAPublicKey`, `.apiServerURLs` serialized to JSON. The administrator or automation tooling
could validate the cluster-id against the discover.Cluster object before asking each kubelet to
join the cluster. Each kubelet could validate cluster-id against the discovery.Cluster object upon
joining the cluster. This adds an extra layer of tamperproofing to the discovery flow and increases
the difficulty of attacking clusters if the discovery API is man in the middled or otherwise
comprimised.

#### Integration with Kubelet TLS Bootstrap

The Kubelet TLS Bootstrap API requires a shared secret that initally authorizes kubelets to post
certificate signing requests to the certificates API. The cluster-id can serve as this shared secret.
Kubelets will use the cluster-id as the Bearer Token for an account that is only authorized (via the
ABAC authorizer) to execute the interactions required to complete the TLS bootstrap flow.

#### Example kubelet command line UX

Using the discovery API, we could reduce the required node configuration to a single parameter, the
cluster-id:

```
$ kubelet \
  --cluster=v1alpha1/${CLUSTER_ID}
$ kubelet \
  --discovery-service=https://k8s-discovery.my-company.internal \
  --cluster=v1alpha1/98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4
```
