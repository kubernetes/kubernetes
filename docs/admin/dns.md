<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# DNS Integration with Kubernetes

As of Kubernetes 0.8, DNS is offered as a [cluster add-on](http://releases.k8s.io/release-1.1/cluster/addons/README.md).
If enabled, a DNS Pod and Service will be scheduled on the cluster, and the kubelets will be
configured to tell individual containers to use the DNS Service's IP to resolve DNS names.

Every Service defined in the cluster (including the DNS server itself) will be
assigned a DNS name.  By default, a client Pod's DNS search list will
include the Pod's own namespace and the cluster's default domain.  This is best
illustrated by example:

Assume a Service named `foo` in the Kubernetes namespace `bar`.  A Pod running
in namespace `bar` can look up this service by simply doing a DNS query for
`foo`.  A Pod running in namespace `quux` can look up this service by doing a
DNS query for `foo.bar`.

The cluster DNS server ([SkyDNS](https://github.com/skynetservices/skydns))
supports forward lookups (A records) and service lookups (SRV records).

## How it Works

The running DNS pod holds 3 containers - skydns, etcd (a private instance which skydns uses),
and a Kubernetes-to-skydns bridge called kube2sky.  The kube2sky process
watches the Kubernetes master for changes in Services, and then writes the
information to etcd, which skydns reads.  This etcd instance is not linked to
any other etcd clusters that might exist, including the Kubernetes master.

## Issues

The skydns service is reachable directly from Kubernetes nodes (outside
of any container) and DNS resolution works if the skydns service is targeted
explicitly. However, nodes are not configured to use the cluster DNS service or
to search the cluster's DNS domain by default.  This may be resolved at a later
time.

## For more information

See [the docs for the DNS cluster addon](http://releases.k8s.io/release-1.1/cluster/addons/dns/README.md).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/dns.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
