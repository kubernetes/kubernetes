# kube2sky
==============

A bridge between Kubernetes and SkyDNS.  This will watch the kubernetes API for
changes in Services and Pods and then publish those changes to SkyDNS through etcd.

For now, this is expected to be run in a pod alongside the etcd and SkyDNS
containers.

## Namespaces

Kubernetes namespaces become another level of the DNS hierarchy.  See the
description of `-domain` and `-pod_domain` below.

## Flags

`-domain`: Set the domain under which all service DNS names will be hosted.  For
example, if this is set to `svc.kubernetes.io`, then a service named "nifty" in the
"default" namespace would be exposed through DNS as
"nifty.default.svc.kubernetes.io".

`-pod_domain`: Set the domain under which all DNS names for pods matching services
will be hosted. If this is set to `pod.kubernetes.io`, then a pod named "dandy"
providing a service named "nifty" in the "default" namespace would be exposed
through DNS as
"dandy.nifty.default.pod.kubernetes.io".

## Named ports

If service ports are named, then port name becomes part of the domain name:
* mainport.nifty.default.svc.kubernetes.io
* dandy.mainport.nifty.default.pod.kubernetes.io


`-verbose`: Log additional information.
