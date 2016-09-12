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

If you are using a released version of Kubernetes, you should refer to the docs that go with that version.

Documentation for other releases can be found at [releases.K8s.io](http://releases.K8s.io).
</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->


# Overview

We describe a proposal for retrieving component configurations (componentconfigs) that govern the behavior of any one of the Kubernetes (K8s) components, which we shall describe as services from now on. By services we mean any entity that is routed via DNS in a K8s cluster, these include internal and external services. 

We propose several important changes:
* adding new authenticated protected path to a service api: 
    * `/api/alpha/componentconfig` 
    * Discussions to be held with SIG-Auth individuals about making such a path secured.
* helping complete the migration of flags to componentconfigs

## Background

The document is built from SIG-Scale and PR discussions. Here we document the results of these and build the proposal of what changes should/will be made to K8s.

## Motivation

By necessity Kubernetes is a stateful thing made up of services who have been configured in a particular way. Currently the K8s code base, as 09/2016, does not support a way nor means by which to query for the set of configurations that led to the current state of any service. As a consequence it is extremely difficult to debug the current state of a K8s cluster, especially when original configurations have been lost, or are simply not available. 

## Design

Services will have either a path indicated by annotations that can point to how to retrieve the componentconfigs or through a well known service api path, as discussed below.

### Service API changes

Services (like kubelet) have endpoints that can be altered to include an endpoint such as

```
 /api/alpha/componentconfigs
```

where (via HTTP GET) the output would be:

```
{
    "kind": "componentconfigs",
    "apiVersion": "alpha",
    "metadata": {
        "component": "kubelet",
        "id": "818B908B-D053-CB11-BC8B-EEA826EBA090"
        "timeStamp": "2016-09-13T14:47:35+00:00",
    },
    "configs": {
        "cloud-provider": "aws",
            ...
    }
}
```

Those componentconfigs that are to be associated with a particular service will be annotated so as to be labeled for collection and served by the proposed endpoint. 

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtimeconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

