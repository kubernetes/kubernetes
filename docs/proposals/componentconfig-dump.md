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

We describe a proposal for retrieving component configurations (componentconfigs) that govern the behavior of any one of the Kubernetes (K8s) components. 

We propose several important changes:
* adding new authenticated protected path to a component api: 
    * `/componentconfig` 
    * Discussions to be held with SIG-Auth individuals about making such a path secured, but we will invoke that similar discussions around metrics have taken place.
* helping complete the migration of flags to componentconfigs

## Background

The document is built from SIG-Scale and PR discussions. Here we document the results of these and build the proposal of what changes should/will be made to K8s.

## Motivation

By necessity Kubernetes is a stateful thing made up of components who have been configured in a particular way. Currently the K8s code base, as 09/2016, does not support a way nor means by which to query for the set of configurations that led to the current state of any component. As a consequence it is extremely difficult to debug the current state of a K8s cluster, especially when original configurations have been lost, or are simply not available. 

## Design

Components will have either a path indicated by annotations that can point to how to retrieve the componentconfigs or through a well known component api path, as discussed below.

### Component API changes

Components (like kubelet) have endpoints that can be altered to include an endpoint such as

```
 /componentconfigs 
```

similar to /metrics endpoints for K8s components and likely to be authenticated in much the same way.

By way of an HTTP GET the output would be:

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

Those componentconfigs that are to be associated with a particular component will be annotated so as to be labeled for collection and served by the proposed endpoint. 

## Alternatives and Further Discussions

Here we mention points of contentions and alternatives as we make progress in the proposal.

* Security/Auth has been brought up as a point that strongly affects the proposed endpoint. Ideas around component endpoints and auth have been discussed for `/metrics` and it has been noted that SIG-Auth may be assigned the `/componentconfigs` endpoint to ensure auth is handled correctly.
* The addition of an api change/addition to APIServer has been discussed, and for reasons of security it was agreed to be left out as a potential addition in the future.
* The addition of kubectl/kubeadmin interactions has also been scrapped for reasons of security with possible addition in the future, but do not gain us much for our present motivations.
* It has been agreed that extracting configs from componentconfigs is the way forward.

## Future Work

We discuss here future work that should be considered an extension of this proposal, and may become actionable work to be included for present development. The idea here is to extend our endpoint with the additional `PUT HTTP method`, as a way to configure components, the motivation is as follows:

>However, we can't yet assume that everyone is selfhosting and grabbing the componentconfig from configmaps. At the community meeting today we saw an example (RackN/Digital Rebar) of tooling that chooses not to use self-hosting. (JBeda). 

The consequence of this would be that our endpoint may be limited and may not be a source of truth that we propose it to be. To resolve this situation, we could have have each component present in their annotations the endpoint (if it is not the default) that will accept changes to the componentconfigs, and thus re-establish the source of truth. 


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtimeconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

