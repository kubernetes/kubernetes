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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/resource-qos.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Initializers in Kubernetes

**Author**: Derek Carr (@derekwaynecarr)

**Status**: Proposed

*This document introduces the concept of initializers in Kubernetes, and describes use cases and implementation details.*

## Motivation

The ability to introduce hooks into the API to modify an object prior to it being acted upon in the system is a common requirement.

The primary mechanism to achieve this use-case today is to write an "admission controller" which is a server-side filter that observes API server actions, and provides the ability for an ordered set of filters to observe, modify, accept, or reject an object from being persisted into the API server.  By providing the ability to give an immediate yes/no response to the user, it provides a good place for hooks for use-cases that require an immediate user response.  For example, the user is able to be told that they are not able to create an object in a namespace without first creating that namespace.  Another common example is quota enforcement where the user is immediately forbidden from exceeding any quota constraints.  In essence, its a synchronous visitor pattern.

While powerful, admission controllers have drawbacks.  The major drawback is that it requires a recompilation and restart of the API server to introduce new behavior.  For those scenarios where an immediate yes/no response was not required for the user experience, it would be ideal if the system could support an asynchronous visitor pattern that would allow controllers external to the API server to observe, and modify a resource prior to it becoming 'active' in the system.  This would allow a new type of controller to be authored in the system that can initialize a resource similar to how other controllers watch and act upon already initialized objects.  There are many scenarios where this would be appropriate.  For example, a pod auto-sizer could be deployed separate from the API server that watches for incoming pods.  The pod-autosizer should be able to assign compute resource requests prior to the pod being scheduled.  If authored as an "initializer", it would allow anyone to bring their own logic to the cluster for how sizing occurs without needing to recompile or change the core components.

## Goals

1. Enable asynchronous controller pattern (i.e. initializers) as a viable alternative to existing admission control to support loose coupling of components where synchronous user response is less critical.
2. Enable registration of a set of initializers to an API resource that must be able to act on an object to support its initialization.

## Model changes

**Initializer**

A new type introduced into the API that allows the registration of an initializer with an associated group and kind in the API.  This allows dynamic registration of initializers without a forced restart or recompilation of the server.

**NOTE: It's possible we could support this without introduction of a new type, maybe via the ConfigData resource, needs thought and input from community**

**ObjectMeta.Initializers**

Map of named tokens representative of external controllers that must visit a resource prior to becoming active. The set of initializers are defaulted based on the API Group and Kind of resource as part of configuration using our existing admission control facility.  For example, if an Initializer (name=initial-sizer) is associated with (group/kind=pod), then when a pod is initially created via the API, it will have its ObjectMeta.Initializers field set to "initial-sizer".

**ObjectMeta.Initialized**

Boolean flag denoting if resource is or is not initialized. If a resource has active initializers, the value is false. Once all initializers have acted, the value is toggled to true. Once a resource is initialized, it is forever initialized. When a resource is not initialized, validation rules for editing are loosened to allow components to make delayed updates. An object must pass strict validation in order to become "initialized".

## REST API Changes

All API objects will support a new endpoint as a sub-resource that allows non-initialized objects to be edited via a PUT that does not carry the same validation constraints as normal updates of a resource.

For example:

```PUT /api/{version}/{group}/{kind}/{name}/initialize/{initializer-name}```

The input to the request would be the normal serialization of the object.  The input object would be persisted, and the named *initializer* would be removed from the object's set of **ObjectMeta.Initializers**.  The only validation that must pass on the object is the validation of the *ObjectMeta* related data.  If the named object was already initialized, the API server would error.  If the initializer was not in the list of **ObjectMeta.Initializers**, the request would error.

In addition, each object would support an API to mark the object as completely initialized.

For example:

```POST /api/initialize```

The input to the request would be a *Initialize* object, that would work similar to *Scale* with HPA that would take a subresource reference to know the object that is to be transitioned from non-initialized to initialized.  If the object was already initialized,
the API would error.  If the object failed strict validation, the system would error.

**NOTE: I will go back and fill in more details here**

## Making the transition to initialized

A new controller is introduced into the system that looks for objects that have no more named "initializers" but are not yet "initialized".

It introspects the set of registered *Initializers*, and uses the *Discovery API* and a generic API client to know what resource types to WATCH that are not yet initialized.  It then creates a set of watchers that look for objects whose list of "initializers" is empty, but the object is not yet "initialized".

It then performs an "/api/initialize" action for each resource that meets the required criteria.  If the API action succeeded, the object is now initialized for use by the system.  If the API action fails, it's considered to be in an error state that is unrecoverable.  When this occurs, the controller will permanently delete the resource from the API server and log an event because the object would forever fail strict validation.

## Existing controller manager components

Objects like a ReplicationController should treat a non-initialized pod as a pod, meaning it should not create new ones to satisfy its desired replica set.

Objects like the Scheduler should treat a non-initialized pod as a pod not yet ready to be scheduled.

Objects like ResourceQuota should enforce during all phases of the initialization process.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/resource-qos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
