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
[here](http://releases.k8s.io/release-1.0/docs/design/api-plugin-model.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes API Plugin Model

## Goals

* Allow new APIs to be registered and serviced by third party components
  dynamically at run time, not requiring a recompilation of any core kubernetes
  component.
* All state is stored together by apiserver; third party components can remain
  stateless.
* Allow core components to be examined by third parties via a well defined API.

## New REST Objects

1. The `Provider` type. The provider specifies:
  1. A REST endpoint to dial out to.
  2. A list of types for which this provider wants web hooks. 'all' is a valid
     choice.
  3. Optionally, for each type, a list of hooks the provider wants called.
2. The `ThirdPartyType` type. This declares a new type to the system, and stores
   all the information needed to identify this type.
  1. APIGroup
  2. Kind
  3. The version of this type.
  4. The above must specify a unique object (maybe we should pack it into the
     name, as well as having it in separate fields for convenience).
  5. Swagger spec for this type (or a `Provider` from which to get this spec).
  6. A choice: validation could be performed by a hook (see below), or by
     apiserver based on the swagger spec, or by both.
  7. Note that it's *optional* for a provider to request hooks for its own
     types; the default behavior will give you ordinary CRUD behavior.
3. Types representing the input and expected outputs for all the hooks below.
  1. This will include at minimum information about the user making the request.

## Hooks

A "hook" is a place in `kube-apiserver`'s code path where it may call a specific
REST endpoint served by a `Provider`. Some hooks are "chainable", meaning that
multiple providers may request the same hook on the same type. For example,
an admission check and several field initializers may all want to hook into a
POST call.

Allowed hooks are:
* pre-POST
  * Called before an object is initially created. This hook will recieve the
    object the user POSTed, and may return a filtered/mutated version of the
    object back, or reject the creation entirely.
  * Chainable.
* pre-PUT
  * Called before an object is modified. This hook will recieve the original
    object, the object the user just PUTed, and may return a filtered/mutated
    version of the object back, or reject the put entirely.
  * Chainable.
* pre-GET
  * Called before an object is returned; may forbid the user from seeing the
    object.
  * Chainable.
* pre-LIST
  * Called before an object list is returned; may forbid the user from seeing
    the list.
  * Chainable.
* pre-WATCH
  * Called before an object watch is accepted; may forbid the user from seeing
    the list.
  * Chainable.
* pre-PROXY
  * Called to interpret the proxy request. Is passed the current object, and is
    expected to return a resolvable address, or forbid the proxy operation.

Hooks are called in serial, and block the request until all have returned.
Initially, it is expected that ordering of the hook calls will not affect the
result.

Possible future optimization, which relaxes that constraint:
* Allow `Providers` to specify "supplies" and "depends-on" lists for their
  hooks, which would be strings specifying the fields that they set or depend
  upon being set. Using this information, a DAG can be formed and some hooks can be
  called in parallel, then their results (if they filtered or initialized some
  fields) can be merged.

## Expected usage

The normal way to use this api is to make a "fooProvider" Pod,
ReplicationController, and a Service. When the pod(s) starts, it checks that the
Service is registered as a `Provider`, and if not, creates a fooProvider object.

If fooProvider provides new api types, it will also check/add relevant
`ThirdPartyType` objects.

Then, it will behave as a normal REST webserver, handling any hook requests.

If it needs to do background work on the objects it provides--as the current
system does for ReplicationControllers, Endpoints, etc.--then it should also
start up a thread to implement the control loop, listing and watching Kubernetes
resources as necessary.

## Admission, Authentication, Initializers etc.

These would be implemented via Providers that register hooks on all types.
Multiple orthogonal admission checks can be done by adding multiple Providers.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/api-plugin-model.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
