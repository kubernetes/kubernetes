<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
###ObjectMeta###

---
* annotations: 
  * **_type_**: any
  * **_description_**: map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about objects; see http://releases.k8s.io/HEAD/docs/annotations.md
* creationTimestamp: 
  * **_type_**: string
  * **_description_**: RFC 3339 date and time at which the object was created; populated by the system, read-only; null for lists; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* deletionTimestamp: 
  * **_type_**: string
  * **_description_**: RFC 3339 date and time at which the object will be deleted; populated by the system when a graceful deletion is requested, read-only; if not set, graceful deletion of the object has not been requested; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* generateName: 
  * **_type_**: string
  * **_description_**: an optional prefix to use to generate a unique name; has the same validation rules as name; optional, and is applied only name if is not specified; see http://releases.k8s.io/HEAD/docs/api-conventions.md#idempotency
* generation: 
  * **_type_**: integer
  * **_description_**: a sequence number representing a specific generation of the desired state; populated by the system; read-only
* labels: 
  * **_type_**: any
  * **_description_**: map of string keys and values that can be used to organize and categorize objects; may match selectors of replication controllers and services; see http://releases.k8s.io/HEAD/docs/labels.md
* name: 
  * **_type_**: string
  * **_description_**: string that identifies an object. Must be unique within a namespace; cannot be updated; see http://releases.k8s.io/HEAD/docs/identifiers.md#names
* namespace: 
  * **_type_**: string
  * **_description_**: namespace of the object; must be a DNS_LABEL; cannot be updated; see http://releases.k8s.io/HEAD/docs/namespaces.md
* resourceVersion: 
  * **_type_**: string
  * **_description_**: string that identifies the internal version of this object that can be used by clients to determine when objects have changed; populated by the system, read-only; value must be treated as opaque by clients and passed unmodified back to the server: http://releases.k8s.io/HEAD/docs/api-conventions.md#concurrency-control-and-consistency
* selfLink: 
  * **_type_**: string
  * **_description_**: URL for the object; populated by the system, read-only
* uid: 
  * **_type_**: string
  * **_description_**: unique UUID across space and time; populated by the system; read-only; see http://releases.k8s.io/HEAD/docs/identifiers.md#uids


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ObjectMeta.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
