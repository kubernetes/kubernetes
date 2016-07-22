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

## Abstract

This proposal aims at extending the auditing log capabilities of the apiserver.

## Motivation and Goals

With https://github.com/kubernetes/kubernetes/pull/27087 basic audit logging is added to Kubernetes. It basically implements `access.log` like http handler based logging of all requests in the apiserver API. It does not do deeper inspection of the API calls or of their payloads. Moreover, it has no specific knowledge of the API objects which are modified. Hence, the log output does not answer the question how API objects actually change.

The log output format of https://github.com/kubernetes/kubernetes/pull/27087 is fixed. It is text based, unstructured (e.g. non-JSON) data which must be parsed to be usable in any advanced external system used to analyse audit logs.

The log output format does not follow any public standard like e.g. https://www.dmtf.org/standards/cadf.

With this proposal we describe how the auditing functionality can be extended in order:

- to allow multiple output formats, e.g. access.log style or structured JSON output
- to allow deep payload inspection to allow
  - either real differential JSON output (which field of an object have changed from which value to which value)
  - or full object output of the new state (and optionally the old state)
- to be extensibile in the future to fully comply with the Cloud Auditing Data Federation standard (https://www.dmtf.org/standards/cadf)
- to allow filtering of the output
  - by kind, e.g. don't log endpoint objects
  - by object path (JSON path), e.g. to ignore all `*.status` changes
  - by user, e.g. to only log enduser action, not those of the controller-manager and schedule
  - by level (request headers, request object, storage object)

while

- not degrading apiserver performance when auditing is disabled.

## Constraints and Assumptions

* it is not the goal to implement all output formats one can imagine. The  main goal is to be extensible with a clear golang interface. Implementations of e.g. CADF must be possible, but won't be discussed there.
* dynamic loading of plugins for new output formats are out of scope.

## Use Cases

1. As a cluster operator I want to enable audit logging of requests to the apiserver in order **to comply with given business regulations** regarding a subset of the 7 Ws of auditing:

  - **what** happened?
  - **when** did it happen?
  - **who** initiated it?
  - **on what** did it happen (e.g. pod foo/bar)?
  - **where** was it observed (e.g. apiserver hostname)?
  - from **where** was it initiated? (e.g. kubectl IP)
  - to **where** was it going? (e.g. node 1.2.3.4 for kubectl proxy).

1. Depending on the environment, as a cluster operator I want to **define the amount of audit logging**, balancing computational overhead for the apiserver with the detail and completeness of the log.

1. As a cluster operator I want to **integrate with external systems**, which will have different requirements for the log format, network protocols and communication modes (e.g. pull vs. push).

1. As a cluster operator I must be able to provide a **complete trace of changes** to API objects.

## Community Work

- Kubernetes basic audit log PR: https://github.com/kubernetes/kubernetes/pull/27087/ 
- OpenStack's implementation of the CADF standard: https://www.dmtf.org/sites/default/files/standards/documents/DSP2038_1.1.0.pdf
- Cloud Auditing Data Federation standard: https://www.dmtf.org/standards/cadf 
- Ceilometer audit blueprint: https://wiki.openstack.org/wiki/Ceilometer/blueprints/support-standard-audit-formats 

## Proposed Design

TODO

### Events

```go
package audit
type Event interface {
  // opening
  ID string
  Timestamp time.Timestamp
  IP string
  Method string
  URI string
  User string
  AsUser string
  ObjectMeta runtime.ObjectMeta
  GroupKind unversioned.GroupKind

  // request object
  RequestObject runtime.Unstructured

  // storage object
  OldObject runtime.Object
  NewObject runtime.Object

  // closing
  Response string
}
```

### Output Plugin Interface

```go
package audit

type OutputBackend interface {
  Log(e *Event) error
  ...
}
```

### Apiserver Command Line Flags

```bash
$ kube-apiserver --audit-output file:path=/var/log/apiserver-audit.log,rotate=1d,max=1024MB,format=json \
  --audit-policy user=EndUser,kind=ReplicationController:output=diff,exclude=status
```

## Examples

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/sysctl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->