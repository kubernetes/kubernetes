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

**WORK-IN-PROCESS -- UNFINISHED -- WORK-IN-PROGRESS**

## Abstract

This proposal aims at extending the auditing log capabilities of the apiserver.

## Motivation and Goals

With https://github.com/kubernetes/kubernetes/pull/27087 basic audit logging is added to Kubernetes. It basically implements `access.log` like http handler based logging of all requests in the apiserver API. It does not do deeper inspection of the API calls or of their payloads. Moreover, it has no specific knowledge of the API objects which are modified. Hence, the log output does not answer the question how API objects actually change.

The log output format of https://github.com/kubernetes/kubernetes/pull/27087 is fixed. It is text based, unstructured (e.g. non-JSON) data which must be parsed to be usable in any advanced external system used to analyze audit logs.

The log output format does not follow any public standard like e.g. https://www.dmtf.org/standards/cadf.

With this proposal we describe how the auditing functionality can be extended in order:

- to allow multiple output formats, e.g. access.log style or structured JSON output
- to allow deep payload inspection to allow
  - either real differential JSON output (which field of an object have changed from which value to which value)
  - or full object output of the new state (and optionally the old state)
- to be extensible in the future to fully comply with the Cloud Auditing Data Federation standard (https://www.dmtf.org/standards/cadf)
- to allow filtering of the output
  - by kind, e.g. don't log endpoint objects
  - by object path (JSON path), e.g. to ignore all `*.status` changes
  - by user, e.g. to only log end user action, not those of the controller-manager and scheduler
  - by level (request headers, request object, storage object)

while

- not degrading apiserver performance when auditing is disabled.

## Constraints and Assumptions

* it is not the goal to implement all output formats one can imagine. The  main goal is to be extensible with a clear golang interface. Implementations of e.g. CADF must be possible, but won't be discussed here.
* dynamic loading of backends for new output formats are out of scope.

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

1. As a cluster operator I must be able to provide a **complete trace of changes to an object** to API objects.

1. As a cluster operator I must be able to create a trace for **all accesses to a secret**.

1. As a cluster operator I want to **define log-rotation** of a file-based output backend.

1. As a cluster operator I must be able to log non-CRUD access like **kubectl exec**, when it started, when it finished and with which initial parameters.

### Out of scope use-cases

1. As a cluster operator I must be able to get a trace of interactive commands executed in **kubectl exec**, **attach** or **run**.

## Community Work

- Kubernetes basic audit log PR: https://github.com/kubernetes/kubernetes/pull/27087/
- OpenStack's implementation of the CADF standard: https://www.dmtf.org/sites/default/files/standards/documents/DSP2038_1.1.0.pdf
- Cloud Auditing Data Federation standard: https://www.dmtf.org/standards/cadf
- Ceilometer audit blueprint: https://wiki.openstack.org/wiki/Ceilometer/blueprints/support-standard-audit-formats
- Talk from IBM: An Introduction to DMTF Cloud Auditing using
the CADF Event Model and Taxonomies https://wiki.openstack.org/w/images/e/e1/Introduction_to_Cloud_Auditing_using_CADF_Event_Model_and_Taxonomy_2013-10-22.pdf

## Architecture

When implementing audit logging there are basically two options:

1. put a logging proxy in front of the apiserver
2. integrate audit logging into the apiserver itself

Both approaches have advantages and disadvanteges:

- **pro proxy**:
  + keeps complexity out of the apiserver
  + reuses existing solutions
- **contra proxy**:
  + has no deeper insight into the Kubernetes api
  + has no knowledge of auth, authn, admission
  + has no access to the storage level for diffential output
  + has to terminate SSL and complicates client certificates based auth

In the following the second approach is described without a proxy.

## Proposed Design

The main concepts are those of

- an audit *event*,
- an audit *policy*,
- an audit *policy action*,
- an audit *output backend*.

An audit event holds all the data necessary for an *output backend* to produce an audit log entry. The *event* is independent of the *output backend*.

The audit event struct is passed through the apiserver layers as an `*audit.Event` pointer inside the http context. It might be `nil` in case auditing is completely disabled.

If auditing is enabled and the policy has an audit policy action (see below), the http handler will attach an `audit.Event` to the context:

```go
package api

func WithAuditEvent(parent Context, e *audit.Event) Context
func AuditEventFrom(ctx Context) (*audit.Event, bool)
```

Depending on the audit policy, different layers of the apiserver (e.g. http handler, storage) will fill the `audit.Event` struct. Certain fields might stay empty or `nil` if the policy does not require that field. E.g. in the case only http headers are supposed to be audit logged, no `OldObject` or `NewObject` is to be retrieved on the storage layer.

The audit policy is a partial mapping from

- kind,
- namespace,
- method,
- and user

to a policy action. An policy action defines the level of audit logging to be performed. The audit level can be

- `HttpHeaders`,
- `RequestObject`,
- `StorageObject`.

In addition the policy action can contain a number of output backend dependent key/values e.g. to define JSON object paths which should be logged or excluded from logging. For portable policy actions, a number of key/values are standardised and ought to be supported by output backends.

When the http request is processed, the request handler will close the audit event by filling in the http response. Then it will pass the event to the configured policy backend.

**Note:** for service creation and deletion there is special REST code in the apiserver which takes care of service/node port (de)allocation and removal of endpoints on service deletion. Hence, these operations are not visible on the API layer and cannot be audit logged therefore. **No other resources** (with the exception of componentstatus which is not of interest here) **implement this kind of custom CRUD operations.**

### Events

```go
package audit

// LogLevel defines the amount of information logged for auditing
type LogLevel int


type Event struct {
  Level LogLevel

  // http header level (HeaderLogLevel and higher)
  ID string
  Timestamp time.Timestamp
  IP string
  Method string
  URI string
  User string
  AsUser string
  Namespace, Name string
  GroupKind unversioned.GroupKind
  Response int

  // CRUD level (RequestLogLevel and higher)
  RequestObject runtime.Unstructured // before admission

  // Storage level (StorageLogLevel and higher)
  OldObject runtime.Object
  NewObject runtime.Object
  Patch     []byte
}
```

### Concurrency/Transaction Model for Events

TODO: define (and maybe adapt interface) how synchronization of the writable audit event of a context takes place.

### Policy

```go
package audit

const (
  // DontLogLevel means to not log at all
  DontLogLevel LogLevel = iota

  // HeaderLogLevel means to log only information from the http headers, not from the body
  HeaderLogLevel

  // RequestLogLevel adds the user request in the body
  RequestLogLevel

  // StorageLogLevel adds the old and the new object that is received from and sent to the storage layer
  StorageLogLevel
)

type Action interface {
  Level() LogLevel
  Value(key string) string
}

type Policy interface {
  func Action(kind, namespace, method, user string) Action
}
```

TODO: how do the storage layer and the http handlers get access to the policy?

### Policy Actions

TODO: define what we need next to the audit level

### Output Backend Interface

```go
package audit

type OutputBackend interface {
  Log(e *Event, a Action) error
}
```

### Apiserver Command Line Flags

```shell
$ kube-apiserver --audit-output file:path=/var/log/apiserver-audit.log,rotate=1d,max=1024MB,format=json \
  --audit-policy user=EndUser,kind=ReplicationController:output=diff,exclude=status
```

## Sensible (not necessarily sequential) Milestones of Implementation

1. add `audit.Event` and `audit.OutputBackend` and implement https://github.com/kubernetes/kubernetes/pull/27087/'s basic auth using them
1. add deep inspection on the storage level to the old and the new object
1. add output level policy support (versus static policy in the initial implementation)

## Examples

### Audit Policy

- never log `*.Status`
- never log `Secret.Data`
- never log `Endpoint`s
- never log if user is "kubelet" or "master"
- only log headers for WATCH, GET
- only log the request object on POST and PUT
- never log OPTIONS

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/sysctl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
