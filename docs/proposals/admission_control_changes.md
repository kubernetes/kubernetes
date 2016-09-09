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

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Admission control changes

Admission control is the key security and cluster policy enforcement mechanism
in Kubernetes. While in the future, we may wish to leverage additional
mechanisms beyond the admission control point, given its depth of use by Kube
users today, we should take steps to organize and streamline the existing
functionality to better enable extension.

We are proposing a series of changes to admission control in an attempt to make
it less error-prone, easier to configure, and more flexible for admission plugin
developers.


## Hard-coded plugin execution order

It's unlikely that administrators need to change the order in which admission
plugins are executed. Furthermore, running certain plugins "out of order" will
result in undesirable or incorrect behavior. For example, if you run
`ResourceQuota` before `LimitRanger`, you could end up incorrectly calculating
quota because it's possible for `LimitRanger` to modify pod values after
`ResourceQuota` performed its quota checks.

Because of this, we want to move to a model where in-tree admission plugin
execution is strongly ordered and not under the administrator's control. We
would therefore deprecate the `--admission-control` setting from
`pkg/genericapiserver/options.ServerRunOptions`. During the deprecation period,
if this flag is set, we will display a warning while continuing to honor the values
specified by the flag (i.e., the administrator can override the hard-coded ordering).


## One config file per plugin

All admission plugins currently share a single config file, specified via
`--admission-control-config-file`. This is problematic if multiple plugins
use the same names for their configuration sections. For
example, if plugin `Foo` wants

```yaml
myConfig:
  someField: red
```

and plugin `Bar` wants

```yaml
myConfig:
  someField:
  - one
  - two
  - three
```

this won't work because of the naming collision and the type of `someField` is
different for the two plugins. What we'd like to see instead is:

1. deprecate `--admission-control-config-file`
1. replace it with `--admission-plugin PLUGIN=CONFIG_FILE` (or an equivalent
   setup in a structured configuration, once we've fully moved over)

You would specify `--admission-plugin` once per plugin that you want to
configure. An example of this would be

```
--admission-plugin AlwaysPullImages=/path/to/config.yaml --admission-plugin MyPlugin=/some/other/config.yaml
```

An example of a structured configuration for admission control inside an
apiserver configuration file might look like this:

```yaml
admissionConfig:
  AlwaysPullImages:
    configurationFile: /path/to/config.yaml
  MyPlugin:
    configurationFile: /some/other/config.yaml
  
```

## Ability to enable/disable individual plugins

With a fixed list of plugins enabled by default, administrators will need to be
able to override the defaults and disable certain plugins that are normally on,
and enable plugins that are normally off. We are proposing that the admission
control section in the apiserver configuration file include support for this. We
can use `enabled` as seen below:

```yaml
admissionConfig:
  AlwaysPullImages:
    # disabled by default, so enable it
    enabled: true
    configurationFile: /path/to/config.yaml
  MyPlugin:
    # enabled by default, so disable it
    enabled: false
    configurationFile: /some/other/config.yaml
```

Note, the enabling/disabling mechanism described in this proposal only works
when using a structured configuration setup for the apiserver. The
`--admission-plugin` flag described above does not provide a means to enable or
disable a plugin (TODO: should it????).

## Code changes

### Admission interfaces

The current interface in `pkg/admission/interfaces.go` looks like this:

```go
type Interface interface {
  Admit(a Attributes) (err error)
  Handles(operation Operation) bool
}
```

We'd like to break this into multiple interfaces, one per phase, called at the appropriate times
within the request lifecycle.

```go
type Defaulter interface {
  // ApplyDefaults is executed after the request body is deserialized into an
  // object for create requests (TODO do we want this for update/patch too?).
  // Plugins are free to modify the object as they see fit. If an error is
  // returned, the request is marked as failed and no changes are persisted.
  ApplyDefaults(a Attributes) error
}

type Validator interface {
  // Validate is executed after the appropriate `rest.Storage` handler's
  // validation occurs. This gives the plugin the opportunity to validate the
  // object after the handler has validated it, but before it is persisted. If
  // an error is returned, the request is marked as failed and no changes are
  // persisted.
  Validate(a Attributes) error
}


type ConstraintEnforcer interface {
  // EnforceConstraints is executed after all Validator plugins have executed.
  // This gives plugins a chance to decide if the request should be allowed to
  // proceed after the object has passed `rest.Storage` handler validation.
  // ConstraintEnforcers should not have any side effects. If an error is
  // returned, the request is marked as failed and no changes are
  // persisted.
  EnforceConstraints(a Attributes) error
}

type Finisher interface {
  // Finish (name TBD) is executed at the very end of the request lifecycle,
  // after calls to the `rest.Storage` persistence methods. If an error is
  // returned, the request is marked as failed, but any changes are preserved
  // and nothing is rolled back.
  Finish(a Attributes) error
}

type Interface interface {
  // Handles returns true if this admission controller can handle the given
  // operation where operation can be one of CREATE, UPDATE, DELETE, or CONNECT.
  Handles(operation Operation) bool
}
```

### rest Storage

To be able to invoke admission plugin validation at the right time, we'll need
to modify the various `rest.Storage` interfaces, as described below.

#### Creater, NamedCreater

- add `BeforeCreate(ctx api.Context, obj runtime.Object) error`
- add `Validate(ctx api.Context, obj runtime.Object) error`

#### Updater, CreaterUpdater

- add `BeforeUpdate(ctx api.Context, obj, old runtime.Object) error`
- add `ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList`

#### Patcher

- inherits changes from `Updater`


#### Deleter, GracefulDeleter

- add `BeforeDelete(ct api.Context, obj runtime.Object, options *api.DeleteOptions) (graceful, gracefulPending bool, err error)`

#### Connector

- QUESTION: do we need to adjust this interface?


## REST flows

### Create

#### Current behavior

This is the current [apiserver create
flow](https://github.com/kubernetes/kubernetes/blob/ef0c9f0c5b8efbba948a0be2c98d9d2e32e0b68c/pkg/apiserver/resthandler.go#L333):

1. decode incoming request
1. admit.Admit
1. r.Create (the items below are part of the generic registry store / strategy-based storage, `pkg/registry/generic/registry/store.go`)
  1. rest.BeforeCreate
    1. strategy.PrepareForCreate
    1. api.FillObjectMetaSystemFields
    1. api.GenerateName
    1. strategy.Validate
    1. validation.ValidateObjectMeta
    1. strategy.Canonicalize
  1. e.Storage.Create
  1. e.AfterCreate
  1. e.Decorator

#### Proposed behavior

This is the proposed new behavior:

1. decode request
1. admit.ApplyDefaults
1. r.BeforeCreate
  1. If strategy-based, call rest.BeforeCreate
    1. strategy.PrepareForCreate
    1. api.FillInObjectMetaSystemFields
    1. api.GenerateName
  1. Otherwise, just do whatever is in the rest.Creater
1. r.Validate
  1. If strategy-based, call rest.Validate
    1. strategy.Validate
    1. validation.ValidateObjectMeta
    1. strategy.Canonicalize
  1. Otherwise, just do whatever is in the rest.Creater
1. admit.Validate
1. admit.EnforceConstraints
  1. e.g. for quota check
1. r.Create
  1. For strategy-based handlers:
    1. e.Storage.Create
    1. e.AfterCreate
    1. e.Decorator
  1. Otherwise, just do whatever is in the rest.Creater
1. admit.Finish (name TBD)

Notes:

- admit.ApplyDefaults replaces admit.Admit and gives plugins a chance to apply
  defaults to the decoded request object
- r.BeforeCreate is the `rest.Creater` interface's new method for performing
  logic after admission defaulting and before validation
- r.Validate is the `rest.Creater` interface's new method for performing
  validation
- admit.Validate is a new admission phase that is executed after the
  `rest.Creater`'s validation (plugins should not modify storage)
- admit.EnforceConstraints is another new admission phase that is executed after
  admit.Validate (plugins can modify storage)
- admit.Finish (name TBD) is the final admission phase, running after the
  `rest.Creater` has modified persistent storage


### Update

#### Current behavior

1. decode request
1. admit.Admit
1. r.Update (the items below are part of the generic registry store / strategy-based storage, `pkg/registry/generic/registry/store.go`)
  1. rest.BeforeUpdate
    1. strategy.PrepareForUpdate
    1. validation.ValidateObjectMetaUpdate
    1. strategy.ValidateUpdate
    1. strategy.Canonicalize
  1. e.Storage.GuaranteedUpdate
  1. e.AfterUpdate
  1. e.Decorator

#### Proposed behavior

1. decode request
1. ??? admit.ApplyDefaults ???
1. r.BeforeUpdate
  1. If strategy-based, call rest.BeforeUpdate
    1. strategy.PrepareForUpdate
  1. Otherwise, just do whatever is in the rest.Updater
1. r.ValidateUpdate
  1. If strategy-based, call rest.ValidateUpdate
    1. validation.ValidateObjectMetaUpdate
    1. strategy.ValidateUpdate
    1. strategy.Canonicalize
  1. Otherwise, just do whatever is in the rest.Updater
1. admit.Validate
1. admit.EnforceConstraints
1. r.Update
  1. For strategy-based handlers:
    1. e.Storage.GuaranteedUpdate
    1. e.AfterUpdate
    1. e.Decorator
  1. Otherwise, just do whatever is in the rest.Updater
1. admit.Finish (name TBD)

Notes:

- Should we have admit.ApplyDefaults?
- The `rest.Updater` interface's Update method has been split into 3:
  - BeforeUpdate
  - ValidateUpdate
  - Update


### Patch

Patch is similar to Update, and similar changes are required to split the call
to `Update` into `BeforeUpdate`, `ValidateUpdate`, and `Update`. Additionally,
calls to `admit.Validate`, `admit.EnforceConstraints`, and `admit.Finish` will
occur in appropriate places within the request lifecycle.

### Delete

#### Current behavior

1. decode request (DeleteOptions)
2. admit.Admit
3. r.Delete (the items below are part of the generic registry store / strategy-based storage, `pkg/registry/generic/registry/store.go`)
  4. e.Storage.Get
  1. rest.BeforeDelete
  1. e.Storage.Delete
  1. e.AfterDelete
  1. e.Decorator

#### Proposed behavior

1. decode request (DeleteOptions)
2. r.Get
1. r.BeforeDelete
  2. If strategy-based, call rest.BeforeDelete
  1. Otherwise, just do whatever is in the rest.Deleter
1. admit.Validate
1. admit.EnforceConstraints
1. r.Delete
  2. For strategy-based handlers:
    1. e.Storage.Delete
    1. e.AfterDelete
    1. e.Decorator
2. admit.Finish

### DeleteCollection

#### Current behavior

1. admit.Admit
2. decode ListOptions from request parameters
3. decode request body (DeleteOptions)
4. r.DeleteCollection (the items below are part of the generic registry store / strategy-based storage, `pkg/registry/generic/registry/store.go`)
  5. e.List
  1. run *n* worker goroutines in parallel:
    1. e.Delete (same as Delete -> Current behavior -> 3)

#### Proposed behavior

1. admit.Validate
2. admit.EnforceConstraints
2. decode ListOptions from request parameters
3. decode request body (DeleteOptions)
4. r.DeleteCollection ...
5. admit.Finish

### Connect

#### Current behavior

1. connecter.NewConnectOptions
2. decode request connect options
3. admit.Admit
4. connecter.Connect().ServeHTTP()

#### Proposed behavior

1. connecter.NewConnectOptions
2. decode request connect options
1. admit.Validate
2. connecter.Connect().ServeHTTP()

## Ability to run out-of-tree plugins

Admission plugins are all currently compiled into the apiserver binary. If you
want to run custom plugins, you have to modify the Kubernetes source code and
recompile the apiserver. Some examples of custom plugins that an administrator
might want to run (these are all actual plugins in OpenShift today):

- default restart-never and restart-on-failure pods' `activeDeadlineSeconds`
  based on either a namespace-level annotation, or the plugin's configured value
  (although this probably should be part of LimitRange).
- apply quota at the cluster level instead of per-namespace
- assign default resource request values relative to user-specified limits for
  pods to support an administrator-defined overcommit target
- restrict who is allowed to set the `nodeName` or `nodeSelector` on pods
- ensure you can't exec or attach to a pod with a privileged security context if
  you don't have privileges to create a privileged pod
- supply default values for certain fields on Build Pods when unset (e.g.
  environment variables)
- enforce cluster-wide configuration values on all Build Pods (i.e.
  override user-supplied data if necessary)
- automatically provision a Jenkins server in a user's namespace if it doesn't
  exist when a pipeline BuildConfig is created

We would like to have the ability to run a single external plugin at the end of
each admission phase. This will enable cluster administrators to inject custom
admission logic into an existing kube-apiserver without having to recompile it.

### OPEN QUESTIONS

#### What happens if the external admission plugin is not available?

Let's imagine that I have a Kubernetes cluster up and running. Now I want to run
OpenShift on top of it, so I deploy its various pods to the cluster, presumably
as Deployment resources. Next, I need to add OpenShift as an external admission
plugin for the kube-apiserver pods and restart them.

Fast forward a bit - now I have everything working. Then let's imagine that my
openshift-apiserver pods fail for whatever reason. What happens to
kube-apiserver requests with respect to admission? With openshift-apiserver
down, any attempt by the kube-apiserver to invoke the external admission plugin
will fail.

Do we ignore an external admission plugin being unreachable and allow requests
to proceed ("fail open")? I don't think this is viable. Presumably the external
admission plugin is there for a reason, and if it is unable to execute, it opens
the possibility for inconsistent or incorrect data to enter the cluster.

Do we instead reject the request because the external admission plugin is
unreachable ("fail closed")? I think this is what we must do, for the reason
stated above. Assuming this is what we choose, how do we restore the cluster to
a healthy state, as the default "fail closed" behavior won't allow any new pods,
and we need at least one new openshift-apiserver pod to handle the external
admission plugin invocations. We need some way to a) indicate that a certain set
of changes to the system are being made to restore the external admission
plugin, and b) ensure that users can't "sneak in" anything while admission isn't
fully operational. Some possible options include:

- Specify via flag/configuration the identity of a user or group whose requests
  will be allowed to proceed even when external admission is down
  - If the openshift-apiserver pods come from a Deployment, then the
    DeploymentController needs permission to create openshift-apiserver pods,
    but not pods for normal users
- Require that kube-apiserver and openshift-apiserver be deployed as containers
  in the same pod. This won't completely eliminate the possibility of all
  openshift-apiservers being down at the same time, but it will significantly
  minimize the likelihood of that happening. We probably still need a mitigation
  plan in place for the event where you're down to 1 running apiserver pod,
  kube-apiserver is running, openshift-apiserver has failed, and the system is
  trying simultaneously to restart the openshift-apiserver container and scale
  the replicas to 3+.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/admission_control_changes.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
