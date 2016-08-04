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

# Overview

Proposes adding a `--runtime-config` to core kube system components:
apiserver (already exists), scheduler, controller-manager, kube-proxy,
and selected addons. This flag will be used to enable/disable alpha
features on a per-component basis.

## Motivation

Motivation is enabling/disabling features that are not tied to
an API group. API groups can be selectively enabled/disabled in the
apiserver via existing `--runtime-config` flag on apiserver, but there is
currently no mechanism to toggle alpha features that are controlled by
e.g. annotations. This means the burden of controlling whether such
features are enabled in a particular cluster is on feature implementors;
they must either define some ad hoc mechanism for toggling (e.g. flag
on component binary) or else toggle the feature on/off at compile time.

This proposal suggests using the existing `--runtime-config` of
kube-apiserver and adding a similar flag to each of the core
kube system components. Alpha features can then be toggled on a
per-component basis by passing `enableAlphaFeature=true|false` to
`--runtime-config`.

## Design

The following components will all get a `--runtime-config` flag,
which loads a `config.ConfigurationMap`:

- kube-apiserver (already present)
- kube-scheduler
- kube-controller-manager
- kube-proxy
- kube-dns

(Note kubelet is omitted, it's dynamic config story is being addressed
by #29459). Alpha features that are not accessed via an alpha API
group should define an `enableFeatureName` flag and use it to toggle
activation of the feature in each system component that the feature
uses.

## Upgrade support

As the primary motivation for cluster config is toggling alpha
features, upgrade support is not in scope. Enabling or disabling
a feature is necessarily a breaking change, so config should
not be altered in a running cluster.

## Rejected designs

### Shared global config file

Cluster-wide runtime config will be specified in a single yaml file
loaded from a known, default path. TBD whether default config path
should be overwriteable via flag/env.

Strawman:
```shell
$ cat /etc/srv/kubernetes/cluster_config.yaml
runtimeConfig:
  enableAppArmor: true
  enableFooBarBaz: false
```

Config will be loaded into a simple `map[string]string` (could
use `config.ConfigurationMap`) on init by an added ClusterConfig library.
Kube system components and addons will then source the library,
and alpha features will reserve a key in the map to toggle on or off.
If the file is not found, default to empty config.

This design rejected as introducing too much coupling at runtime,
and potentially resulting in 3 different ways to pass config to
components (commandline flag, per-component config file, shared config).

### ConfigMap in API

Defining runtime config as a configmap in the kubernetes API itself
is another option but is more complicated. The apiserver would still need
a way to get the config (either from file or via the existing
`--runtime-config` option), and would then publish that config
at a known endpoint. Other kube system components would have to
be modified to watch the endpoint on startup to initialize features
before starting control loop. A problem with this approach is that
the config map could be deleted/mutated via normal API operations,
which could cause components to start up without config.

## Future work

1. Eventual plan is for component config to be managed by versioned
APIs and not flags. It may make sense (or not) to build support for
toggling alpha features into ComponentConfig (#12245).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtimeconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
