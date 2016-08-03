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

Proposes a mechanism for specifying cluster-wide runtime config
for all kube system components via a single config file. This will
provide an easy way to specify init settings that are shared between
system components.

## Motivation

Primary motivation is enabling/disabling features that are not tied to
an API group. API groups can be selectively enabled/disabled in the
apiserver via the `--runtime-config` commandline flag, but there is
currently no mechanism to toggle alpha features that are controlled by
e.g. annotations. This means the burden of controlling whether such
features are enabled in a particular cluster is on feature implementors;
they must either define some ad hoc mechanism for toggling (e.g. flag
on component binary) or else toggle the feature on/off at compile time.

Providing a mechanism for specifying cluster-wide config makes
deploying non-API features simpler for implementors and cluster admins.

## Design

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

## Alternative designs

### Common flag

Another option is to have every kube-system binary define a
`--runtime-config` flag. The disadvantage to that approach is
that commandline flags are harder to manage, since kubernetes
deployment solutions typically use templating to define the command
for kube components. It's also common to containerize kube-system
components, meaning runtime config would have to be stuck in the
container command of a manifest; also harder to manage.

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

## Upgrade support

As the primary motivation for cluster config is toggling alpha
features, upgrade support is not in scope. Enabling or disabling
a feature is necessarily a breaking change, so config should
not be altered in a running cluster.

## Future work

1. Consider migrating existing `--runtime-config` settings on
kube-apiserver into cluster config file and deprecating the
`--runtime-config` flag.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/clusterconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
