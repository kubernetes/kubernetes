# Overview

Proposes adding a `--feature-config` to core kube system components:
apiserver , scheduler, controller-manager, kube-proxy, and selected addons.
This flag will be used to enable/disable alpha features on a per-component basis.

## Motivation

Motivation is enabling/disabling features that are not tied to
an API group. API groups can be selectively enabled/disabled in the
apiserver via existing `--runtime-config` flag on apiserver, but there is
currently no mechanism to toggle alpha features that are controlled by
e.g. annotations. This means the burden of controlling whether such
features are enabled in a particular cluster is on feature implementors;
they must either define some ad hoc mechanism for toggling (e.g. flag
on component binary) or else toggle the feature on/off at compile time.

By adding a`--feature-config` to all kube-system components, alpha features
can be toggled on a per-component basis by passing `enableAlphaFeature=true|false`
to `--feature-config` for each component that the feature touches.

## Design

The following components will all get a `--feature-config` flag,
which loads a `config.ConfigurationMap`:

- kube-apiserver
- kube-scheduler
- kube-controller-manager
- kube-proxy
- kube-dns

(Note kubelet is omitted, it's dynamic config story is being addressed
by #29459). Alpha features that are not accessed via an alpha API
group should define an `enableFeatureName` flag and use it to toggle
activation of the feature in each system component that the feature
uses.

## Suggested conventions

This proposal only covers adding a mechanism to toggle features in
system components. Implementation details will still depend on the alpha
feature's owner(s). The following are suggested conventions:

- Naming for feature config entries should follow the pattern
  "enable<FeatureName>=true".
- Features that touch multiple components should reserve the same key
  in each component to toggle on/off.
- Alpha features should be disabled by default. Beta features may
  be enabled by default. Refer to docs/devel/api_changes.md#alpha-beta-and-stable-versions
  for more detailed guidance on alpha vs. beta.

## Upgrade support

As the primary motivation for cluster config is toggling alpha
features, upgrade support is not in scope. Enabling or disabling
a feature is necessarily a breaking change, so config should
not be altered in a running cluster.

## Future work

1. The eventual plan is for component config to be managed by versioned
APIs and not flags (#12245). When that is added, toggling of features
could be handled by versioned component config and the component flags
deprecated.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtimeconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
