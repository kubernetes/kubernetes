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

# Dynamic Kubelet Configuration

## Abstract

A proposal for making it possible to (re)configure Kubelets in a live cluster by providing config via the API server. Some subordinate items include local checkpointing of Kubelet configuration and the ability for the Kubelet to read config from a file on disk, rather than from command line flags.

## Motivation

The Kubelet is currently configured via command-line flags. This is painful for a number of reasons:
- It makes it difficult to change the way Kubelets are configured in a running cluster, because it is often tedious to change the Kubelet startup configuration (without adding your own configuration management system e.g. Ansible, Salt, Puppet).
- It makes it difficult to manage different Kubelet configurations for different nodes, e.g. if you want to canary a new config or slowly flip the switch on a new feature.
- The current lack of versioned Kubelet configuration means that any time we change Kubelet flags, we risk breaking someone's setup.

## Example Use Cases

- Staged rollout of configuration chages, including tuning adjustments and enabling new Kubelet features.
- Streamline cluster bootstrap. The Kubeadm folks want to plug in to dynamic config, for example: [kubernetes/kubeadm#28](https://github.com/kubernetes/kubeadm/issues/28).
- Making it easier to run tests with different Kubelet configurations, because you can specify the desired Kubelet configuration from the test itself (this is technically already possible with the alpha version of the feature).

## Primary Goals of the Design

K8s should:

- Provide a versioned object to represent the Kubelet configuration.
- Provide the ability to specify a dynamic configuration source to each Kubelet (for example, provide the name of a `ConfigMap` that contains the configuration).
- Provide a way to share the same configuration source between nodes.
- Protect against bad configuration pushes.
- Recommend, but not mandate, the basics of a workflow for updating configuration.

Additionally, we should:

- Add Kubelet support for consuming configuration via a file on disk. This aids work towards deprecating flags in favor of on-disk configuration. This functionality can also be reused for locak checkpointing of Kubelet configuration.
- Make it possible to opt-out of remote configuration as an extra layer of protection. This should probably be a flag so that you can't dynamically turn off dynamic config by accident.

## Design

Two really important questions:
1. How should one organize and represent configuration in a cluster?
2. How should one orchestrate changes to that configuration?

### Organization of the Kubelet's Configuration Type

- We should remove the `HostNameOverride` and `NodeIP` fields from the KubeletConfiguration API object; these should just stay flags for now - They likely do not need to change after node provisioning and keeping them in the configuration struct complicates sharing config objects between nodes (because these values are always node-unique). IN PROGRESS: [#40117](https://github.com/kubernetes/kubernetes/pull/40117).
- The Kubelet's configuration type will no longer align with it's flags; we should add a separate struct that contains the flag variables to prevent them from migrating all over the codebase (similar to work being done in [#32215](https://github.com/kubernetes/kubernetes/issues/32215)). IN PROGRESS: [#40117](https://github.com/kubernetes/kubernetes/pull/40117).
- We should add more structure to the Kubelet configuration for readability.
- We should expose Kubelet API groups via groups in `pkg/kubelet/apis` and move the location of the configuration type to this subtree - e.g. to `pkg/kubelet/apis/config`. IN PROGRESS: [#42759](https://github.com/kubernetes/kubernetes/pull/42759).
- We need to be able to add and remove experimental fields from the `KubeletConfiguration` without having to rev the API version. A simple solution is to just have a string representation of the experimental fields as part of the KubeletConfiguration, that the Kubelet can parse as necessary. THIS OPTION: [#41082](https://github.com/kubernetes/kubernetes/pull/41082). That said, this option is essentially a hack, and it is likely better just to rely on the standard API versioning mechanism. The overhead of up-versioning `pkg/kubelet/apis/config` is low enough to do so, because the former only relates to one component. This is much easier than versioning `pkg/apis/componentconfig`. Eventually, fine-grain field versioning ([#34508](https://github.com/kubernetes/kubernetes/issues/34508)) will solve this entirely, but it may be a while before that work is complete.


### Representing and Referencing Configuration

#### Cluster-level object
- The Kubelet's configuration should be stored as a `JSON` or `YAML` blob under the `kubelet` key in a `ConfigMap` object. This allows the `ConfigMap`'s schema to be extended to other `Node`-level components, e.g. adding a key for `kube-proxy`.
- On local disk, the Kubelet's configuration should be stored in a `.json` or `.yaml` file.
- The Kubelet's configuration type should be organized in the cluster as a structured monolith. 
  + *Structured*, so that it is readable.
  + *Monolithic*, so that all Kubelet parameters roll out to a given `Node` in unison.
- The `ConfigMap` containing the desired configuration should be specified via the `Node` object corresponding to the Kubelet. Eventually, the `Node` may even have a `NodeConfig` field, but we should start by using an annotation, because additions to GA APIs are hard to change later if we get them wrong.

`ConfigMap` object containing just the Kubelet's configuration:
```
kind: ConfigMap
metadata:
  name: node-config-{hash of `data` for verification}
data:
  kubelet: "{JSON blob}"
```

With an additional `Node`-level component:
```
kind: ConfigMap
metadata:
  name: node-config-{hash of `data` for verification}
data:
 kubelet: "{JSON blob}"
 kube-proxy: "YAML blob"
```

#### On-disk

- The Kubelet should accept a `--config-dir` flag that specifies a directory (e.g. `config-dir`) for storing `ConfigMap` contents.
- The Kubelet will use this directory to checkpoint downloaded `ConfigMap`s.
  + The name of each subdirectory of `config-dir` is the same as the name of the corresponding `ConfigMap`, e.g. `node-config-{hash}`.
  + Each subdirectory of `config-dir` contains the `Data` of the corresponding `ConfigMap` as a set of files. Given the above `ConfigMap` examples, reading `config-dir/node-config-{hash}/kubelet` would return `{JSON blob}`, reading `config-dir/node-config-{hash}/kube-proxy` would return `YAML blob`.
- The user can additionally pre-populate `config-dir/init` with an initial set of configuration files, to be used prior to the `Kubelet` being able to access the API server (including when running in standalone mode).

### Orchestration of configuration

There are a lot of opinions around how to orchestrate configuration in a cluster. The following items form the basis of a robust solution:

#### Robust Kubelet behavior

To make config updates robust, the Kubelet should be able to locally and automatically recover from bad config pushes. We should strive to avoid requiring operator intervention, though this may not be possible in all scenarios.

Recovery involves:
- Checkpointing configuration on-disk, so prior versions are locally available for rollback.
- Tracking a LKG (last-known-good) configuration, which will be the rollback target if the current configuration turns out to be bad.
- Tracking the health of the Kubelet against a given configuration and remembering, at least for a time, if a certain configuration correlated with poor health.
- Rolling back to LKG if the current configuration resulted in poor health.
- Providing operators an escape hatch for exiting dead-end states.

##### Finding and checkpointing intended configuration

The Kubelet finds its intended configuration by looking for the name of a `ConfigMap` via it's `Node`'s `node-config` annotation. For now, this `ConfigMap` is assumed to be in the `kube-system` namespace. In the future, we should use an actual field on the `Node` and an `ObjectRef` instead of a string name.

If the annotation is absent, the Kubelet will use its `config-dir/init` configuration, or built-in defaults if `config-dir/init` is also absent.

The mappings `node-config: ""` and `node-config: "init"` will be treated as intentions to use `config-dir/init`. If `config-dir/init` is not present, the Kubelet should fall back to its built-in defaults and report the absence of `config-dir/init` in the node status.

If the referenced `ConfigMap` does not exist, the Kubelet will continue using its current configuration and report the non-existence via the node status.

If the Kubelet can find the referenced `ConfigMap`, it then downloads this `ConfigMap` to `config-dir`, storing each `Data` key's contents in a file as described above in the *Representing and Referencing Configuration* section.

The Kubelet sets its current configuration by directing a symlink called `_current`, which lives at `config-dir/_current`, to point at the directory associated with the desired `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a `ConfigMap` is downloaded, and then point `config-dir/_current` to the associated directory.

To begin using a new configuration, the Kubelet simply sets `config-dir/_current`, calls `os.Exit(0)`, and relies on the process manager (e.g. `systemd`) to restart it.

The Kubelet detects new configuration by watching the `Node` object for changes to the `node-config` annotation. When the Kubelet detects new configuration, it checkpoints it as necessary, sets `config-dir/_current`, and restarts to begin using it.

##### Verifying Configuration Integrity

The name of the `ConfigMap` in several of the above examples is `node-config-{hash}`. The `{hash}` portion is a hash of the `YAML` serialization of the `ConfigMap`'s `Data` field at creation time, and can be used to verify that its contents at Kubelet-download-time are the same as at user-creation-time. This helps protect against unintentional `ConfigMap` corruption (the `Data` fields of the `ConfigMap` may be accidentally mutated after its configuration). Today, the user would have to manually generate this hash when creating the `ConfigMap`.

##### Possible Metrics for Bad Configuration

These are possible metrics we can use to detect bad configuration. Some are perfect indicators (validation) and some are imperfect (`P(ThinkBad == ActuallyBad) < 1`).

Metrics:
- `KubeletConfiguration` cannot be deserialized
- `KubeletConfiguration` fails a validation step
- Kubelet restarts occur above a frequency threshold when using a given configuration

Possible extensions depending on what the Kubelet is responsible for:
- sudden loss of network or degredation of network performance

##### Tracking LKG (last-known-good) configuration

Any initial on-disk configuration (`config-dir/init`) will be automatically considered good. This should be ok, because operators should notice if a `Node` doesn't initially spin up in a healthy state.

Any configuration retrieved from the API server must persist beyond a probationary period before it can be considered LKG. If `bad(config)` is the threshold for bad configuration, then `bad^-1(config) + (use_time > threshold)` is the point at which we adopt a new configuration as LKG.

The Kubelet will track its LKG configuration by directing a symlink called `_lkg`, which lives at `config-dir/_lkg`, to point at the directory associated with the LKG `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a downloaded `ConfigMap` passes the probationary period required to become the LKG. If the symlink does not exist, and the current configuration is considered "bad," the Kubelet will roll back to its built-in defaults.

##### Rolling back to the LKG config

When a configuration correlates too strongly with poor health, the Kubelet will "roll-back" to its last-known-good configuration. This process involves three components:
1. The Kubelet must begin using its LKG configuration instead of its intended current configuration.
2. The Kubelet must remember which configuration was bad, so it doesn't just roll forward to that configuration again.
3. The Kubelet must report that it rolled back to LKG due to the *belief* that it had a bad configuration.

Regarding (1), the Kubelet will set `config-dir/_current` to point to the same location as `config-dir/_lkg`, and then call `os.Exit(0)`, as above.

Regarding (2), when the Kubelet detects a bad configuration, it will add an entry to a file, `config-dir/bad-configs.json`, mapping the name of the `ConfigMap` to the time at which it was determined to be a bad config, and the reason. The Kubelet will not roll forward to any of these configurations again until their entries are removed from this file. For example, the contents of this file might look like:
```
{
  "node-config-{hash}": {
    "time": "YYYY-MM-DDThh:mm:ss.sTZD", 
    "reason": "The configuration failed validation."
  }
}
```

Regarding (3), the Kubelet should report via the `Node`'s status:
- That it is using LKG.
- The `ConfigMap` LKG referrs to.
- The supposedly bad `ConfigMap` that the Kubelet decided to avoid.
- The reason it thinks the `ConfigMap` is bad.

##### Dead-end states

We may use imperfect indicators to detect bad configuration. Thus, the Kubelet may revert to LKG when the configuration was not actually the issue. These are two possible solutions to this problem:
- Allow the belief that a configuration is bad to decay over time. The user may adjust this period to balance time to respond against time to automatically recover from false alarms. This period would be measured against the timestamp in `config-dir/bad-configs.json`.
- Provide a Kubelet endpoint, e.g. `/accept-node-config`, which forces the Kubelet to remove the `config-dir/bad-configs.json` entry for the `ConfigMap` currently referenced by the `Node`'s `node-config` annotation, and then restart to take up the `Node`'s specified config.

Additionally, imagine that one configuration prevents the Kubelet from accessing the API server to detect new configurations. There are at least a few options here:
- Correlate API server connectivity with the configuration lifetime and use it to detect bad configuration. This, of course, means that network issues may cause false alarms regarding bad config.
- Provide a Kubelet endpoint, e.g. `/mark-config-bad`, that forces the Kubelet to mark its current configuration as bad. This will cause it to roll back to LKG. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/_current`. Thus, just after restarting, the Kubelet should detect the replacement configuration. If for some reason there is not a replacement, or the replacement cannot be detected, the Kubelet will just end up using LKG.

If the problem is *really* bad - e.g. so bad that the Kubelet isn't serving endpoints, the user should set the new configuration via the API server and then reboot the affected `Node`. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/_current`, in case `config-dir/_current` referenced a really-bad-configuration and the user had to swap this out.

### Extension to additional components

#### Plumbing configuration to components that run in pods

How do we plumb configuration that should be managed (e.g. rolled out) at the `Node` level to `Pod`s on the `Node`? These could be static pods, or daemonsets, etc. Here's one idea:

Add a volume source that allows the `Node`'s configuration to be exposed in the filesystem of the `Pod`.

For example, plumbing configuration into the `kube-proxy`:
```
kind: Pod
  metadata:
    name: kube-proxy
  spec:
    containers:
      - name: kube-proxy
        volumeMounts:
          - mountPath: /etc/node-config
            name: node-config
            readOnly: true
    volumes:
      - nodeConfig:
          name: node-config
```

This volume will be backed by the `config-dir/_current` symlink. When the Kubelet downloads a new configuration and updates this symlink, the configuration update will become simultaneously visible to all `Pod`s using a `NodeConfig` volume. The `Pod`s could, for example, use `inotify` on the mounted directory to watch for changes to the configuration.

This volume should always be mounted read-only. It would be ideal if the volume source could enforce this, regardless of what is set in the `Pod` spec.

It may be advantageous to restrict the exposed configuration to only the necessary keys. For example, `kube-proxy` only sees the data for the `kube-proxy` configuration key in the `ConfigMap`:

```
kind: Pod
  metadata:
    name: kube-proxy
  spec:
    containers:
      - name: kube-proxy
        volumeMounts:
          - mountPath: /etc/node-config
            name: node-config
            readOnly: true
    volumes:
      - nodeConfig:
          name: node-config
          keys:
            - kube-proxy
```

That said, per-key filtering may be considerably more difficult to implement.

Users can manage permission to mount `NodeConfig` via [PodSecurityPolicy](https://kubernetes.io/docs/user-guide/pod-security-policy/) ([proposal](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/security-context-constraints.md)), which can restrict the types of volumes that a `Pod` is allowed to mount.

The use of this pipeline is, of course, totally optional for any component but the Kubelet. It is useful for bundling configuration for `Node` components into a single rollout. But there are other ways to update daemon configurations. For example: have two `DaemonSet`s represent two configurations for the same component, and label `Nodes` with which `DaemonSet` to use. The similarity in rollout mechanism (progressively update `Node` objects) should be noted. This proposal does not dictate *how* that `Node` level information is updated; orchestration is open-ended.

### Operational Considerations

#### Rollout workflow

Kubernetes does not have the concepts of immutable, or even undeleteable API objects. This makes it easy to shoot yourself in the foot by modifying or deleting a `ConfigMap`. This results in undefined behavior given the behaviors described in this document, because the assumption is that these `ConfigMaps` are not mutated once deployed. For example, this design includes no method for invalidating the Kubelet's local cache of configurations, so there is no concept of eventually consistent results from edits or deletes of a `ConfigMap`. You may, in such a scenario, end up with partially consistent results or no results at all.

Thus, we recommend that rollout workflow consist only of creating new `ConfigMap` objects and updating the `node-config` annotation on each `Node` to point to that new object. This results in a controlled rollout with well-defined behavior.

There is discussion in [#10179](https://github.com/kubernetes/kubernetes/issues/10179) regarding ways to prevent unintentional mutation and deletion of objects.

## Additional concerns not-yet-addressed

### Monitoring configuration status

- A way to query/monitor the config in-use on a given node. Today this is possible via the configz endpoint, but there are a number of other potential solutions, e.g. exposing live config via Prometheus.
- Once the Kubelet tracks restarts to detect bad configuration, it would be useful to report, e.g. via `NodeStatus`, that the Kubelet thinks it is in a bad configuration state.

### Orchestration
- A specific orchestration solution for rolling out kubelet configuration. There are several factors to think about, including these general rollout considerations:
    + Selecting the objects that the new config should be rolled out to (Today it is probably okay to roll out `Node` config with the intention that the nodes are eventually homogeneously configured across the cluster. But what if a user intentionally wants different sets of nodes to be configured differently? A cluster running multiple separate applications, for example.).
    + Specifying the desired end-state of the rollout.
    + Specifying the rate at which to roll out.
    + Detecting problems with the rollout and automatically stopping bad rollouts.
    + Specifying how risk-averse to be when deciding to stop a rollout.
    + Recording a history of rollouts so that it is possible to roll back to previous versions.
    + Properly handling objects that are added or removed while a configuration rollout is in process. For example, `Nodes` might be added or removed due to an autoscaling feature, etc.
    + Reconciling configuration with objects added after the completion of a rollout, e.g. new `Nodes`.
    + Pausing/resuming a rollout.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/dynamic-kubelet-settings.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
