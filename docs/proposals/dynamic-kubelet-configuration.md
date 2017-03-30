# Dynamic Kubelet Configuration

## Abstract

A proposal for making it possible to (re)configure Kubelets in a live cluster by providing config via the API server. Some subordinate items include local checkpointing of Kubelet configuration and the ability for the Kubelet to read config from a file on disk, rather than from command line flags.

## Motivation

The Kubelet is currently configured via command-line flags. This is painful for a number of reasons:
- It makes it difficult to change the way Kubelets are configured in a running cluster, because it is often tedious to change the Kubelet startup configuration (without adding your own configuration management system e.g. Ansible, Salt, Puppet).
- It makes it difficult to manage different Kubelet configurations for different nodes, e.g. if you want to canary a new config or slowly flip the switch on a new feature.
- The current lack of versioned Kubelet configuration means that any time we change Kubelet flags, we risk breaking someone's setup.

## Example Use Cases

- Staged rollout of configuration changes, including tuning adjustments and enabling new Kubelet features.
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

- Add Kubelet support for consuming configuration via a file on disk. This aids work towards deprecating flags in favor of on-disk configuration. This functionality can also be reused for local checkpointing of Kubelet configuration.
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
- The Kubelet's configuration type should be organized in the cluster as a structured monolith. 
  + *Structured*, so that it is readable.
  + *Monolithic*, so that all Kubelet parameters roll out to a given `Node` in unison.
- The `ConfigMap` containing the desired configuration should be specified via the `Node` object corresponding to the Kubelet. The `Node` will have a new `spec` subfield, `config`, which is an `ObjectReference` intended to refer to a `ConfigMap`.
- The name of the `ConfigMap` containing the desired configuration should be of the form `blah-blah-blah-{hash}`, where `{hash}` is a SHA-1 hash of the `data` field of the `ConfigMap`. The hash will be produced by serializing the `data` to a `JSON` string, and then taking the hash of this string. Depending on ordering guarantees, we may also need to ensure that keys are sorted in the serialization to ensure consistent hashing.
  + The Kubelet will verify the downloaded `ConfigMap` by performing this same procedure and comparing the result to the hash in the name. This helps prevent the "shoot yourself in the foot" scenario detailed below in *Operational Considerations/Rollout workflow*.

`ConfigMap` object containing just the Kubelet's configuration:
```
kind: ConfigMap
metadata:
  name: node-config-{sha1 hash of ConfigMap, sans identifying information, for verification}
data:
  kubelet: "{JSON blob}"
```

#### On-disk

- The Kubelet should accept a `--config-dir` flag that specifies a directory (e.g. `config-dir`).
- The Kubelet will use this directory to checkpoint downloaded `ConfigMap`s.
  + When the Kubelet downloads a `ConfigMap`, it will save its contents to a directory named after the UID of that `ConfigMap`, in a subdirectory of `config-dir` called `configmaps`, e.g. `config-dir/configmaps/{uid}`.
  + The directory `config-dir/configmaps/{uid}` contains the `Data` of the corresponding `ConfigMap` as a set of files. Given the most recent `ConfigMap` example above, reading `config-dir/configmaps/{uid-of-the-example}/kubelet` would return `{JSON blob}`.
- The user can additionally pre-populate `config-dir/init` with an initial set of configuration files, to be used prior to the Kubelet being able to access the API server (including when running in standalone mode).

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

The Kubelet finds its intended configuration by looking for the `ConfigMap` referenced via it's `Node`'s `spec.config` field. This field is an `ObjectReference`. We consider this field to be "empty" if it lacks the information necessary to resolve to an object. The field must provide at least `name` or `uid`, to be considered "non-empty." If no `namespace` or `uid` is provided, the `default` namespace is assumed. 

The Kubelet must have permission to read the namespace that contains the referenced `ConfigMap`. 

If the field is empty, the Kubelet will use its `config-dir/init` configuration, or built-in defaults if `config-dir/init` is also absent.

If `config-dir/init` is not present, the Kubelet should fall back to its built-in defaults.

If the referenced `ConfigMap` does not exist, the Kubelet will continue using its current configuration and report the non-existence via the node status.

If the Kubelet can find the referenced `ConfigMap`, it then downloads this `ConfigMap` to `config-dir`, storing each `Data` key's contents in a file as described above in the *Representing and Referencing Configuration* section.

The Kubelet sets its current configuration by directing a symlink called `_current`, which lives at `config-dir/_current`, to point at the directory associated with the desired `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a `ConfigMap` is downloaded, and then point `config-dir/_current` to the associated directory.

To begin using a new configuration, the Kubelet simply sets `config-dir/_current`, calls `os.Exit(0)`, and relies on the process manager (e.g. `systemd`) to restart it.

The Kubelet detects new configuration by watching the `Node` object for changes to the `NodeConfig` field. When the Kubelet detects new configuration, it checkpoints it as necessary, sets `config-dir/_current`, and restarts to begin using it.

##### Metrics for Bad Configuration

These are metrics we can use to detect bad configuration. Some are perfect indicators (validation) and some are imperfect (`P(ThinkBad == ActuallyBad) < 1`).

Perfect Metrics:
- `KubeletConfiguration` cannot be deserialized
- `KubeletConfiguration` fails a validation step

Imperfect Metrics:
- Kubelet restarts occur above a frequency threshold when using a given configuration, before that configuration is out of a "trial period."

We should use the perfect metrics we have available. These immediately tell us when we have bad configuration. Adding a user-defined trial period, within which crash loops can be attributed to the current configuration, adds some protection against more complex configuration mishaps.  

More advanced error detection probably requires an out-of-band component, like the Node Problem Detector. We shouldn't go overboard attempting to make the Kubelet too smart. 

##### Tracking LKG (last-known-good) configuration

The Kubelet will track its LKG configuration by directing a symlink called `_lkg`, which lives at `config-dir/_lkg`, to point at the directory associated with the LKG `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a downloaded `ConfigMap` passes the trial period required to become the LKG. If the symlink does not exist and the current configuration is determined to be bad, the Kubelet will roll back to its built-in defaults.

Any configuration retrieved from the API server must persist beyond a trial period before it can be considered LKG. This trial period will be called `ConfigTrialPeriod`, will be a `Duration` as defined by `k8s.io/apimachinery/pkg/apis/meta/v1/duration.go`, and will be a parameter of the `KubeletConfiguration`. The trial period on a given configuration is the trial period used for that configuration (as opposed to, say, using the trial period set on the previous configuration). This is useful if you have, for example, a configuration you want to roll out with a longer trial period for additional caution.

When the time since the `_current` symlink was set to point at the new configuration exceeds the trial period, `_lkg` will be set to point to the same configuration as `_current`.

The init configuration (`config-dir/init`) will be automatically considered good. If a node is provisioned with an init configuration, it MUST be a valid configuration. The Kubelet will always attempt to deserialize the init configuration and validate it on startup, regardless of whether a remote configuration exists. If this fails, the Kubelet will submit an appropriate message to `Node.status.conditions` and enter a crash loop. We want invalid init configurations to be extremely obvious.

This is very important, because the init configuration is the initial last-known-good configuration. If the init configuration turns out to be bad, there is nothing to fall back to. We presume a user provisions nodes with an init configuration when the Kubelet defaults are inappropriate for their use case. It would thus be inappropriate to fall back to the Kubelet defaults if the init configuration exists.

As the init configuration and the built-in defaults are automatically considered good, intentionally setting `spec.config` on the `Node` to its empty default will reset the last-known-good symlink. If `config-dir/init` exists, the symlink will be updated to point there. If `config-dir/init` does not exist, the symlink will be deleted, meaning the built-in defaults become the last-known-good config.

##### Rolling back to the LKG config

When a configuration correlates too strongly with poor health, the Kubelet will "roll-back" to its last-known-good configuration. This process involves three components:
1. The Kubelet must begin using its LKG configuration instead of its intended current configuration.
2. The Kubelet must remember which configuration was bad, so it doesn't just roll forward to that configuration again.
3. The Kubelet must report that it rolled back to LKG due to the *belief* that it had a bad configuration.

Regarding (1), the Kubelet will set `config-dir/_current` to point to the same location as `config-dir/_lkg`, and then call `os.Exit(0)`, as above.

Regarding (2), when the Kubelet detects a bad configuration, it will add an entry to a file, `config-dir/bad-configs.json`, mapping the namespace and name of the `ConfigMap` to the time at which it was determined to be a bad config, and the reason. The Kubelet will not roll forward to any of these configurations again until their entries are removed from this file. For example, the contents of this file might look like (shown here with a `reason` matching what would be reported in the `Node`'s `status.conditions`):
```
{
  "{uid}": {
    "time": "YYYY-MM-DDThh:mm:ss.sTZD", 
    "reason": "failed deserialize: namespace/name.kubelet; search 'config deserialize' in Kubelet log for details"
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
- Provide a Kubelet endpoint, e.g. `/accept-node-config`, which forces the Kubelet to remove the `config-dir/bad-configs.json` entry for the `ConfigMap` currently referenced by the `Node`'s `NodeConfig` field, and then restart to take up the `Node`'s specified config.

Additionally, imagine that one configuration prevents the Kubelet from accessing the API server to detect new configurations. There are at least a few options here:
- Correlate API server connectivity with the configuration lifetime and use it to detect bad configuration. This, of course, means that network issues may cause false alarms regarding bad config.
- Provide a Kubelet endpoint, e.g. `/mark-config-bad`, that forces the Kubelet to mark its current configuration as bad. This will cause it to roll back to LKG. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/_current`. Thus, just after restarting, the Kubelet should detect the replacement configuration. If for some reason there is not a replacement, or the replacement cannot be detected, the Kubelet will just end up using LKG.

If the problem is *really* bad - e.g. so bad that the Kubelet isn't serving endpoints, the user should set the new configuration via the API server and then reboot the affected `Node`. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/_current`, in case `config-dir/_current` referenced a really-bad-configuration and the user had to swap this out.

##### Reporting Configuration Status

Succinct messages related to the state of `Node` configuration should be reported in a `NodeCondition`, in `status.conditions`. These should inform users which `ConfigMap` the Kubelet is using, and if the Kubelet has detected any issues with the configuration. The Kubelet should report this condition during startup, after attempting to validate configuration but before actually using it, so that the chance of a bad configuration inhibiting status reporting is minimized.

All `NodeCondition`s contain the fields: `lastHeartbeatTime:Time`, `lastTransitionTime:Time`, `message:string`, `reason:string`, `status:string(True|False|Unknown)`, and `type:string`.

These are some brief descriptions of how these fields should be interpreted for node-configuration related conditions:
- `lastHeartbeatTime`: The last time the Kubelet updated the condition. The Kubelet will typically do this whenever it is restarted, because that is when configuration changes occur. The Kubelet will update this on restart regardless of whether the configuration, or the condition, has changed.
- `lastTransitionTime`: The last time this condition changed. The Kubelet will not update this unless it intends to set a different condition than is currently set.
- `message`: Think of this as the "effect" of the `reason`.
- `reason`: Think of this as the "cause" of the `message`.
- `status`: `True` if the currently set configuration is considered OK, `False` otherwise. `Unknown` should not be used in these condition messages. One *might* say that a config in its trial period has "unknown" goodness, but the `status` should still be `True` because the Kubelet is treating it as OK.
- `type`: `ConfigOK` will always be used for these conditions.

The following list of example conditions, sans `type`, `lastHeartbeatTime`, and `lastTransitionTime`, can be used to get a feel for the relationship between `message`, `reason`, and `status`:

Config still in trial period:
```
message: "using current: namespace/name"
reason: "in trial period: namespace/name"
status: "True"
```

Config passed trial period:
```
message: "using current: namespace/name"
reason: "passed trial period: namespace/name"
status: "True"
```

No remote config specified:
```
message: "using local init config"
reason: "this Node's spec.config was empty"
status: "True"
```

If `Node.spec.config` refers to an object which is not a `ConfigMap`, we treat similarly to an empty `spec.config`, but report status `False`, as this is likely an error:
```
message: "using local init config"
reason: "this Node's spec.config does not refer to a ConfigMap"
status: "False"
```

No remote config specified, no local `init` config provided:
```
message: "using defaults"
reason: "this Node's spec.config was empty, and local init config does not exist"
status: "True"
```

When reading or validation fails, the specific `ConfigMap` key(s) should be indicated. As `reason`s should be brief, the precise details of the error should be available in the Kubelet log. The `reason` should specify a short string to search for in the Kubelet log, to make finding the error easier.

Deserialization error on one key:
```
message: "using last known good: namespace/name"
reason: "failed deserialize: namespace/name.kubelet; search 'config deserialize' in Kubelet log for details"
status: "False"
```

Validation errors on multiple keys:
```
message: "using last known good: namespace/name"
reason: "failed validation: namespace/name.kubelet,kube-proxy; search 'config validation' in Kubelet log for details"
status: "False"
```

Kubelet enters crash loop during a configuration trial period:
```
message: "using last known good: namespace/name"
reason: "failed trial period due to crash loop: namespace/name"
status: "False"
```

If the init config fails validation:
```
message: "local init config is bad, Kubelet will crash loop until this is fixed!"
reason: "failed validation: local init config"
status: "False"
```

### Operational Considerations

#### Rollout workflow

Kubernetes does not have the concepts of immutable, or even undeleteable API objects. This makes it easy to "shoot yourself in the foot" by modifying or deleting a `ConfigMap`. This results in undefined behavior given the behaviors described in this document, because the assumption is that these `ConfigMaps` are not mutated once deployed. For example, this design includes no method for invalidating the Kubelet's local cache of configurations, so there is no concept of eventually consistent results from edits or deletes of a `ConfigMap`. You may, in such a scenario, end up with partially consistent results or no results at all.

Thus, we recommend that rollout workflow consist only of creating new `ConfigMap` objects and updating the `NodeConfig` field on each `Node` to point to that new object. This results in a controlled rollout with well-defined behavior.

There is discussion in [#10179](https://github.com/kubernetes/kubernetes/issues/10179) regarding ways to prevent unintentional mutation and deletion of objects.

## Additional concerns not-yet-addressed

### Monitoring configuration status

- A way to query/monitor the config in-use on a given node. Today this is possible via the configz endpoint, but this is just for debugging, not production use. There are a number of other potential solutions, e.g. exposing live config via Prometheus.

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
