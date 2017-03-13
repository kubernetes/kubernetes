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

This doc primarily focuses on (1) and the downstream aspects of (2).

### Organization of the Kubelet's Configuration Type

In general, components should expose their configuration types from their own source trees. The types are currently in the alpha `componentconfig` API group, and should be broken out into the trees of their individual components. PR [#42759](https://github.com/kubernetes/kubernetes/pull/42759) reorganizes the Kubelet's tree to facilitate this.

Components with several same-configured instances, like the Kubelet, should be able to share configuration sources. A 1:N mapping of config-object:instances is much more efficient than requiring a config object per-instance. As one example, we removed the `HostNameOverride` and `NodeIP` fields from the configuration type because these cannot be shared between Nodes - [#40117](https://github.com/kubernetes/kubernetes/pull/40117).

Components that currently take command line flags should not just map these flags directly into their configuration types. We should, in general, think about which parameters make sense to configure dynamically, which can be shared between instances, and which are so low-level that they shouldn't really be exposed on the component's interface in the first place. Thus, the Kubelet's flags should be kept separate from configuration - [#40117](https://github.com/kubernetes/kubernetes/pull/40117).

The Kubelet's current configuration type is an unreadable monolith. We should decompose it into sub-objects for convenience of composition and management. An example grouping is in PR [#44252](https://github.com/kubernetes/kubernetes/pull/44252).

### Representing and Referencing Configuration

#### Cluster-level object
- The Kubelet's configuration should be stored as a `JSON` or `YAML` blob under the `kubelet` key in a `ConfigMap` object. This allows the `ConfigMap`'s schema to be extended to other `Node`-level components, e.g. adding a key for `kube-proxy`.
- The Kubelet's configuration information should be organized in the cluster as a structured monolith. 
  + *Structured*, so that it is readable.
  + *Monolithic*, so that all Kubelet parameters roll out to a given `Node` in unison.
  + Note that this does not mean the type itself has to be monolithic, just that everything the Kubelet needs makes it to the `Node` in one piece.
- The `ConfigMap` containing the desired configuration should be specified via the `Node` object corresponding to the Kubelet. The `Node` will have a new `spec` subfield, `config`, which is a new type, `NodeConfigSource` (described below).
- The name of the `ConfigMap` containing the desired configuration should be of the form (using Go regexp syntax) `^(?P<nameDash>(?P<name>[a-z0-9.\-]*){0,1}-){0,1}(?P<alg>[a-z0-9]+-(?P<hash>[a-f0-9]+)$`, where `name` is a human-readable identifier, `nameDash` is just a helper to handle the dash separator between `name` and `alg`, `alg` is the hash algorithm to use, and `hash` is the lowercase hexadecimal value of the hash, as would be formatted by Go's `%x`.
  + For now, the only supported hash algorithm is `sha256`.
  + The hash will be produced extracting the key-value pairs from the `ConfigMap`'s `Data` field into a list, sorting them in byte-alphabetic order by key, serializing this sorted list to a string of the format `key:value,key:value,...,` (trailing comma only if `Data` is non-empty) and using `alg` to produce the hash of the string. For example, see this reference implementation in Go:
  ```
  // Serializes m into a string of pairs, in byte-alphabetic order by key, and takes the hash using alg.
  // Keys and values are separated with `:` and pairs are separated with `,`. If m is non-empty,
  // there is a trailing comma in the pre-hash serialization. If m is empty, there is no trailing comma.
  func dataHash(alg string, m map[string]string) (string, error) {
    // extract key-value pairs from data
    kv := kvPairs(m)
    // sort based on keys
    sort.Slice(kv, func(i, j int) bool {
      return kv[i][0] < kv[j][0]
    })
    // serialize to a string
    s := ""
    for _, p := range kv {
      s = s + p[0] + ":" + p[1] + ","
    }
    // return the hash
    return hash(alg, s)
  }
  
  // extract the key-value pairs from m
  func kvPairs(m map[string]string) [][]string {
    kv := make([][]string, len(m))
    i := 0
    for k, v := range m {
      kv[i] = []string{k, v}
      i++
    }
    return kv
  }
  
  const (
    sha256Alg = "sha256"
  )
  
  // hashes data with alg if alg is supported
  func hash(alg string, data string) (string, error) {
    // take the hash based on alg
    switch alg {
    case sha256Alg:
      sum := sha256.Sum256([]byte(data))
      return fmt.Sprintf("%x", sum), nil
    default:
      return "", fmt.Errorf("requested hash algorithm %q is not supported", alg)
    }
  }
  ```
  + The Kubelet will verify the downloaded `ConfigMap` by performing this same procedure and comparing the result to the hash in the name. This helps prevent the "shoot yourself in the foot" scenario detailed below in *Operational Considerations/Rollout workflow*.

`ConfigMap` object containing just the Kubelet's configuration:
```
kind: ConfigMap
metadata:
  name: node-config-sha256-{hash of ConfigMap, sans identifying information, for verification}
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
- Tracking the state of the Kubelet against a given configuration and remembering, at least for a time, if a certain configuration correlated with a crash loop.
- Rolling back to LKG if the current configuration resulted in a crash loop.
- Providing operators an escape hatch for exiting dead-end states.

##### Finding and checkpointing intended configuration

The Kubelet finds its intended configuration by looking for the `ConfigMap` referenced via it's `Node`'s optional `spec.configSource` field. This field will be a new type:
```
type NodeConfigSource struct {
  ConfigMapRef *ObjectReference
}
```

For now, this type just contains an `ObjectReference`. The `spec.configSource` field will be of type `*NodeConfigSource`, because it is optional. 

The `spec.configSource` field can be considered "correct," "empty," or "invalid." The field is "empty" if and only if it is `nil`. The field is "correct" if and only if it is neither "empty" nor "invalid." The field is "invalid" if it fails to meet any of the following criteria:
- Exactly one subfield of `NodeConfigSource` must be non-`nil`.
- All information contained in the non-`nil` subfield meets the requirements of that subfield.

The requirements of the `ConfigMapRef` subfield are as follows:
- All of  `ConfigMapRef.UID`, `ConfigMapRef.Namespace`, and `ConfigMapRef.Name`must be non-empty. 
- All of  `ConfigMapRef.UID`, `ConfigMapRef.Namespace`, and `ConfigMapRef.Name` must unambiguously resolve to the same object.
- The referent must exist.
- The referent must be a `ConfigMap` object.
- As noted above, the referent `ConfigMap`'s name must be of the form `^(?P<nameDash>(?P<name>[a-z0-9.\-]*){0,1}-){0,1}(?P<alg>[a-z0-9]+-(?P<hash>[a-f0-9]+)$`.

The Kubelet must have permission to read the namespace that contains the referenced `ConfigMap`. 

If the `spec.configSource` is empty, the Kubelet will use its `config-dir/init` configuration, or built-in defaults if `config-dir/init` is also absent.

If the `spec.configSource` is invalid, the Kubelet will defer to its last-known-good configuration and report the error via a `NodeCondition` in `Node.status.conditions` (described later in this proposal).

If the `spec.configSource` is correct and using `ConfigMapRef`, the Kubelet downloads this `ConfigMap` to `config-dir`, storing each `Data` key's contents in a file as described above in the *Representing and Referencing Configuration* section.

The Kubelet sets its current configuration by directing a symlink called `.cur`, which lives at `config-dir/.cur`, to point at the directory associated with the desired `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a `ConfigMap` is downloaded, and then point `config-dir/.cur` to the associated directory.

To begin using a new configuration, the Kubelet simply sets `config-dir/.cur`, calls `os.Exit(0)`, and relies on the process manager (e.g. `systemd`) to restart it.

The Kubelet detects new configuration by watching the `Node` object for changes to the `spec.configSource` field. When the Kubelet detects new configuration, it checkpoints it as necessary, sets `config-dir/.cur`, and restarts to begin using it.

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

The Kubelet will track its LKG configuration by directing a symlink called `.lkg`, which lives at `config-dir/.lkg`, to point at the directory associated with the LKG `ConfigMap`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. If `config-dir/init` exists, the Kubelet will initially create the symlink such that it points to `config-dir/init`. Otherwise, the Kubelet will wait until a downloaded `ConfigMap` passes the trial period required to become the LKG. If the symlink does not exist and the current configuration is determined to be bad, the Kubelet will roll back to its built-in defaults.

Any configuration retrieved from the API server must persist beyond a trial period before it can be considered LKG. This trial period will be called `ConfigTrialPeriod`, will be a `Duration` as defined by `k8s.io/apimachinery/pkg/apis/meta/v1/duration.go`, and will be a parameter of the `KubeletConfiguration`. The trial period on a given configuration is the trial period used for that configuration (as opposed to, say, using the trial period set on the previous configuration). This is useful if you have, for example, a configuration you want to roll out with a longer trial period for additional caution.

When the time since the `.cur` symlink was set to point at the new configuration exceeds the trial period, `.lkg` will be set to point to the same configuration as `.cur`.

The init configuration (`config-dir/init`) will be automatically considered good. If a node is provisioned with an init configuration, it MUST be a valid configuration. The Kubelet will always attempt to deserialize the init configuration and validate it on startup, regardless of whether a remote configuration exists. If this fails, the Kubelet will submit an appropriate message to `Node.status.conditions` and enter a crash loop. We want invalid init configurations to be extremely obvious.

This is very important, because the init configuration is the initial last-known-good configuration. If the init configuration turns out to be bad, there is nothing to fall back to. We presume a user provisions nodes with an init configuration when the Kubelet defaults are inappropriate for their use case. It would thus be inappropriate to fall back to the Kubelet defaults if the init configuration exists.

As the init configuration and the built-in defaults are automatically considered good, intentionally setting `spec.configSource` on the `Node` to its empty default will reset the last-known-good symlink. If `config-dir/init` exists, the symlink will be updated to point there. If `config-dir/init` does not exist, the symlink will be deleted, meaning the built-in defaults become the last-known-good config.

##### Rolling back to the LKG config

When a configuration correlates too strongly with a crash loop, the Kubelet will "roll-back" to its last-known-good configuration. This process involves three components:
1. The Kubelet must begin using its LKG configuration instead of its intended current configuration.
2. The Kubelet must remember which configuration was bad, so it doesn't just roll forward to that configuration again.
3. The Kubelet must report that it rolled back to LKG due to the *belief* that it had a bad configuration.

Regarding (2), when the Kubelet detects a bad configuration, it will add an entry to a file, `config-dir/.bad-configs.json`, mapping the namespace and name of the `ConfigMap` to the time at which it was determined to be a bad config, and the reason. The Kubelet will not roll forward to any of these configurations again until their entries are removed from this file. For example, the contents of this file might look like (shown here with a `reason` matching what would be reported in the `Node`'s `status.conditions`):
```
{
  "{uid}": {
    "time": "RFC 3339 formatted timestamp", 
    "reason": "failed deserialize: namespace/name.kubelet; search 'config deserialize' in Kubelet log for details"
  }
}
```

Regarding (1), the Kubelet will check the `config-dir/.bad-configs.json` file on startup. It will use the config referenced by `config-dir/.lkg` if the config referenced on the `Node` is listed in `config-dir/.bad-configs.json` and the timestamp is not outside the belief-decay period (noted below). If the api server is unavailable, `config-dir/.cur` is compared against `config-dir/.bad-configs.json` instead of the config referenced via the `Node`.

Regarding (3), the Kubelet should report via the `Node`'s status:
- That it is using LKG.
- The `ConfigMap` LKG referrs to.
- The supposedly bad `ConfigMap` that the Kubelet decided to avoid.
- The reason it thinks the `ConfigMap` is bad.

##### Tracking restart frequency against the current configuration

Every time the Kubelet starts up, it will append the RFC 3339 timestamp of the startup to `config-dir/.startups.json`. This file is a JSON list of string timestamps. The timestamps are the same format as in `config-dir/.bad-configs.json`. 

Let:
- `t0` be the current time
- `tcfg` be the modification timestamp of the `.cur` symlink
- `rt` be the num-restarts threshold
- `p` be the length of the probationary period
- `e` be the elapsed portion of the probationary period

Then `e` is the interval `[max(tcfg,t0-pmax),t0]`.

Users should be able to configure `rt` and `p`. Both parameters should be dynamically configurable, so that an operator can adjust them if bad configurations tend to cause crash loops with a frequency not exceeding their initial `rt/p`.

The number of timestamps in `config-dir/.startups.json` should be limited so that it does not experience unbounded growth. Since users can configure `rt`, the maximum number of timestamps in `config-dir/.startups.json` must be at least equal to `rt`. This is the minimum number of timestamps sufficient to detect `rt` restarts above the threshold. We will therefore cap the number of timestamps in `config-dir/.startups.json` at `rt`.

The Kubelet will count the timestamps in `config-dir/.startups.json` that fall inside the `e` interval. If this number exceeds `rt`, the current config will be recorded in `config-dir/.bad-configs.json`, and the Kubelet will exit so that config can revert to `config-dir/.lkg`. If the config pointed to by `config-dir/.cur` is already recorded in `conig-dir/.bad-configs.json` and the Kubelet is already using `config-dir/.lkg`, it will not exit.

In fact, since the Kubelet is only tracking at most `rt` timestamps, timestamps are checked on every restart, and timestamps are recorded in-order, we have met or exceeded `rt` for a given config in a period `p` if and only if the first timestamp in the file falls within the interval `e` and the file contains exactly `rt` timestamps.

##### Dead-end states

We may use imperfect indicators to detect bad configuration. Thus, the Kubelet may revert to LKG when the configuration was not actually the issue. These are two possible solutions to this problem:
- Allow the belief that a configuration is bad to decay over time. The user may adjust this period to balance time to respond against time to automatically recover from false alarms. This period would be measured against the timestamp in `config-dir/.bad-configs.json`.
- Provide a Kubelet endpoint, e.g. `/accept-node-config`, which forces the Kubelet to remove the `config-dir/.bad-configs.json` entry for the `ConfigMap` currently referenced by the `Node`'s `spec.configSource` field, and then restart to take up the `Node`'s specified config.

Additionally, imagine that one configuration prevents the Kubelet from accessing the API server to detect new configurations. There are at least a few options here:
- Correlate API server connectivity with the configuration lifetime and use it to detect bad configuration. This, of course, means that network issues may cause false alarms regarding bad config.
- Provide a Kubelet endpoint, e.g. `/mark-config-bad`, that forces the Kubelet to mark its current configuration as bad. This will cause it to roll back to LKG. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/.cur`. Thus, just after restarting, the Kubelet should detect the replacement configuration. If for some reason there is not a replacement, or the replacement cannot be detected, the Kubelet will just end up using LKG.

If the problem is *really* bad - e.g. so bad that the Kubelet isn't serving endpoints, the user should set the new configuration via the API server and then reboot the affected `Node`. The Kubelet should always check for new configuration before loading the configuration referenced by `config-dir/.cur`, in case `config-dir/.cur` referenced a really-bad-configuration and the user had to swap this out.

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
reason: "this Node's spec.configSource was empty"
status: "True"
```

No remote config specified, no local `init` config provided:
```
message: "using defaults"
reason: "this Node's spec.configSource was empty, and local init config does not exist"
status: "True"
```

If `Node.spec.configSource` is invalid:
```
message: "using last-known-good: namespace/name"
reason: "this Node's spec.configSource was invalid"
status: "False"
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

Thus, we recommend that rollout workflow consist only of creating new `ConfigMap` objects and updating the `spec.configSource` field on each `Node` to point to that new object. This results in a controlled rollout with well-defined behavior.

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
