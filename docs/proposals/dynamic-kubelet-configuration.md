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
- Making it easier to run tests with different Kubelet configurations, because tests can modify Kubelet configuration on the fly.

## Primary Goals of the Design

Kubernetes should:

- Provide a versioned object to represent the Kubelet configuration.
- Provide the ability to specify a dynamic configuration source for each node.
- Provide a way to share the same configuration source between nodes.
- Protect against bad configuration pushes.
- Recommend, but not mandate, the basics of a workflow for updating configuration.

Additionally, we should:

- Add Kubelet support for consuming configuration via a file on disk. This aids work towards deprecating flags in favor of on-disk configuration.
- Make it possible to opt-out of remote configuration as an extra layer of protection. This should probably be a flag, rather than a dynamic field, as it would otherwise be too easy to accidentally turn off dynamic config with a config push.

## Design

Two really important questions:
1. How should one organize and represent configuration in a cluster?
2. How should one orchestrate changes to that configuration?

This doc primarily focuses on (1) and the downstream (API server -> Kubelet) aspects of (2).

### Organization of the Kubelet's Configuration Type

In general, components should expose their configuration types from their own source trees. The types are currently in the alpha `componentconfig` API group, and should be broken out into the trees of their individual components. PR [#42759](https://github.com/kubernetes/kubernetes/pull/42759) reorganized the Kubelet's tree to facilitate this. PR [#44252](https://github.com/kubernetes/kubernetes/pull/44252) initiates the decomposition of the type.

Components with several same-configured instances, like the Kubelet, should be able to share configuration sources. A 1:N mapping of config-object:instances is much more efficient than requiring a config object per-instance. As one example, we removed the `HostNameOverride` and `NodeIP` fields from the configuration type because these cannot be shared between Nodes - [#40117](https://github.com/kubernetes/kubernetes/pull/40117).

Components that currently take command line flags should not just map these flags directly into their configuration types. We should, in general, think about which parameters make sense to configure dynamically, which can be shared between instances, and which are so low-level that they shouldn't really be exposed on the component's interface in the first place. Thus, the Kubelet's flags should be kept separate from configuration - [#40117](https://github.com/kubernetes/kubernetes/pull/40117).

The Kubelet's current configuration type is an unreadable monolith. We should decompose it into sub-objects for convenience of composition and management. An example grouping is in PR [#44252](https://github.com/kubernetes/kubernetes/pull/44252).

### Representing and Referencing Configuration

#### Cluster-level object
- The Kubelet's configuration information should be organized in the cluster as a structured monolith. 
  + *Structured* into sub-categories, so that it is readable.
  + *Monolithic payload*, so that all Kubelet parameters roll out to a given `Node` in unison.
  + Note that this does not mean the type itself has to be monolithic, just that the entire configuration should be contained in a single payload.
- The Kubelet's configuration should be stored in the `Data` of a `ConfigMap` object. Each value should be a `YAML` blob, and should be associated with the correct key.
  + Note that today, there is only a single `KubeletConfiguration` object (required under the `kubelet` key).
- The `ConfigMap` containing the desired configuration should be specified via the `Node` object corresponding to the Kubelet. The `Node` will have a new `spec` subfield, `configSource`, which is a new type, `NodeConfigSource` (described below).
- The name of the `ConfigMap` containing the desired configuration should be of the form (using Go regexp syntax) `^(?P<nameDash>(?P<name>[a-z0-9.\-]*){0,1}-){0,1}(?P<alg>[a-z0-9]+)-(?P<hash>[a-f0-9]+)$`, where `name` is a human-readable identifier, `nameDash` is a helper to handle the dash separator between `name` and `alg`, `alg` is the hash algorithm to use, and `hash` is the lowercase hexadecimal value of the hash, as would be formatted by Go's `%x`.
  + For now, the only supported hash algorithm is `sha256`.
  + The Kubelet will verify the downloaded `ConfigMap` by performing the below procedure and comparing the result to the hash in the name. This helps prevent the "shoot yourself in the foot" scenario detailed below in *Operational Considerations/Rollout workflow*.
  + The hash will be produced extracting the key-value pairs from the `ConfigMap`'s `Data` field into a list, sorting them in byte-alphabetic order by key, serializing this sorted list to a string of the format `key:value,key:value,...,` (trailing comma if and only if `Data` is non-empty) and using `alg` to produce the hash of the string. For example, see this reference implementation in Go:
```
const sha256Alg = "sha256"

// MapStringStringHash serializes `m` into a string of pairs, in byte-alphabetic order by key, and takes the hash using `alg`.
// Keys and values are separated with `:` and pairs are separated with `,`. If m is non-empty,
// there is a trailing comma in the pre-hash serialization. If m is empty, there is no trailing comma.
// MapStringStringHash is public because it is used as a utility in some of our tests.
func MapStringStringHash(alg string, m map[string]string) (string, error) {
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
  return hash(alg, s)
}

// kvPairs extracts the key-value pairs from `m`
func kvPairs(m map[string]string) [][]string {
  kv := make([][]string, len(m))
  i := 0
  for k, v := range m {
    kv[i] = []string{k, v}
    i++
  }
  return kv
}

// hash hashes `data` with `alg` if `alg` is supported
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


`ConfigMap` object containing just the Kubelet's configuration:
```
kind: ConfigMap
metadata:
  name: node-config-sha256-{hash-of-data}
data:
  kubelet: "{JSON blob}"
```

#### On-disk

- The Kubelet should accept a `--node-config-dir` flag that specifies a directory (e.g. `config-dir`).
- The Kubelet will use this directory to checkpoint downloaded `ConfigMap`s.
  + When the Kubelet downloads a `ConfigMap`, it will checkpoint a serialization of the `ConfigMap` object to a file at `{node-config-dir}/checkpoints/{UID}/{object-name}`.
  + We checkpoint the entire object, rather than unpacking the contents as a set of files, because the former is less complex and reduces chance for errors during the checkpoint process.
- The user can additionally pre-populate `{node-config-dir}/init` with an initial set of configuration files, to be used prior to the Kubelet being able to access the API server (including when running in standalone mode). These files will be treated as if each filename is a key in a configuration `ConfigMap`, and as if each file's contents is the value associated with the respective key.

### Orchestration of configuration

There are a lot of opinions around how to orchestrate configuration in a cluster. The following items form the basis of a robust solution:

#### Robust Kubelet behavior

To make config updates robust, the Kubelet should be able to locally and automatically recover from bad config pushes. We should strive to avoid requiring operator intervention, though this may not be possible in all scenarios.

Recovery involves:
- Checkpointing configuration on-disk, so prior versions are locally available for rollback.
- Tracking a last-known-good (LKG) configuration, which will be the rollback target if the current configuration turns out to be bad.
- Tracking bad configurations, so the Kubelet can avoid using known-bad configurations across restarts.
- Detecting whether a crash-loop correlates with a new configuration, marking these configurations bad, and rolling back to the last-known-good when this happens.

##### Finding and checkpointing intended configuration

The Kubelet finds its intended configuration by looking for the `ConfigMap` referenced via it's `Node`'s optional `spec.configSource` field. This field will be a new type:
```
type NodeConfigSource struct {
  ConfigMapRef *ObjectReference
}
```

For now, this type just contains an `ObjectReference` that points to a `ConfigMap`. The `spec.configSource` field will be of type `*NodeConfigSource`, because it is optional. 

The `spec.configSource` field can be considered "correct," "empty," or "invalid." The field is "empty" if and only if it is `nil`. The field is "correct" if and only if it is neither "empty" nor "invalid." The field is "invalid" if it fails to meet any of the following criteria:
- Exactly one subfield of `NodeConfigSource` must be non-`nil`.
- All information contained in the non-`nil` subfield must meet the requirements of that subfield.

The requirements of the `ConfigMapRef` subfield are as follows:
- All of  `ConfigMapRef.UID`, `ConfigMapRef.Namespace`, and `ConfigMapRef.Name`must be non-empty. 
- Both  `ConfigMapRef.UID` and the `ConfigMapRef.Namespace`-`ConfigMapRef.Name` pair must unambiguously refer to the same object.
- The referenced object must exist.
- The referenced object must be a `ConfigMap`.
- As noted above, the referent `ConfigMap`'s name must be of the form `^(?P<nameDash>(?P<name>[a-z0-9.\-]*){0,1}-){0,1}(?P<alg>[a-z0-9]+-(?P<hash>[a-f0-9]+)$`.

The Kubelet must have permission to read `ConfigMap`s in the namespace that contains the referenced `ConfigMap`. 

If the `spec.configSource` is empty, the Kubelet will use its `config-dir/init` configuration or built-in defaults (including values from flags that currently map to configuration) if `config-dir/init` is also absent.

If the `spec.configSource` is invalid (or if some other issue prevents syncing configuration with what is specified on the `Node`):
- If the Kubelet is in its startup sequence, it will defer to its LKG configuration and report the failure to determine the desired configuration in via a `NodeCondition` (discussed later).
- If the failure to determine desired configuration occurs as part of the configuration sync-loop operation of a live Kubelet, the failure will be reported in a `NodeCondition` (discussed later), but the Kubelet will not change its configuration. This is to prevent disrupting live Kubelets in the event of user error.

If the `spec.configSource` is correct and using `ConfigMapRef`, the Kubelet checkpoints this `ConfigMap` to `config-dir`, as specified above in the *Representing and Referencing Configuration* section.

The Kubelet sets its current configuration by directing a symlink called `.cur`, which lives at `{node-config-dir}/.cur`, to point at the directory associated with the desired configuration checkpoint. For instance, it might point to `{node-config-dir}/checkpoints/{UID}`. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. The default destination of the symlink will be `/dev/zero`, which indicates that the init configuration or built-in defaults should be treated as the current configuration.

To begin using a new configuration, the Kubelet simply sets `{node-config-dir}/.cur`, calls `os.Exit(0)`, and relies on the process manager (e.g. `systemd`) to restart it.

The Kubelet detects new configuration by watching the `Node` object for changes to the `spec.configSource` field. When the Kubelet detects new configuration, it checkpoints it as necessary, sets `{node-config-dir}/.cur`, and restarts to begin using it.

##### Metrics for Bad Configuration

These are metrics we can use to detect bad configuration. Some are perfect indicators (validation) and some are imperfect (`P(ThinkBad == ActuallyBad) < 1`).

Perfect Metrics:
- `KubeletConfiguration` cannot be deserialized
- `KubeletConfiguration` fails a validation step

Imperfect Metrics:
- Kubelet restarts occur above a frequency threshold when using a given configuration, before that configuration is out of a "trial period."

We should absolutely use the perfect metrics we have available. These immediately tell us when we have bad configuration. Adding a user-defined trial period, within which restarts above a user-defined frequency can be treated as crash-loops caused by the current configuration, adds some protection against more complex configuration mishaps.

More advanced error detection is probably best left to an out-of-band component, like the Node Problem Detector. We shouldn't go overboard attempting to make the Kubelet too smart. 

##### Tracking LKG (last-known-good) configuration

The Kubelet will track its LKG configuration by directing a symlink called `.lkg`, which lives at `{node-config-dir}/.lkg`, to point at the directory associated with the LKG configuration checkpoint. This symlink will not exist when the node is initially provisioned; the Kubelet will create it if it does not exist. The default destination of the symlink will be `/dev/zero`, which indicates that the init configuration or built-in defaults should be treated as the LKG.

Any configuration retrieved from the API server must persist beyond a trial period before it can be considered LKG. This trial period will be called `ConfigTrialDuration`, will be a `Duration` as defined by `k8s.io/apimachinery/pkg/apis/meta/v1/duration.go`, and will be a parameter of the `KubeletConfiguration`. The trial period on a given configuration is the trial period used for that configuration (as opposed to, say, using the trial period set on the previous configuration). This is useful if you have, for example, a configuration you want to roll out with a longer trial period for additional caution. 

Similarly, the number of restarts tolerated during the trial period is exposed to the user via the `CrashLoopThreshold` field of the `KubeletConfiguration`. This field has a minimum of `0` and a maximum of `10`. The maximum of `10` is an arbitrary threshold to prevent unbounded growth of the startups-tracking file (discussed later). We implicitly allow one more restart than the user-provided threshold, because one restart is necessary to begin using a new configuration.

If the Kubelet restarts after the trial period without changing `.cur`, `.lkg` will be set to point to the same configuration as `.cur`.

The init configuration (`config-dir/init`) will be automatically considered good. If a node is provisioned with an init configuration, it MUST be a valid configuration. The Kubelet will always attempt to deserialize the init configuration and validate it on startup, regardless of whether a remote configuration exists. If this fails, the Kubelet will refuse to start. Similarly, the Kubelet will refuse to start if the sum total of built-in defaults and flag values that still map to configuration is invalid. This is to make invalid node provisioning extremely obvious.

It is very important to be strict about the validity of the init and default configurations, because the init configuration is the initial last-known-good configuration. If either configuration turns out to be bad, there is nothing to fall back to. We presume a user provisions nodes with an init configuration when the Kubelet defaults are inappropriate for their use case. It would thus be inappropriate to fall back to the Kubelet defaults if the init configuration exists.

As the init configuration and the built-in defaults are automatically considered good, intentionally setting `spec.configSource` on the `Node` to its empty default will reset the last-known-good symlink to point to `/dev/zero`.

##### Rolling back to the LKG config

When a configuration correlates too strongly with a crash loop, the Kubelet will "roll-back" to its last-known-good configuration. This process involves three components:
1. The Kubelet must choose to use its LKG configuration instead of its intended current configuration.
2. The Kubelet must remember which configuration was bad, so it doesn't roll forward to that configuration again.
3. The Kubelet must report that it rolled back to LKG due to the *belief* that it had a bad configuration.

Regarding (2), when the Kubelet detects a bad configuration, it will add an entry to a file, `{node-config-dir}/.bad-configs.json`, mapping the namespace and name of the `ConfigMap` to the time at which it was determined to be a bad config and the reason it was marked bad. The Kubelet will not roll forward to any of these configurations again unless their entries are removed from this file. For example, the contents of this file might look like (shown here with a `reason` matching what would be reported in a `NodeCondition`:
```
{
  "{uid}": {
    "time": "RFC 3339 formatted timestamp", 
    "reason": "failed to validate current (UID: {UID})"
  }
}
```

Regarding (1), the Kubelet will check the `{node-config-dir}/.bad-configs.json` file on startup. It will use the config referenced by `{node-config-dir}/.lkg` if the config referenced by `{node-config-dir}/.cur` is listed in `{node-config-dir}/.bad-configs.json`.

Regarding (3), the Kubelet should report via the `Node`'s status:
- That it is using LKG.
- The configuration LKG referrs to.
- The supposedly bad configuration that the Kubelet decided to avoid.
- The reason it thinks the configuration is bad.

##### Tracking restart frequency against the current configuration

Every time the Kubelet starts up, it will append the startup time to `{node-config-dir}/.startups.json`. This file is a JSON list of string RFC3339-formatted timestamps. On Kubelet startup, if the time elapsed since the last modification to `.cur` does not exceed `ConfigTrialDuration`, the Kubelet will count the number of timestamps in this file that occur after the last modification to `.cur`. If this number exceeds the `CrashLoopThreshold`, the configuration will be marked bad and considered the cause of the crash-loop. The Kubelet will then roll back to its LKG configuration. We use "exceeds" as the trigger, because the Kubelet must be able to restart once to adopt a new configuration. 

##### Dead-end states

We may use imperfect indicators to detect bad configuration. It is possible for a crash-loop unrelated to the current configuration to cause that configuration to be marked bad. This becomes evident when the Kubelet rolls back to the LKG configuration and continues to crash. In this scenario, an out-of-band node repair is required to revive the Kubelet. Since the current configuration was not, in fact, the cause of the issue, the component in charge of node repair should also reset that belief by removing the entry for the current configuration from the `{node-config-dir}/.bad-configs.json` file.

##### Reporting Configuration Status

Succinct messages related to the state of `Node` configuration should be reported in a `NodeCondition`, in `status.conditions`. These should inform users which `ConfigMap` the Kubelet is using, and if the Kubelet has detected any issues with the configuration. The Kubelet should report this condition during startup, after attempting to validate configuration but before actually using it, so that the chance of a bad configuration inhibiting status reporting is minimized.

All `NodeCondition`s contain the fields: `lastHeartbeatTime:Time`, `lastTransitionTime:Time`, `message:string`, `reason:string`, `status:string(True|False|Unknown)`, and `type:string`.

These are some brief descriptions of how these fields should be interpreted for node-configuration related conditions:
- `lastHeartbeatTime`: The last time the Kubelet updated the condition. The Kubelet will typically do this whenever it is restarted, because that is when configuration changes occur. The Kubelet will update this on restart regardless of whether the configuration, or the condition, has changed.
- `lastTransitionTime`: The last time this condition changed. The Kubelet will not update this unless it intends to set a different condition than is currently set.
- `message`: Think of this as the "effect" of the `reason`.
- `reason`: Think of this as the "cause" of the `message`.
- `status`: `True` if the currently set configuration is considered OK, `False` if it is known not to be. `Unknown` is used when the Kubelet cannot determine the user's desired configuration.
- `type`: `ConfigOK` will always be used for these conditions.

The following list of example conditions, sans `type`, `lastHeartbeatTime`, and `lastTransitionTime`, can be used to get a feel for the relationship between `message`, `reason`, and `status`:

Config is OK:
```
message: "using current (UID: {cur-UID})"
reason: "all checks passed"
status: "True"
```

No remote config specified:
```
message: "using current (init)"
reason: "current is set to the local default, and an init config was provided"
status: "True"
```

No remote config specified, no local `init` config provided:
```
message: "using current (default)"
reason: "current is set to the local default, and no init config was provided"
status: "True"
```

If `Node.spec.configSource` is invalid during Kubelet startup:
```
message: "using last-known-good (init)"
reason: "failed to sync, desired config unclear, cause: invalid NodeConfigSource, exactly one subfield must be non-nil, but all were nil"
status: "Unknown"
```

Verification of a configuration's integrity fails:
```
message: "using last known good (UID: {lkg-UID})"
reason: "failed to verify current (UID: {cur-UID})"
status: "False"
```

Validation of a configuration fails:
```
message: "using last known good: (UID: {lkg-UID})"
reason: "failed to validate current (UID: {cur-UID})"
status: "False"
```

The same text as the `reason`, along with more details on the precise nature of the error, will be printed in the Kubelet log.

### Operational Considerations

#### Rollout workflow

Kubernetes does not have the concepts of immutable, or even undeleteable API objects. This makes it easy to "shoot yourself in the foot" by modifying or deleting a `ConfigMap`. This results in undefined behavior, because the assumption is that these `ConfigMaps` are not mutated once deployed. For example, this design includes no method for invalidating the Kubelet's local cache of configurations, so there is no concept of eventually consistent results from edits or deletes of a `ConfigMap`. You may, in such a scenario, end up with partially consistent results or no results at all.

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
