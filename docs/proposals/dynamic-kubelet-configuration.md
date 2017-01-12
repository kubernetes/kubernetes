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

- We should remove the `HostNameOverride` and `NodeIP` fields from the KubeletConfiguration API object; these should just stay flags for now - They likely do not need to change after node provisioning and keeping them in the configuration struct complicates sharing config objects between nodes (because these values are always node-unique).
- The Kubelet's configuration type will no longer align with it's flags; we should add a separate struct that contains the flag variables to prevent them from migrating all over the codebase (similar to work being done in [#32215](https://github.com/kubernetes/kubernetes/issues/32215)).
- We should add more structure to the Kubelet configuration for readability, and I would like to move the location of the type closer to the Kubelet in the K8s tree. The details of this portion can be discussed via a separate refactoring PR (**TODO** add a link here once I open that PR).
- We need to be able to add and remove experimental fields from the `KubeletConfiguration` without having to rev the API version. A simple solution is to just have a string representation of the experimental fields as part of the KubeletConfiguration, that the Kubelet can parse as necessary. 
    + Some additional thought needs to go toward the graduation policy for experimental fields. Will the kubelet continue to try to parse them out of the experimental section? For how long? What if the first-class field and experimental field conflict?


### Representation and Organization of Kubelet Configuration in a Cluster

- As a cluster-level object, Kubelet's configuration should be stored in a `ConfigMap` object.
- On local disk, the Kubelet's configuration should be stored in a `.json` or `.yaml` file.
- The Kubelet's configuration should be, at least initially, organized in the cluster as a structured monolith. 
    + *Structured*, so that it is readable.
    + *Monolithic*, to provide atomicity over the entire configuration object.
        * This likely means that the Kubelet configuration will be stored as a string blob (JSON or YAML) in the value associated with a given key on a `ConfigMap`.
        * If leaky boundaries occur between the substructures, we don't want the problem of coordinating non-atomic updates across separately-referenced substructures.
        * If, in the future, the ability to independently orchestrate the substructures of the configuration is desired, we can move down that road. But today this is probably overkill, because most K8s cluster configuration is eventually homogeneous anyway. Even in a non-homogeneously configured cluster, we would have to carefully consider the downside of losing atomicity on the config object against the upside of more flexibility for splitting up configuration responsibility and e.g. being able to roll out changes to separate subcomponents at different rates.

An example of what the `ConfigMap` object for just the Kubelet could look like:
```
kind: ConfigMap
metadata:
  name: my-kubelet-config-<hash of `data` for verification>
data:
 config: "{JSON representation of config, for example}"
```

If this idea extended so that all `Node` config is packaged together:
```
kind: ConfigMap
metadata:
  name: my-node-config-<hash of `data` for verification>
data:
 kubelet: "{JSON representation of config, for example}"
 kube-proxy: "{JSON representation of config, for example}"
```

### Referencing Configuration

How do you tell the kubelet what to use?

- On disk: The configuration should be specified via a path passed to a command-line flag. This should be an absolute path.
- Cluster level object: The `ConfigMap` containing the desired configuration should be specified via the `Node` object corresponding to the Kubelet. Eventually, the `Node` may even have a `NodeConfig` field, but we ought to start by using an annotation, because additions to GA APIs are hard to change later if we get them wrong.

### Orchestration of configuration

There are a lot of opinions around how to orchestrate configuration in a cluster. The following items start to form the basis of a robust solution:

#### Robust Kubelet behavior

To make config updates robust, the Kubelet should:

- track the last-known-good version of config that it has been using
- track the number/frequency of kubelet restarts to detect crash-loops caused by new config, and fall back to the last-known-good config if necessary.

##### Kubelet Configuration Uptake

In the alpha implementation of this feature, the Kubelet checks for relevant dynamic config during its bootstrapping phase. If it finds it, it downloads it and uses it. During runtime, the Kubelet periodically checks for changes to its configuration source. If it finds a difference from its previous dynamic config, it exits, relying on a process manager like systemd to bring the Kubelet process back up, at which point it will again check for relevant dynamic config.

This model of the Kubelet restarting to uptake new config aids Kubelet robustness, because the Kubelet's internal environment is always freshly constructed for new config.

#### Recommendations regarding update workflow

Kubernetes does not have the concepts of immutable, or even undeleteable API objects. This makes it easy to shoot yourself in the foot by modifying or deleting a `ConfigMap` referenced by multiple nodes, potentially causing dangerous immediate changes in aggregate. To protect against this, rollout workflow should consist of creating a new `ConfigMap` and updating the reference on each node to point to that new object.

## Additional concerns not-yet-addressed

### Monitoring configuration status

- A way to query/monitor the config in-use on a given node. Today this is possible via the configz endpoint, but there are a number of other potential solutions, e.g. exposing live config via Prometheus.
- Once the Kubelet tracks restarts to detect bad configuration, it would be useful to report, e.g. via `NodeStatus`, that the Kubelet thinks it is in a bad configuration state.

### Orchestration
- A specific orchestration solution for rolling out kubelet configuration. It may be enough to extend `DaemonSet` deployments so they can manage `Node` specs. We may need something more. There are several factors to think about, including these general rollout considerations:
    + Selecting the objects that the new config should be rolled out to (Today it is probably okay to roll out `Node` config with the intention that the nodes are eventually homogeneously configured across the cluster. But what if a user intentionally wants different sets of nodes to be configured differently? A cluster running multiple separate applications, for example.).
    + Specifying the desired end-state of the rollout.
    + Specifying the rate at which to roll out.
    + Detecting problems with the rollout and automatically stopping bad rollouts.
    + Specifying how risk-averse to be when deciding to stop a rollout.
    + Recording a history of rollouts so that it is possible to roll back to previous versions.
    + Properly handling objects that are added or removed while a configuration rollout is in process. For example, `Nodes` might be added or removed due to an autoscaling feature, etc.
    + Reconciling configuration with objects added after the completion of a rollout, e.g. new `Nodes`.
    + Pausing/resuming a rollout.

### Role-Based Access Control for Configuration
A side-effect of this design is that anyone who wants to read e.g. the `Node` `ConfigMap` has to read it directly, and therefore have the appropriate RBAC permissions. This complicates the concept of `Node` config for things that run in `Pod`s, e.g. `KubeProxy`, which will run as a `DaemonSet`.

### Messy edge cases around the Kubelet's perception of "bad" config
- How would we detect that the new config is the source of the crash loop to prevent an unrelated problem from unexpectedly reverting a kubelet's config? 
    + The simple answer might be that we don't distinguish between these without operator intervention.
- How do you retake control of a Kubelet that is ignoring the config it has been told to use because it thinks it is bad? 
    + In the case that you actually pushed a bad config, push a new config (since the Kubelet is looking at an API object to determine what to use, it can detect the change).
    + In the case that you pushed a good config that the kubelet thinks is bad because something else caused a crash loop, I'm not sure what the best solution is. Maybe we provide a way to poke a Kubelet to reset it's perception that a given config is bad.
- What should I do if I push a bad config that is so bad prevents Kubelets from being able to notice when I revert the change? 
    + If it's bad enough of a config that it causes the Kubelet to enter a crash loop, then the Kubelet should fall back to it's last-known-good config.
    + If it's not that bad but still bad enough to prevent further config uptake, then maybe there needs to be a way to do a manual intervention. I shiver at the thought of turning off dynamic config via dynamic config and not being able to easily turn it back on.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/dynamic-kubelet-settings.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
