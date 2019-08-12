/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
)

const (
	// SchedulerDefaultLockObjectNamespace defines default scheduler lock object namespace ("kube-system")
	SchedulerDefaultLockObjectNamespace string = metav1.NamespaceSystem

	// SchedulerDefaultLockObjectName defines default scheduler lock object name ("kube-scheduler")
	SchedulerDefaultLockObjectName = "kube-scheduler"

	// SchedulerDefaultProviderName defines the default provider names
	SchedulerDefaultProviderName = "DefaultProvider"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeSchedulerConfiguration configures a scheduler
type KubeSchedulerConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// SchedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string `json:"schedulerName"`
	// AlgorithmSource specifies the scheduler algorithm source.
	AlgorithmSource SchedulerAlgorithmSource `json:"algorithmSource"`
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int32 `json:"hardPodAffinitySymmetricWeight"`

	// LeaderElection defines the configuration of leader election client.
	LeaderElection KubeSchedulerLeaderElectionConfiguration `json:"leaderElection"`

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection componentbaseconfigv1alpha1.ClientConnectionConfiguration `json:"clientConnection"`
	// HealthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10251
	HealthzBindAddress string `json:"healthzBindAddress"`
	// MetricsBindAddress is the IP address and port for the metrics server to
	// serve on, defaulting to 0.0.0.0:10251.
	MetricsBindAddress string `json:"metricsBindAddress"`

	// DebuggingConfiguration holds configuration for Debugging related features
	// TODO: We might wanna make this a substruct like Debugging componentbaseconfigv1alpha1.DebuggingConfiguration
	componentbaseconfigv1alpha1.DebuggingConfiguration `json:",inline"`

	// DisablePreemption disables the pod preemption feature.
	DisablePreemption bool `json:"disablePreemption"`

	// PercentageOfNodeToScore is the percentage of all nodes that once found feasible
	// for running a pod, the scheduler stops its search for more feasible nodes in
	// the cluster. This helps improve scheduler's performance. Scheduler always tries to find
	// at least "minFeasibleNodesToFind" feasible nodes no matter what the value of this flag is.
	// Example: if the cluster size is 500 nodes and the value of this flag is 30,
	// then scheduler stops finding further feasible nodes once it finds 150 feasible ones.
	// When the value is 0, default percentage (5%--50% based on the size of the cluster) of the
	// nodes will be scored.
	PercentageOfNodesToScore int32 `json:"percentageOfNodesToScore"`

	// Duration to wait for a binding operation to complete before timing out
	// Value must be non-negative integer. The value zero indicates no waiting.
	// If this value is nil, the default value will be used.
	BindTimeoutSeconds *int64 `json:"bindTimeoutSeconds"`

	// Plugins specify the set of plugins that should be enabled or disabled. Enabled plugins are the
	// ones that should be enabled in addition to the default plugins. Disabled plugins are any of the
	// default plugins that should be disabled.
	// When no enabled or disabled plugin is specified for an extension point, default plugins for
	// that extension point will be used if there is any.
	Plugins *Plugins `json:"plugins,omitempty"`

	// PluginConfig is an optional set of custom plugin arguments for each plugin.
	// Omitting config args for a plugin is equivalent to using the default config for that plugin.
	PluginConfig []PluginConfig `json:"pluginConfig,omitempty"`
}

// SchedulerAlgorithmSource is the source of a scheduler algorithm. One source
// field must be specified, and source fields are mutually exclusive.
type SchedulerAlgorithmSource struct {
	// Policy is a policy based algorithm source.
	Policy *SchedulerPolicySource `json:"policy,omitempty"`
	// Provider is the name of a scheduling algorithm provider to use.
	Provider *string `json:"provider,omitempty"`
}

// SchedulerPolicySource configures a means to obtain a scheduler Policy. One
// source field must be specified, and source fields are mutually exclusive.
type SchedulerPolicySource struct {
	// File is a file policy source.
	File *SchedulerPolicyFileSource `json:"file,omitempty"`
	// ConfigMap is a config map policy source.
	ConfigMap *SchedulerPolicyConfigMapSource `json:"configMap,omitempty"`
}

// SchedulerPolicyFileSource is a policy serialized to disk and accessed via
// path.
type SchedulerPolicyFileSource struct {
	// Path is the location of a serialized policy.
	Path string `json:"path"`
}

// SchedulerPolicyConfigMapSource is a policy serialized into a config map value
// under the SchedulerPolicyConfigMapKey key.
type SchedulerPolicyConfigMapSource struct {
	// Namespace is the namespace of the policy config map.
	Namespace string `json:"namespace"`
	// Name is the name of hte policy config map.
	Name string `json:"name"`
}

// KubeSchedulerLeaderElectionConfiguration expands LeaderElectionConfiguration
// to include scheduler specific configuration.
type KubeSchedulerLeaderElectionConfiguration struct {
	componentbaseconfigv1alpha1.LeaderElectionConfiguration `json:",inline"`
	// LockObjectNamespace defines the namespace of the lock object
	// DEPRECATED: will be removed in favor of resourceNamespace
	LockObjectNamespace string `json:"lockObjectNamespace"`
	// LockObjectName defines the lock object name
	// DEPRECATED: will be removed in favor of resourceName
	LockObjectName string `json:"lockObjectName"`
}

// Plugins include multiple extension points. When specified, the list of plugins for
// a particular extension point are the only ones enabled. If an extension point is
// omitted from the config, then the default set of plugins is used for that extension point.
// Enabled plugins are called in the order specified here, after default plugins. If they need to
// be invoked before default plugins, default plugins must be disabled and re-enabled here in desired order.
type Plugins struct {
	// QueueSort is a list of plugins that should be invoked when sorting pods in the scheduling queue.
	QueueSort *PluginSet `json:"queueSort,omitempty"`

	// PreFilter is a list of plugins that should be invoked at "PreFilter" extension point of the scheduling framework.
	PreFilter *PluginSet `json:"preFilter,omitempty"`

	// Filter is a list of plugins that should be invoked when filtering out nodes that cannot run the Pod.
	Filter *PluginSet `json:"filter,omitempty"`

	// PostFilter is a list of plugins that are invoked after filtering out infeasible nodes.
	PostFilter *PluginSet `json:"postFilter,omitempty"`

	// Score is a list of plugins that should be invoked when ranking nodes that have passed the filtering phase.
	Score *PluginSet `json:"score,omitempty"`

	// NormalizeScore is a list of plugins that should be invoked after the scoring phase to normalize scores.
	NormalizeScore *PluginSet `json:"normalizeScore,omitempty"`

	// Reserve is a list of plugins invoked when reserving a node to run the pod.
	Reserve *PluginSet `json:"reserve,omitempty"`

	// Permit is a list of plugins that control binding of a Pod. These plugins can prevent or delay binding of a Pod.
	Permit *PluginSet `json:"permit,omitempty"`

	// PreBind is a list of plugins that should be invoked before a pod is bound.
	PreBind *PluginSet `json:"preBind,omitempty"`

	// Bind is a list of plugins that should be invoked at "Bind" extension point of the scheduling framework.
	// The scheduler call these plugins in order. Scheduler skips the rest of these plugins as soon as one returns success.
	Bind *PluginSet `json:"bind,omitempty"`

	// PostBind is a list of plugins that should be invoked after a pod is successfully bound.
	PostBind *PluginSet `json:"postBind,omitempty"`

	// Unreserve is a list of plugins invoked when a pod that was previously reserved is rejected in a later phase.
	Unreserve *PluginSet `json:"unreserve,omitempty"`
}

// PluginSet specifies enabled and disabled plugins for an extension point.
// If an array is empty, missing, or nil, default plugins at that extension point will be used.
type PluginSet struct {
	// Enabled specifies plugins that should be enabled in addition to default plugins.
	// These are called after default plugins and in the same order specified here.
	Enabled []Plugin `json:"enabled,omitempty"`
	// Disabled specifies default plugins that should be disabled.
	// When all default plugins need to be disabled, an array containing only one "*" should be provided.
	Disabled []Plugin `json:"disabled,omitempty"`
}

// Plugin specifies a plugin name and its weight when applicable. Weight is used only for Score plugins.
type Plugin struct {
	// Name defines the name of plugin
	Name string `json:"name"`
	// Weight defines the weight of plugin, only used for Score plugins.
	Weight int32 `json:"weight,omitempty"`
}

// PluginConfig specifies arguments that should be passed to a plugin at the time of initialization.
// A plugin that is invoked at multiple extension points is initialized once. Args can have arbitrary structure.
// It is up to the plugin to process these Args.
type PluginConfig struct {
	// Name defines the name of plugin being configured
	Name string `json:"name"`
	// Args defines the arguments passed to the plugins at the time of initialization. Args can have arbitrary structure.
	Args runtime.Unknown `json:"args,omitempty"`
}
