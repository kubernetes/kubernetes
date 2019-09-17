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

package config

import (
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	componentbaseconfig "k8s.io/component-base/config"
)

const (
	// SchedulerDefaultLockObjectNamespace defines default scheduler lock object namespace ("kube-system")
	SchedulerDefaultLockObjectNamespace string = metav1.NamespaceSystem

	// SchedulerDefaultLockObjectName defines default scheduler lock object name ("kube-scheduler")
	SchedulerDefaultLockObjectName = "kube-scheduler"

	// SchedulerPolicyConfigMapKey defines the key of the element in the
	// scheduler's policy ConfigMap that contains scheduler's policy config.
	SchedulerPolicyConfigMapKey = "policy.cfg"

	// SchedulerDefaultProviderName defines the default provider names
	SchedulerDefaultProviderName = "DefaultProvider"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeSchedulerConfiguration configures a scheduler
type KubeSchedulerConfiguration struct {
	metav1.TypeMeta

	// SchedulerName is name of the scheduler, used to select which pods
	// will be processed by this scheduler, based on pod's "spec.SchedulerName".
	SchedulerName string
	// AlgorithmSource specifies the scheduler algorithm source.
	AlgorithmSource SchedulerAlgorithmSource
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 0-100.
	HardPodAffinitySymmetricWeight int32

	// LeaderElection defines the configuration of leader election client.
	LeaderElection KubeSchedulerLeaderElectionConfiguration

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection componentbaseconfig.ClientConnectionConfiguration
	// HealthzBindAddress is the IP address and port for the health check server to serve on,
	// defaulting to 0.0.0.0:10251
	HealthzBindAddress string
	// MetricsBindAddress is the IP address and port for the metrics server to
	// serve on, defaulting to 0.0.0.0:10251.
	MetricsBindAddress string

	// DebuggingConfiguration holds configuration for Debugging related features
	// TODO: We might wanna make this a substruct like Debugging componentbaseconfig.DebuggingConfiguration
	componentbaseconfig.DebuggingConfiguration

	// DisablePreemption disables the pod preemption feature.
	DisablePreemption bool

	// PercentageOfNodeToScore is the percentage of all nodes that once found feasible
	// for running a pod, the scheduler stops its search for more feasible nodes in
	// the cluster. This helps improve scheduler's performance. Scheduler always tries to find
	// at least "minFeasibleNodesToFind" feasible nodes no matter what the value of this flag is.
	// Example: if the cluster size is 500 nodes and the value of this flag is 30,
	// then scheduler stops finding further feasible nodes once it finds 150 feasible ones.
	// When the value is 0, default percentage (5%--50% based on the size of the cluster) of the
	// nodes will be scored.
	PercentageOfNodesToScore int32

	// Duration to wait for a binding operation to complete before timing out
	// Value must be non-negative integer. The value zero indicates no waiting.
	// If this value is nil, the default value will be used.
	BindTimeoutSeconds *int64

	// Plugins specify the set of plugins that should be enabled or disabled. Enabled plugins are the
	// ones that should be enabled in addition to the default plugins. Disabled plugins are any of the
	// default plugins that should be disabled.
	// When no enabled or disabled plugin is specified for an extension point, default plugins for
	// that extension point will be used if there is any.
	Plugins *Plugins

	// PluginConfig is an optional set of custom plugin arguments for each plugin.
	// Omitting config args for a plugin is equivalent to using the default config for that plugin.
	PluginConfig []PluginConfig
}

// SchedulerAlgorithmSource is the source of a scheduler algorithm. One source
// field must be specified, and source fields are mutually exclusive.
type SchedulerAlgorithmSource struct {
	// Policy is a policy based algorithm source.
	Policy *SchedulerPolicySource
	// Provider is the name of a scheduling algorithm provider to use.
	Provider *string
}

// SchedulerPolicySource configures a means to obtain a scheduler Policy. One
// source field must be specified, and source fields are mutually exclusive.
type SchedulerPolicySource struct {
	// File is a file policy source.
	File *SchedulerPolicyFileSource
	// ConfigMap is a config map policy source.
	ConfigMap *SchedulerPolicyConfigMapSource
}

// SchedulerPolicyFileSource is a policy serialized to disk and accessed via
// path.
type SchedulerPolicyFileSource struct {
	// Path is the location of a serialized policy.
	Path string
}

// SchedulerPolicyConfigMapSource is a policy serialized into a config map value
// under the SchedulerPolicyConfigMapKey key.
type SchedulerPolicyConfigMapSource struct {
	// Namespace is the namespace of the policy config map.
	Namespace string
	// Name is the name of hte policy config map.
	Name string
}

// KubeSchedulerLeaderElectionConfiguration expands LeaderElectionConfiguration
// to include scheduler specific configuration.
type KubeSchedulerLeaderElectionConfiguration struct {
	componentbaseconfig.LeaderElectionConfiguration
}

// Plugins include multiple extension points. When specified, the list of plugins for
// a particular extension point are the only ones enabled. If an extension point is
// omitted from the config, then the default set of plugins is used for that extension point.
// Enabled plugins are called in the order specified here, after default plugins. If they need to
// be invoked before default plugins, default plugins must be disabled and re-enabled here in desired order.
type Plugins struct {
	// QueueSort is a list of plugins that should be invoked when sorting pods in the scheduling queue.
	QueueSort *PluginSet

	// PreFilter is a list of plugins that should be invoked at "PreFilter" extension point of the scheduling framework.
	PreFilter *PluginSet

	// Filter is a list of plugins that should be invoked when filtering out nodes that cannot run the Pod.
	Filter *PluginSet

	// PostFilter is a list of plugins that are invoked after filtering out infeasible nodes.
	PostFilter *PluginSet

	// Score is a list of plugins that should be invoked when ranking nodes that have passed the filtering phase.
	Score *PluginSet

	// Reserve is a list of plugins invoked when reserving a node to run the pod.
	Reserve *PluginSet

	// Permit is a list of plugins that control binding of a Pod. These plugins can prevent or delay binding of a Pod.
	Permit *PluginSet

	// PreBind is a list of plugins that should be invoked before a pod is bound.
	PreBind *PluginSet

	// Bind is a list of plugins that should be invoked at "Bind" extension point of the scheduling framework.
	// The scheduler call these plugins in order. Scheduler skips the rest of these plugins as soon as one returns success.
	Bind *PluginSet

	// PostBind is a list of plugins that should be invoked after a pod is successfully bound.
	PostBind *PluginSet

	// Unreserve is a list of plugins invoked when a pod that was previously reserved is rejected in a later phase.
	Unreserve *PluginSet
}

// PluginSet specifies enabled and disabled plugins for an extension point.
// If an array is empty, missing, or nil, default plugins at that extension point will be used.
type PluginSet struct {
	// Enabled specifies plugins that should be enabled in addition to default plugins.
	// These are called after default plugins and in the same order specified here.
	Enabled []Plugin
	// Disabled specifies default plugins that should be disabled.
	// When all default plugins need to be disabled, an array containing only one "*" should be provided.
	Disabled []Plugin
}

// Plugin specifies a plugin name and its weight when applicable. Weight is used only for Score plugins.
type Plugin struct {
	// Name defines the name of plugin
	Name string
	// Weight defines the weight of plugin, only used for Score plugins.
	Weight int32
}

// PluginConfig specifies arguments that should be passed to a plugin at the time of initialization.
// A plugin that is invoked at multiple extension points is initialized once. Args can have arbitrary structure.
// It is up to the plugin to process these Args.
type PluginConfig struct {
	// Name defines the name of plugin being configured
	Name string
	// Args defines the arguments passed to the plugins at the time of initialization. Args can have arbitrary structure.
	Args runtime.Unknown
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Policy describes a struct of a policy resource in api.
type Policy struct {
	metav1.TypeMeta
	// Holds the information to configure the fit predicate functions.
	// If unspecified, the default predicate functions will be applied.
	// If empty list, all predicates (except the mandatory ones) will be
	// bypassed.
	Predicates []PredicatePolicy
	// Holds the information to configure the priority functions.
	// If unspecified, the default priority functions will be applied.
	// If empty list, all priority functions will be bypassed.
	Priorities []PriorityPolicy
	// Holds the information to communicate with the extender(s)
	ExtenderConfigs []ExtenderConfig
	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// HardPodAffinitySymmetricWeight represents the weight of implicit PreferredDuringScheduling affinity rule, in the range 1-100.
	HardPodAffinitySymmetricWeight int32

	// When AlwaysCheckAllPredicates is set to true, scheduler checks all
	// the configured predicates even after one or more of them fails.
	// When the flag is set to false, scheduler skips checking the rest
	// of the predicates after it finds one predicate that failed.
	AlwaysCheckAllPredicates bool
}

// PredicatePolicy describes a struct of a predicate policy.
type PredicatePolicy struct {
	// Identifier of the predicate policy
	// For a custom predicate, the name can be user-defined
	// For the Kubernetes provided predicates, the name is the identifier of the pre-defined predicate
	Name string
	// Holds the parameters to configure the given predicate
	Argument *PredicateArgument
}

// PriorityPolicy describes a struct of a priority policy.
type PriorityPolicy struct {
	// Identifier of the priority policy
	// For a custom priority, the name can be user-defined
	// For the Kubernetes provided priority functions, the name is the identifier of the pre-defined priority function
	Name string
	// The numeric multiplier for the node scores that the priority function generates
	// The weight should be a positive integer
	Weight int64
	// Holds the parameters to configure the given priority function
	Argument *PriorityArgument
}

// PredicateArgument represents the arguments to configure predicate functions in scheduler policy configuration.
// Only one of its members may be specified
type PredicateArgument struct {
	// The predicate that provides affinity for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAffinity *ServiceAffinity
	// The predicate that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelsPresence *LabelsPresence
}

// PriorityArgument represents the arguments to configure priority functions in scheduler policy configuration.
// Only one of its members may be specified
type PriorityArgument struct {
	// The priority function that ensures a good spread (anti-affinity) for pods belonging to a service
	// It uses a label to identify nodes that belong to the same "group"
	ServiceAntiAffinity *ServiceAntiAffinity
	// The priority function that checks whether a particular node has a certain label
	// defined or not, regardless of value
	LabelPreference *LabelPreference
	// The RequestedToCapacityRatio priority function is parametrized with function shape.
	RequestedToCapacityRatioArguments *RequestedToCapacityRatioArguments
}

// ServiceAffinity holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type ServiceAffinity struct {
	// The list of labels that identify node "groups"
	// All of the labels should match for the node to be considered a fit for hosting the pod
	Labels []string
}

// LabelsPresence holds the parameters that are used to configure the corresponding predicate in scheduler policy configuration.
type LabelsPresence struct {
	// The list of labels that identify node "groups"
	// All of the labels should be either present (or absent) for the node to be considered a fit for hosting the pod
	Labels []string
	// The boolean flag that indicates whether the labels should be present or absent from the node
	Presence bool
}

// ServiceAntiAffinity holds the parameters that are used to configure the corresponding priority function
type ServiceAntiAffinity struct {
	// Used to identify node "groups"
	Label string
}

// LabelPreference holds the parameters that are used to configure the corresponding priority function
type LabelPreference struct {
	// Used to identify node "groups"
	Label string
	// This is a boolean flag
	// If true, higher priority is given to nodes that have the label
	// If false, higher priority is given to nodes that do not have the label
	Presence bool
}

// RequestedToCapacityRatioArguments holds arguments specific to RequestedToCapacityRatio priority function.
type RequestedToCapacityRatioArguments struct {
	// Array of point defining priority function shape
	UtilizationShape []UtilizationShapePoint
	Resources        []ResourceSpec
}

// UtilizationShapePoint represents single point of priority function shape
type UtilizationShapePoint struct {
	// Utilization (x axis). Valid values are 0 to 100. Fully utilized node maps to 100.
	Utilization int32
	// Score assigned to given utilization (y axis). Valid values are 0 to 10.
	Score int32
}

// ResourceSpec represents single resource for bin packing of priority RequestedToCapacityRatioArguments.
type ResourceSpec struct {
	// Name of the resource to be managed by RequestedToCapacityRatio function.
	Name v1.ResourceName
	// Weight of the resource.
	Weight int64
}

// ExtenderManagedResource describes the arguments of extended resources
// managed by an extender.
type ExtenderManagedResource struct {
	// Name is the extended resource name.
	Name v1.ResourceName
	// IgnoredByScheduler indicates whether kube-scheduler should ignore this
	// resource when applying predicates.
	IgnoredByScheduler bool
}

// ExtenderTLSConfig contains settings to enable TLS with extender
type ExtenderTLSConfig struct {
	// Server should be accessed without verifying the TLS certificate. For testing only.
	Insecure bool
	// ServerName is passed to the server for SNI and is used in the client to check server
	// certificates against. If ServerName is empty, the hostname used to contact the
	// server is used.
	ServerName string

	// Server requires TLS client certificate authentication
	CertFile string
	// Server requires TLS client certificate authentication
	KeyFile string
	// Trusted root certificates for server
	CAFile string

	// CertData holds PEM-encoded bytes (typically read from a client certificate file).
	// CertData takes precedence over CertFile
	CertData []byte
	// KeyData holds PEM-encoded bytes (typically read from a client certificate key file).
	// KeyData takes precedence over KeyFile
	KeyData []byte
	// CAData holds PEM-encoded bytes (typically read from a root certificates bundle).
	// CAData takes precedence over CAFile
	CAData []byte
}

// ExtenderConfig holds the parameters used to communicate with the extender. If a verb is unspecified/empty,
// it is assumed that the extender chose not to provide that extension.
type ExtenderConfig struct {
	// URLPrefix at which the extender is available
	URLPrefix string
	// Verb for the filter call, empty if not supported. This verb is appended to the URLPrefix when issuing the filter call to extender.
	FilterVerb string
	// Verb for the preempt call, empty if not supported. This verb is appended to the URLPrefix when issuing the preempt call to extender.
	PreemptVerb string
	// Verb for the prioritize call, empty if not supported. This verb is appended to the URLPrefix when issuing the prioritize call to extender.
	PrioritizeVerb string
	// The numeric multiplier for the node scores that the prioritize call generates.
	// The weight should be a positive integer
	Weight int64
	// Verb for the bind call, empty if not supported. This verb is appended to the URLPrefix when issuing the bind call to extender.
	// If this method is implemented by the extender, it is the extender's responsibility to bind the pod to apiserver. Only one extender
	// can implement this function.
	BindVerb string
	// EnableHTTPS specifies whether https should be used to communicate with the extender
	EnableHTTPS bool
	// TLSConfig specifies the transport layer security config
	TLSConfig *ExtenderTLSConfig
	// HTTPTimeout specifies the timeout duration for a call to the extender. Filter timeout fails the scheduling of the pod. Prioritize
	// timeout is ignored, k8s/other extenders priorities are used to select the node.
	HTTPTimeout time.Duration
	// NodeCacheCapable specifies that the extender is capable of caching node information,
	// so the scheduler should only send minimal information about the eligible nodes
	// assuming that the extender already cached full details of all nodes in the cluster
	NodeCacheCapable bool
	// ManagedResources is a list of extended resources that are managed by
	// this extender.
	// - A pod will be sent to the extender on the Filter, Prioritize and Bind
	//   (if the extender is the binder) phases iff the pod requests at least
	//   one of the extended resources in this list. If empty or unspecified,
	//   all pods will be sent to this extender.
	// - If IgnoredByScheduler is set to true for a resource, kube-scheduler
	//   will skip checking the resource in predicates.
	// +optional
	ManagedResources []ExtenderManagedResource
	// Ignorable specifies if the extender is ignorable, i.e. scheduling should not
	// fail when the extender returns an error or is not reachable.
	Ignorable bool
}

// ExtenderPreemptionResult represents the result returned by preemption phase of extender.
type ExtenderPreemptionResult struct {
	NodeNameToMetaVictims map[string]*MetaVictims
}

// ExtenderPreemptionArgs represents the arguments needed by the extender to preempt pods on nodes.
type ExtenderPreemptionArgs struct {
	// Pod being scheduled
	Pod *v1.Pod
	// Victims map generated by scheduler preemption phase
	// Only set NodeNameToMetaVictims if ExtenderConfig.NodeCacheCapable == true. Otherwise, only set NodeNameToVictims.
	NodeNameToVictims     map[string]*Victims
	NodeNameToMetaVictims map[string]*MetaVictims
}

// Victims represents:
//   pods:  a group of pods expected to be preempted.
//   numPDBViolations: the count of violations of PodDisruptionBudget
type Victims struct {
	Pods             []*v1.Pod
	NumPDBViolations int64
}

// MetaPod represent identifier for a v1.Pod
type MetaPod struct {
	UID string
}

// MetaVictims represents:
//   pods:  a group of pods expected to be preempted.
//     Only Pod identifiers will be sent and user are expect to get v1.Pod in their own way.
//   numPDBViolations: the count of violations of PodDisruptionBudget
type MetaVictims struct {
	Pods             []*MetaPod
	NumPDBViolations int64
}

// ExtenderArgs represents the arguments needed by the extender to filter/prioritize
// nodes for a pod.
type ExtenderArgs struct {
	// Pod being scheduled
	Pod *v1.Pod
	// List of candidate nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == false
	Nodes *v1.NodeList
	// List of candidate node names where the pod can be scheduled; to be
	// populated only if ExtenderConfig.NodeCacheCapable == true
	NodeNames *[]string
}

// FailedNodesMap represents the filtered out nodes, with node names and failure messages
type FailedNodesMap map[string]string

// ExtenderFilterResult represents the results of a filter call to an extender
type ExtenderFilterResult struct {
	// Filtered set of nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == false
	Nodes *v1.NodeList
	// Filtered set of nodes where the pod can be scheduled; to be populated
	// only if ExtenderConfig.NodeCacheCapable == true
	NodeNames *[]string
	// Filtered out nodes where the pod can't be scheduled and the failure messages
	FailedNodes FailedNodesMap
	// Error message indicating failure
	Error string
}

// ExtenderBindingArgs represents the arguments to an extender for binding a pod to a node.
type ExtenderBindingArgs struct {
	// PodName is the name of the pod being bound
	PodName string
	// PodNamespace is the namespace of the pod being bound
	PodNamespace string
	// PodUID is the UID of the pod being bound
	PodUID types.UID
	// Node selected by the scheduler
	Node string
}

// ExtenderBindingResult represents the result of binding of a pod to a node from an extender.
type ExtenderBindingResult struct {
	// Error message indicating failure
	Error string
}

// HostPriority represents the priority of scheduling to a particular host, higher priority is better.
type HostPriority struct {
	// Name of the host
	Host string
	// Score associated with the host
	Score int64
}

// HostPriorityList declares a []HostPriority type.
type HostPriorityList []HostPriority

func (h HostPriorityList) Len() int {
	return len(h)
}

func (h HostPriorityList) Less(i, j int) bool {
	if h[i].Score == h[j].Score {
		return h[i].Host < h[j].Host
	}
	return h[i].Score < h[j].Score
}

func (h HostPriorityList) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
