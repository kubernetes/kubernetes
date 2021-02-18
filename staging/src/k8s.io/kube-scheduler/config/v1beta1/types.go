/*
Copyright 2020 The Kubernetes Authors.

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

package v1beta1

import (
	"bytes"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
	v1 "k8s.io/kube-scheduler/config/v1"
	"sigs.k8s.io/yaml"
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

	// The amount of parallelism in algorithms for scheduling Pods.
	// Must be greater than 0. Defaults to 16.
	Parallelism *int32 `json:"parallelism,omitempty"`

	// The configuration of leader election client.
	LeaderElection componentbaseconfigv1alpha1.LeaderElectionConfiguration `json:"leaderElection"`

	// This specifies the kubeconfig file and client connection settings for the scheduler
	// when communicating with the API server.
	ClientConnection componentbaseconfigv1alpha1.ClientConnectionConfiguration `json:"clientConnection"`
	// The IP address and port for the health check server to serve on,
	// defaulting to "0.0.0.0:10251".
	HealthzBindAddress *string `json:"healthzBindAddress,omitempty"`
	// The IP address and port for the metrics server to serve on,
	// defaulting to "0.0.0.0:10251".
	MetricsBindAddress *string `json:"metricsBindAddress,omitempty"`

	// The configuration for debugging related features.
	componentbaseconfigv1alpha1.DebuggingConfiguration `json:",inline"`

	// The percentage of all nodes that once found feasible for running a Pod.
	// The scheduler stops its search for more feasible nodes in the cluster.
	// This helps improve the scheduler's performance. The scheduler always tries to find
	// at least "minFeasibleNodesToFind" feasible nodes no matter what the value of
	// this field is.
	//
	// Example: if the cluster size is 500 nodes and the value of this field is 30,
	// then the scheduler stops finding further feasible nodes once it finds 150 feasible ones.
	// When the value is 0, default percentage (5%--50% based on the size of the cluster) of the
	// nodes will be scored.
	PercentageOfNodesToScore *int32 `json:"percentageOfNodesToScore,omitempty"`

	// The initial backoff for unschedulable Pods. If specified, it must be greater than 0.
	// If this value is null, the default value ("1s") will be used.
	PodInitialBackoffSeconds *int64 `json:"podInitialBackoffSeconds,omitempty"`

	// The max backoff for unschedulable Pods. If specified, it must be greater than
	// the `podInitialBackoffSeconds` value. If this value is null, the default value
	// ("10s") is used.
	PodMaxBackoffSeconds *int64 `json:"podMaxBackoffSeconds,omitempty"`

	// The scheduling profiles that kube-scheduler supports. Pods can
	// choose to be scheduled under a particular profile by setting its associated
	// `schedulerName`. Pods that don't specify any scheduler name are scheduled
	// with the "default-scheduler" profile, if present here.
	// +listType=map
	// +listMapKey=schedulerName
	Profiles []KubeSchedulerProfile `json:"profiles,omitempty"`

	// The list of scheduler Extenders, each holding the values of how to communicate
	// with the Extender. These Extenders are shared by all scheduler profiles.
	// +listType=set
	Extenders []Extender `json:"extenders,omitempty"`
}

// DecodeNestedObjects decodes plugin args for known types.
func (c *KubeSchedulerConfiguration) DecodeNestedObjects(d runtime.Decoder) error {
	for i := range c.Profiles {
		prof := &c.Profiles[i]
		for j := range prof.PluginConfig {
			err := prof.PluginConfig[j].decodeNestedObjects(d)
			if err != nil {
				return fmt.Errorf("decoding .profiles[%d].pluginConfig[%d]: %w", i, j, err)
			}
		}
	}
	return nil
}

// EncodeNestedObjects encodes plugin args.
func (c *KubeSchedulerConfiguration) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Profiles {
		prof := &c.Profiles[i]
		for j := range prof.PluginConfig {
			err := prof.PluginConfig[j].encodeNestedObjects(e)
			if err != nil {
				return fmt.Errorf("encoding .profiles[%d].pluginConfig[%d]: %w", i, j, err)
			}
		}
	}
	return nil
}

// KubeSchedulerProfile is a scheduling profile.
type KubeSchedulerProfile struct {
	// The name of the scheduler associated with this profile.
	// If `schedulerName` matches with a Pod's `spec.schedulerName`, the Pod
	// is scheduled with this profile.
	SchedulerName *string `json:"schedulerName,omitempty"`

	// The set of Plugins that should be enabled or disabled.
	// Enabled Plugins are the ones that should be enabled in addition to the
	// default Plugins. Disabled Plugins are any of the default Plugins that
	// should be disabled.
	// When no enabled or disabled Plugin is specified for an extension point,
	// default Plugins for that extension point will be used if there is any.
	// If a "QueueSort" plugin is specified, the same "QueueSort" Plugin and
	// PluginConfig must be specified for all profiles.
	Plugins *Plugins `json:"plugins,omitempty"`

	// An optional set of custom plugin arguments for each Plugin.
	// Omitting config args for a Plugin is equivalent to using the default config
	// for that Plugin.
	// +listType=map
	// +listMapKey=name
	PluginConfig []PluginConfig `json:"pluginConfig,omitempty"`
}

// Plugins include multiple extension points. When specified, the list of Plugins for
// a particular extension point are the only ones enabled. If an extension point is
// omitted from the config, then the default set of plugins are used for that extension point.
// Enabled Plugins are called in the order specified here, after default Plugins. If they need to
// be invoked before default Plugins, default Plugins must be disabled and re-enabled here
// in a desired order.
type Plugins struct {
	// A list of Plugins that should be invoked when sorting Pods in the scheduling queue.
	QueueSort *PluginSet `json:"queueSort,omitempty"`

	// A list of Plugins that should be invoked at "PreFilter"
	// extension point of the scheduling framework.
	PreFilter *PluginSet `json:"preFilter,omitempty"`

	// A list of Plugins that should be invoked when filtering out nodes that cannot run a Pod.
	Filter *PluginSet `json:"filter,omitempty"`

	// A list of Plugins that are invoked after the "Filter" phase, no matter whether
	// filtering succeeds or not.
	PostFilter *PluginSet `json:"postFilter,omitempty"`

	// A list of Plugins that are invoked before the "Score" phase.
	PreScore *PluginSet `json:"preScore,omitempty"`

	// A list of Plugins that should be invoked when ranking nodes that
	// have passed the "Filter" phase.
	Score *PluginSet `json:"score,omitempty"`

	// A list of Plugins invoked when reserving/unreserving resources after a node
	// is assigned to run a Pod.
	Reserve *PluginSet `json:"reserve,omitempty"`

	// A list of Plugins that control the binding of a Pod. These Plugins can
	// prevent or delay binding of a Pod.
	Permit *PluginSet `json:"permit,omitempty"`

	// A list of Plugins that should be invoked before a Pod is bound.
	PreBind *PluginSet `json:"preBind,omitempty"`

	// A list of Plugins that should be invoked at "Bind" extension point of
	// the scheduling framework.
	// The scheduler executes these Plugins in order until the first Plugin that
	// returns success.
	Bind *PluginSet `json:"bind,omitempty"`

	// A list of Plugins that should be invoked after a Pod is successfully bound.
	PostBind *PluginSet `json:"postBind,omitempty"`
}

// PluginSet specifies enabled and disabled Plugins for an extension point.
// If an array is empty, missing, or nil, default plugins at that extension point will be used.
type PluginSet struct {
	// The Plugins that should be enabled in addition to default Plugins.
	// These are called after default Plugins and in the same order specified here.
	// +listType=atomic
	Enabled []Plugin `json:"enabled,omitempty"`
	// The default Plugins that should be disabled.
	// In particular, an array with a single `*` element means to disable all Plugins.
	// +listType=map
	// +listMapKey=name
	Disabled []Plugin `json:"disabled,omitempty"`
}

// Plugin specifies a Plugin name and its weight when applicable.
type Plugin struct {
	// The name of the Plugin.
	Name string `json:"name"`
	// The weight of the Plugin, only used for "Score" plugins.
	Weight *int32 `json:"weight,omitempty"`
}

// PluginConfig specifies arguments that should be passed to a Plugin at the time of initialization.
// A Plugin that is invoked at multiple extension points is initialized once. Args can have arbitrary structure.
// It is up to the Plugin to process these args.
type PluginConfig struct {
	// The name of Plugin being configured.
	Name string `json:"name"`
	// The arguments passed to the Plugin at the time of initialization.
	// The `args` can have arbitrary structure.
	Args runtime.RawExtension `json:"args,omitempty"`
}

func (c *PluginConfig) decodeNestedObjects(d runtime.Decoder) error {
	gvk := SchemeGroupVersion.WithKind(c.Name + "Args")
	// dry-run to detect and skip out-of-tree plugin args.
	if _, _, err := d.Decode(nil, &gvk, nil); runtime.IsNotRegisteredError(err) {
		return nil
	}

	obj, parsedGvk, err := d.Decode(c.Args.Raw, &gvk, nil)
	if err != nil {
		return fmt.Errorf("decoding args for plugin %s: %w", c.Name, err)
	}
	if parsedGvk.GroupKind() != gvk.GroupKind() {
		return fmt.Errorf("args for plugin %s were not of type %s, got %s", c.Name, gvk.GroupKind(), parsedGvk.GroupKind())
	}
	c.Args.Object = obj
	return nil
}

func (c *PluginConfig) encodeNestedObjects(e runtime.Encoder) error {
	if c.Args.Object == nil {
		return nil
	}
	var buf bytes.Buffer
	err := e.Encode(c.Args.Object, &buf)
	if err != nil {
		return err
	}
	// The <e> encoder might be a YAML encoder, but the parent encoder expects
	// JSON output, so we convert YAML back to JSON.
	// This is a no-op if <e> produces JSON.
	json, err := yaml.YAMLToJSON(buf.Bytes())
	if err != nil {
		return err
	}
	c.Args.Raw = json
	return nil
}

// Extender holds the parameters used to communicate with the extender. If a verb is unspecified/empty,
// it is assumed that the extender chose not to provide that extension.
type Extender struct {
	// The URL prefix at which the extender is available.
	URLPrefix string `json:"urlPrefix"`
	// The verb for the `filter` call, empty if not supported. This verb is appended to the
	// `urlPrefix` when issuing the `filter` call to the Extender.
	FilterVerb string `json:"filterVerb,omitempty"`
	// The verb for the `preempt` call, empty if not supported. This verb is appended to the
	// `urlPrefix` when issuing the `preempt` call to the Extender.
	PreemptVerb string `json:"preemptVerb,omitempty"`
	// The verb for the `prioritize` call, empty if not supported. This verb is appended to
	// the `urlPrefix` when issuing the `prioritize` call to the Extender.
	PrioritizeVerb string `json:"prioritizeVerb,omitempty"`
	// The numeric multiplier for the node scores that the `prioritize` call generates.
	// The weight should be a positive integer
	Weight int64 `json:"weight,omitempty"`
	// The verb for the `bind` call, empty if not supported. This verb is appended to the
	// `urlPrefix` when issuing the `bind` call to the Extender.
	// If this method is implemented by the Extender, it is the Extender's responsibility
	// to bind the Pod to the API server. Only one Extender can implement this function.
	BindVerb string `json:"bindVerb,omitempty"`
	// This flag specifies whether HTTPS should be used to communicate with the Extender.
	EnableHTTPS bool `json:"enableHTTPS,omitempty"`
	// This specifies the transport layer security (TLS) configuration.
	TLSConfig *v1.ExtenderTLSConfig `json:"tlsConfig,omitempty"`
	// This specifies the timeout duration for a call to the Extender. Filter timeout fails
	// the scheduling of a Pod. Prioritize timeout is ignored, k8s/other extenders priorities
	// are used to select a node.
	HTTPTimeout metav1.Duration `json:"httpTimeout,omitempty"`
	// This flag indicates that the Extender is capable of caching node information,
	// so the scheduler should only send minimal information about the eligible nodes
	// assuming that the extender already cached full details of all nodes in the cluster.
	NodeCacheCapable bool `json:"nodeCacheCapable,omitempty"`
	// A list of extended resources that are managed by this Extender.
	//
	// - A Pod will be sent to the Extender on the "Filter", "Prioritize" and "Bind"
	//   (if the extender is the binder) phases if the Pod requests at least
	//   one of the extended resources in this list. If empty or unspecified,
	//   all Pods are sent to this Extender.
	// - If IgnoredByScheduler is set to true for a resource, kube-scheduler
	//   will skip checking the resource in predicates.
	// +optional
	// +listType=atomic
	ManagedResources []v1.ExtenderManagedResource `json:"managedResources,omitempty"`
	// This specifies if the Extender is ignorable, i.e. scheduling should not
	// fail when the Extender returns an error or is not reachable.
	Ignorable bool `json:"ignorable,omitempty"`
}
