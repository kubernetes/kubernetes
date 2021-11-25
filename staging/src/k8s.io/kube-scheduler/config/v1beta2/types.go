/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta2

import (
	"bytes"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfigv1alpha1 "k8s.io/component-base/config/v1alpha1"
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

	// Parallelism defines the amount of parallelism in algorithms for scheduling a Pods. Must be greater than 0. Defaults to 16
	Parallelism *int32 `json:"parallelism,omitempty"`

	// LeaderElection defines the configuration of leader election client.
	LeaderElection componentbaseconfigv1alpha1.LeaderElectionConfiguration `json:"leaderElection"`

	// ClientConnection specifies the kubeconfig file and client connection
	// settings for the proxy server to use when communicating with the apiserver.
	ClientConnection componentbaseconfigv1alpha1.ClientConnectionConfiguration `json:"clientConnection"`

	// Note: Both HealthzBindAddress and MetricsBindAddress fields are deprecated.
	// Only empty address or port 0 is allowed. Anything else will fail validation.
	// HealthzBindAddress is the IP address and port for the health check server to serve on.
	HealthzBindAddress *string `json:"healthzBindAddress,omitempty"`
	// MetricsBindAddress is the IP address and port for the metrics server to serve on.
	MetricsBindAddress *string `json:"metricsBindAddress,omitempty"`

	// DebuggingConfiguration holds configuration for Debugging related features
	// TODO: We might wanna make this a substruct like Debugging componentbaseconfigv1alpha1.DebuggingConfiguration
	componentbaseconfigv1alpha1.DebuggingConfiguration `json:",inline"`

	// PercentageOfNodesToScore is the percentage of all nodes that once found feasible
	// for running a pod, the scheduler stops its search for more feasible nodes in
	// the cluster. This helps improve scheduler's performance. Scheduler always tries to find
	// at least "minFeasibleNodesToFind" feasible nodes no matter what the value of this flag is.
	// Example: if the cluster size is 500 nodes and the value of this flag is 30,
	// then scheduler stops finding further feasible nodes once it finds 150 feasible ones.
	// When the value is 0, default percentage (5%--50% based on the size of the cluster) of the
	// nodes will be scored.
	PercentageOfNodesToScore *int32 `json:"percentageOfNodesToScore,omitempty"`

	// PodInitialBackoffSeconds is the initial backoff for unschedulable pods.
	// If specified, it must be greater than 0. If this value is null, the default value (1s)
	// will be used.
	PodInitialBackoffSeconds *int64 `json:"podInitialBackoffSeconds,omitempty"`

	// PodMaxBackoffSeconds is the max backoff for unschedulable pods.
	// If specified, it must be greater than podInitialBackoffSeconds. If this value is null,
	// the default value (10s) will be used.
	PodMaxBackoffSeconds *int64 `json:"podMaxBackoffSeconds,omitempty"`

	// Profiles are scheduling profiles that kube-scheduler supports. Pods can
	// choose to be scheduled under a particular profile by setting its associated
	// scheduler name. Pods that don't specify any scheduler name are scheduled
	// with the "default-scheduler" profile, if present here.
	// +listType=map
	// +listMapKey=schedulerName
	Profiles []KubeSchedulerProfile `json:"profiles,omitempty"`

	// Extenders are the list of scheduler extenders, each holding the values of how to communicate
	// with the extender. These extenders are shared by all scheduler profiles.
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
	// SchedulerName is the name of the scheduler associated to this profile.
	// If SchedulerName matches with the pod's "spec.schedulerName", then the pod
	// is scheduled with this profile.
	SchedulerName *string `json:"schedulerName,omitempty"`

	// Plugins specify the set of plugins that should be enabled or disabled.
	// Enabled plugins are the ones that should be enabled in addition to the
	// default plugins. Disabled plugins are any of the default plugins that
	// should be disabled.
	// When no enabled or disabled plugin is specified for an extension point,
	// default plugins for that extension point will be used if there is any.
	// If a QueueSort plugin is specified, the same QueueSort Plugin and
	// PluginConfig must be specified for all profiles.
	Plugins *Plugins `json:"plugins,omitempty"`

	// PluginConfig is an optional set of custom plugin arguments for each plugin.
	// Omitting config args for a plugin is equivalent to using the default config
	// for that plugin.
	// +listType=map
	// +listMapKey=name
	PluginConfig []PluginConfig `json:"pluginConfig,omitempty"`
}

// Plugins include multiple extension points. When specified, the list of plugins for
// a particular extension point are the only ones enabled. If an extension point is
// omitted from the config, then the default set of plugins is used for that extension point.
// Enabled plugins are called in the order specified here, after default plugins. If they need to
// be invoked before default plugins, default plugins must be disabled and re-enabled here in desired order.
type Plugins struct {
	// QueueSort is a list of plugins that should be invoked when sorting pods in the scheduling queue.
	QueueSort PluginSet `json:"queueSort,omitempty"`

	// PreFilter is a list of plugins that should be invoked at "PreFilter" extension point of the scheduling framework.
	PreFilter PluginSet `json:"preFilter,omitempty"`

	// Filter is a list of plugins that should be invoked when filtering out nodes that cannot run the Pod.
	Filter PluginSet `json:"filter,omitempty"`

	// PostFilter is a list of plugins that are invoked after filtering phase, but only when no feasible nodes were found for the pod.
	PostFilter PluginSet `json:"postFilter,omitempty"`

	// PreScore is a list of plugins that are invoked before scoring.
	PreScore PluginSet `json:"preScore,omitempty"`

	// Score is a list of plugins that should be invoked when ranking nodes that have passed the filtering phase.
	Score PluginSet `json:"score,omitempty"`

	// Reserve is a list of plugins invoked when reserving/unreserving resources
	// after a node is assigned to run the pod.
	Reserve PluginSet `json:"reserve,omitempty"`

	// Permit is a list of plugins that control binding of a Pod. These plugins can prevent or delay binding of a Pod.
	Permit PluginSet `json:"permit,omitempty"`

	// PreBind is a list of plugins that should be invoked before a pod is bound.
	PreBind PluginSet `json:"preBind,omitempty"`

	// Bind is a list of plugins that should be invoked at "Bind" extension point of the scheduling framework.
	// The scheduler call these plugins in order. Scheduler skips the rest of these plugins as soon as one returns success.
	Bind PluginSet `json:"bind,omitempty"`

	// PostBind is a list of plugins that should be invoked after a pod is successfully bound.
	PostBind PluginSet `json:"postBind,omitempty"`

	// MultiPoint is a simplified config section to enable plugins for all valid extension points.
	MultiPoint PluginSet `json:"multiPoint,omitempty"`
}

// PluginSet specifies enabled and disabled plugins for an extension point.
// If an array is empty, missing, or nil, default plugins at that extension point will be used.
type PluginSet struct {
	// Enabled specifies plugins that should be enabled in addition to default plugins.
	// If the default plugin is also configured in the scheduler config file, the weight of plugin will
	// be overridden accordingly.
	// These are called after default plugins and in the same order specified here.
	// +listType=atomic
	Enabled []Plugin `json:"enabled,omitempty"`
	// Disabled specifies default plugins that should be disabled.
	// When all default plugins need to be disabled, an array containing only one "*" should be provided.
	// +listType=map
	// +listMapKey=name
	Disabled []Plugin `json:"disabled,omitempty"`
}

// Plugin specifies a plugin name and its weight when applicable. Weight is used only for Score plugins.
type Plugin struct {
	// Name defines the name of plugin
	Name string `json:"name"`
	// Weight defines the weight of plugin, only used for Score plugins.
	Weight *int32 `json:"weight,omitempty"`
}

// PluginConfig specifies arguments that should be passed to a plugin at the time of initialization.
// A plugin that is invoked at multiple extension points is initialized once. Args can have arbitrary structure.
// It is up to the plugin to process these Args.
type PluginConfig struct {
	// Name defines the name of plugin being configured
	Name string `json:"name"`
	// Args defines the arguments passed to the plugins at the time of initialization. Args can have arbitrary structure.
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
	// URLPrefix at which the extender is available
	URLPrefix string `json:"urlPrefix"`
	// Verb for the filter call, empty if not supported. This verb is appended to the URLPrefix when issuing the filter call to extender.
	FilterVerb string `json:"filterVerb,omitempty"`
	// Verb for the preempt call, empty if not supported. This verb is appended to the URLPrefix when issuing the preempt call to extender.
	PreemptVerb string `json:"preemptVerb,omitempty"`
	// Verb for the prioritize call, empty if not supported. This verb is appended to the URLPrefix when issuing the prioritize call to extender.
	PrioritizeVerb string `json:"prioritizeVerb,omitempty"`
	// The numeric multiplier for the node scores that the prioritize call generates.
	// The weight should be a positive integer
	Weight int64 `json:"weight,omitempty"`
	// Verb for the bind call, empty if not supported. This verb is appended to the URLPrefix when issuing the bind call to extender.
	// If this method is implemented by the extender, it is the extender's responsibility to bind the pod to apiserver. Only one extender
	// can implement this function.
	BindVerb string `json:"bindVerb,omitempty"`
	// EnableHTTPS specifies whether https should be used to communicate with the extender
	EnableHTTPS bool `json:"enableHTTPS,omitempty"`
	// TLSConfig specifies the transport layer security config
	TLSConfig *ExtenderTLSConfig `json:"tlsConfig,omitempty"`
	// HTTPTimeout specifies the timeout duration for a call to the extender. Filter timeout fails the scheduling of the pod. Prioritize
	// timeout is ignored, k8s/other extenders priorities are used to select the node.
	HTTPTimeout metav1.Duration `json:"httpTimeout,omitempty"`
	// NodeCacheCapable specifies that the extender is capable of caching node information,
	// so the scheduler should only send minimal information about the eligible nodes
	// assuming that the extender already cached full details of all nodes in the cluster
	NodeCacheCapable bool `json:"nodeCacheCapable,omitempty"`
	// ManagedResources is a list of extended resources that are managed by
	// this extender.
	// - A pod will be sent to the extender on the Filter, Prioritize and Bind
	//   (if the extender is the binder) phases iff the pod requests at least
	//   one of the extended resources in this list. If empty or unspecified,
	//   all pods will be sent to this extender.
	// - If IgnoredByScheduler is set to true for a resource, kube-scheduler
	//   will skip checking the resource in predicates.
	// +optional
	// +listType=atomic
	ManagedResources []ExtenderManagedResource `json:"managedResources,omitempty"`
	// Ignorable specifies if the extender is ignorable, i.e. scheduling should not
	// fail when the extender returns an error or is not reachable.
	Ignorable bool `json:"ignorable,omitempty"`
}

// ExtenderManagedResource describes the arguments of extended resources
// managed by an extender.
type ExtenderManagedResource struct {
	// Name is the extended resource name.
	Name string `json:"name"`
	// IgnoredByScheduler indicates whether kube-scheduler should ignore this
	// resource when applying predicates.
	IgnoredByScheduler bool `json:"ignoredByScheduler,omitempty"`
}

// ExtenderTLSConfig contains settings to enable TLS with extender
type ExtenderTLSConfig struct {
	// Server should be accessed without verifying the TLS certificate. For testing only.
	Insecure bool `json:"insecure,omitempty"`
	// ServerName is passed to the server for SNI and is used in the client to check server
	// certificates against. If ServerName is empty, the hostname used to contact the
	// server is used.
	ServerName string `json:"serverName,omitempty"`

	// Server requires TLS client certificate authentication
	CertFile string `json:"certFile,omitempty"`
	// Server requires TLS client certificate authentication
	KeyFile string `json:"keyFile,omitempty"`
	// Trusted root certificates for server
	CAFile string `json:"caFile,omitempty"`

	// CertData holds PEM-encoded bytes (typically read from a client certificate file).
	// CertData takes precedence over CertFile
	CertData []byte `json:"certData,omitempty"`
	// KeyData holds PEM-encoded bytes (typically read from a client certificate key file).
	// KeyData takes precedence over KeyFile
	KeyData []byte `json:"keyData,omitempty"`
	// CAData holds PEM-encoded bytes (typically read from a root certificates bundle).
	// CAData takes precedence over CAFile
	CAData []byte `json:"caData,omitempty"`
}
