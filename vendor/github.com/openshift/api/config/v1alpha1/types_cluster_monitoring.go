/*
Copyright 2024.

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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterMonitoring is the Custom Resource object which holds the current status of Cluster Monitoring Operator. CMO is a central component of the monitoring stack.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:internal
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1929
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=clustermonitorings,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations="description=Cluster Monitoring Operators configuration API"
// +openshift:enable:FeatureGate=ClusterMonitoringConfig
// ClusterMonitoring is the Schema for the Cluster Monitoring Operators API
type ClusterMonitoring struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user configuration for the Cluster Monitoring Operator
	// +required
	Spec ClusterMonitoringSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ClusterMonitoringStatus `json:"status,omitempty"`
}

// ClusterMonitoringStatus defines the observed state of ClusterMonitoring
type ClusterMonitoringStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:internal
type ClusterMonitoringList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of ClusterMonitoring
	// +optional
	Items []ClusterMonitoring `json:"items"`
}

// ClusterMonitoringSpec defines the desired state of Cluster Monitoring Operator
// +kubebuilder:validation:MinProperties=1
type ClusterMonitoringSpec struct {
	// userDefined set the deployment mode for user-defined monitoring in addition to the default platform monitoring.
	// userDefined is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default value is `Disabled`.
	// +optional
	UserDefined UserDefinedMonitoring `json:"userDefined,omitempty,omitzero"`
	// alertmanagerConfig allows users to configure how the default Alertmanager instance
	// should be deployed in the `openshift-monitoring` namespace.
	// alertmanagerConfig is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	// The current default value is `DefaultConfig`.
	// +optional
	AlertmanagerConfig AlertmanagerConfig `json:"alertmanagerConfig,omitempty,omitzero"`
	// prometheusConfig provides configuration options for the default platform Prometheus instance
	// that runs in the `openshift-monitoring` namespace. This configuration applies only to the
	// platform Prometheus instance; user-workload Prometheus instances are configured separately.
	//
	// This field allows you to customize how the platform Prometheus is deployed and operated, including:
	//   - Pod scheduling (node selectors, tolerations, topology spread constraints)
	//   - Resource allocation (CPU, memory requests/limits)
	//   - Retention policies (how long metrics are stored)
	//   - External integrations (remote write, additional alertmanagers)
	//
	// This field is optional. When omitted, the platform chooses reasonable defaults, which may change over time.
	// +optional
	PrometheusConfig PrometheusConfig `json:"prometheusConfig,omitempty,omitzero"`
	// metricsServerConfig is an optional field that can be used to configure the Kubernetes Metrics Server that runs in the openshift-monitoring namespace.
	// Specifically, it can configure how the Metrics Server instance is deployed, pod scheduling, its audit policy and log verbosity.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	MetricsServerConfig MetricsServerConfig `json:"metricsServerConfig,omitempty,omitzero"`
	// prometheusOperatorConfig is an optional field that can be used to configure the Prometheus Operator component.
	// Specifically, it can configure how the Prometheus Operator instance is deployed, pod scheduling, and resource allocation.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	PrometheusOperatorConfig PrometheusOperatorConfig `json:"prometheusOperatorConfig,omitempty,omitzero"`
	// prometheusOperatorAdmissionWebhookConfig is an optional field that can be used to configure the
	// admission webhook component of Prometheus Operator that runs in the openshift-monitoring namespace.
	// The admission webhook validates PrometheusRule and AlertmanagerConfig objects to ensure they are
	// semantically valid, mutates PrometheusRule annotations, and converts AlertmanagerConfig objects
	// between API versions.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	PrometheusOperatorAdmissionWebhookConfig PrometheusOperatorAdmissionWebhookConfig `json:"prometheusOperatorAdmissionWebhookConfig,omitempty,omitzero"`
	// openShiftStateMetricsConfig is an optional field that can be used to configure the openshift-state-metrics
	// agent that runs in the openshift-monitoring namespace. The openshift-state-metrics agent generates metrics
	// about the state of OpenShift-specific Kubernetes objects, such as routes, builds, and deployments.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	OpenShiftStateMetricsConfig OpenShiftStateMetricsConfig `json:"openShiftStateMetricsConfig,omitempty,omitzero"`
	// telemeterClientConfig is an optional field that can be used to configure the Telemeter Client
	// component that runs in the openshift-monitoring namespace. The Telemeter Client collects
	// selected monitoring metrics and forwards them to Red Hat for telemetry purposes.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// When set, at least one field must be specified within telemeterClientConfig.
	// +optional
	TelemeterClientConfig TelemeterClientConfig `json:"telemeterClientConfig,omitempty,omitzero"`
	// thanosQuerierConfig is an optional field that can be used to configure the Thanos Querier
	// component that runs in the openshift-monitoring namespace. The Thanos Querier provides
	// a global query view by aggregating and deduplicating metrics from multiple Prometheus instances.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default deploys the Thanos Querier on linux nodes with 5m CPU and 12Mi memory
	// requests, and no custom tolerations or topology spread constraints.
	// When set, at least one field must be specified within thanosQuerierConfig.
	// +optional
	ThanosQuerierConfig ThanosQuerierConfig `json:"thanosQuerierConfig,omitempty,omitzero"`
	// nodeExporterConfig is an optional field that can be used to configure the node-exporter agent
	// that runs as a DaemonSet in the openshift-monitoring namespace. The node-exporter agent collects
	// hardware and OS-level metrics from every node in the cluster.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	NodeExporterConfig NodeExporterConfig `json:"nodeExporterConfig,omitempty,omitzero"`
	// monitoringPluginConfig is an optional field that can be used to configure the monitoring plugin
	// that runs as a dynamic plugin of the OpenShift web console. The monitoring plugin provides
	// the monitoring UI in the OpenShift web console for visualizing metrics, alerts, and dashboards.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default deploys the monitoring-plugin as a single-replica Deployment
	// on linux nodes with 10m CPU and 50Mi memory requests, and no custom tolerations
	// or topology spread constraints.
	// When set, at least one field must be specified within monitoringPluginConfig.
	// +optional
	MonitoringPluginConfig MonitoringPluginConfig `json:"monitoringPluginConfig,omitempty,omitzero"`
}

// OpenShiftStateMetricsConfig provides configuration options for the openshift-state-metrics agent
// that runs in the `openshift-monitoring` namespace. The openshift-state-metrics agent generates
// metrics about the state of OpenShift-specific Kubernetes objects, such as routes, builds, and deployments.
// +kubebuilder:validation:MinProperties=1
type OpenShiftStateMetricsConfig struct {
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the openshift-state-metrics container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 1m
	//      limit: null
	//    - name: memory
	//      request: 32Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how openshift-state-metrics Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// NodeExporterConfig provides configuration options for the node-exporter agent
// that runs as a DaemonSet in the `openshift-monitoring` namespace. The node-exporter agent collects
// hardware and OS-level metrics from every node in the cluster, including CPU, memory, disk, and
// network statistics.
// At least one field must be specified.
// +kubebuilder:validation:MinProperties=1
type NodeExporterConfig struct {
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the node-exporter container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 8m
	//      limit: null
	//    - name: memory
	//      request: 32Mi
	//      limit: null
	// ---
	// maxItems is set to 5 to stay within the Kubernetes CRD CEL validation cost budget.
	// See the MaxItems comment near the ContainerResource type definition for details.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is to tolerate all taints (operator: Exists without any key),
	// which is typical for DaemonSets that must run on every node.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// collectors configures which node-exporter metric collectors are enabled.
	// collectors is optional.
	// Each collector can be individually enabled or disabled. Some collectors may have
	// additional configuration options.
	//
	// When omitted, this means no opinion and the platform is left to choose a reasonable
	// default, which is subject to change over time.
	// +optional
	Collectors NodeExporterCollectorConfig `json:"collectors,omitempty,omitzero"`
	// maxProcs sets the target number of CPUs on which the node-exporter process will run.
	// maxProcs is optional.
	// Use this setting to override the default value, which is set either to 4 or to the number
	// of CPUs on the host, whichever is smaller.
	// The default value is computed at runtime and set via the GOMAXPROCS environment variable before
	// node-exporter is launched.
	// If a kernel deadlock occurs or if performance degrades when reading from sysfs concurrently,
	// you can change this value to 1, which limits node-exporter to running on one CPU.
	// For nodes with a high CPU count, setting the limit to a low number saves resources by preventing
	// Go routines from being scheduled to run on all CPUs. However, I/O performance degrades if the
	// maxProcs value is set too low and there are many metrics to collect.
	// The minimum value is 1 and the maximum value is 1024.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is min(4, number of host CPUs).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1024
	MaxProcs int32 `json:"maxProcs,omitempty"`
	// ignoredNetworkDevices is a list of regular expression patterns that match network devices
	// to be excluded from the relevant collector configuration such as netdev, netclass, and ethtool.
	// ignoredNetworkDevices is optional.
	//
	// When omitted, the Cluster Monitoring Operator uses a predefined list of devices to be excluded
	// to minimize the impact on memory usage.
	// When set as an empty list, no devices are excluded.
	// If you modify this setting, monitor the prometheus-k8s deployment closely for excessive memory usage.
	// Maximum length for this list is 50.
	// Each entry must be at least 1 character and at most 1024 characters long.
	// +kubebuilder:validation:MaxItems=50
	// +kubebuilder:validation:MinItems=0
	// +listType=set
	// +optional
	IgnoredNetworkDevices *[]NodeExporterIgnoredNetworkDevice `json:"ignoredNetworkDevices,omitempty"`
}

// NodeExporterIgnoredNetworkDevice is a string that is interpreted as a Go regular expression
// pattern by the controller to match network device names to exclude from node-exporter
// metric collection for collectors such as netdev, netclass, and ethtool.
// Invalid regular expressions will cause a controller-level error at runtime.
// Must be at least 1 character and at most 1024 characters.
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=1024
type NodeExporterIgnoredNetworkDevice string

// NodeExporterCollectorCollectionPolicy declares whether a node-exporter collector should collect metrics.
// Valid values are "Collect" and "DoNotCollect".
// +kubebuilder:validation:Enum=Collect;DoNotCollect
// +enum
type NodeExporterCollectorCollectionPolicy string

const (
	// NodeExporterCollectorCollectionPolicyCollect means the collector is active and will produce metrics.
	NodeExporterCollectorCollectionPolicyCollect NodeExporterCollectorCollectionPolicy = "Collect"
	// NodeExporterCollectorCollectionPolicyDoNotCollect means the collector is inactive and will not produce metrics.
	NodeExporterCollectorCollectionPolicyDoNotCollect NodeExporterCollectorCollectionPolicy = "DoNotCollect"
)

// NodeExporterNetclassStatsGatherer identifies how the netclass collector gathers device statistics
// (for example via sysfs or netlink, as implemented in node_exporter).
// Valid values are "Sysfs" and "Netlink".
// +kubebuilder:validation:Enum=Sysfs;Netlink
// +enum
type NodeExporterNetclassStatsGatherer string

const (
	// NodeExporterNetclassStatsGathererSysfs uses the sysfs-based implementation.
	NodeExporterNetclassStatsGathererSysfs NodeExporterNetclassStatsGatherer = "Sysfs"
	// NodeExporterNetclassStatsGathererNetlink uses the netlink-based implementation.
	NodeExporterNetclassStatsGathererNetlink NodeExporterNetclassStatsGatherer = "Netlink"
)

// NodeExporterCollectorConfig defines settings for individual collectors
// of the node-exporter agent. Each collector can be individually set to collect or not collect metrics.
// At least one collector must be specified.
// +kubebuilder:validation:MinProperties=1
type NodeExporterCollectorConfig struct {
	// cpuFreq configures the cpufreq collector, which collects CPU frequency statistics.
	// cpuFreq is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Consider enabling when you need to observe CPU frequency scaling; expect higher CPU usage on
	// many-core nodes when collectionPolicy is Collect.
	// +optional
	CpuFreq NodeExporterCollectorCpufreqConfig `json:"cpuFreq,omitempty,omitzero"`
	// tcpStat configures the tcpstat collector, which collects TCP connection statistics.
	// tcpStat is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable when debugging TCP connection behavior or capacity at the node level.
	// +optional
	TcpStat NodeExporterCollectorTcpStatConfig `json:"tcpStat,omitempty,omitzero"`
	// ethtool configures the ethtool collector, which collects ethernet device statistics.
	// ethtool is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable when you need NIC driver-level ethtool metrics beyond generic netdev counters.
	// +optional
	Ethtool NodeExporterCollectorEthtoolConfig `json:"ethtool,omitempty,omitzero"`
	// netDev configures the netdev collector, which collects network device statistics.
	// netDev is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is enabled.
	// Turn off if you must reduce per-interface metric cardinality on hosts with many virtual interfaces.
	// +optional
	NetDev NodeExporterCollectorNetDevConfig `json:"netDev,omitempty,omitzero"`
	// netClass configures the netclass collector, which collects information about network devices.
	// netClass is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is enabled with netlink mode active.
	// Use statsGatherer when sysfs vs netlink implementation matters or when matching node_exporter tuning.
	// +optional
	NetClass NodeExporterCollectorNetClassConfig `json:"netClass,omitempty,omitzero"`
	// buddyInfo configures the buddyinfo collector, which collects statistics about memory
	// fragmentation from the node_buddyinfo_blocks metric. This metric collects data from /proc/buddyinfo.
	// buddyInfo is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable when investigating kernel memory fragmentation; typically for advanced troubleshooting only.
	// +optional
	BuddyInfo NodeExporterCollectorBuddyInfoConfig `json:"buddyInfo,omitempty,omitzero"`
	// mountStats configures the mountstats collector, which collects statistics about NFS volume
	// I/O activities.
	// mountStats is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enabling this collector may produce metrics with high cardinality. If you enable this
	// collector, closely monitor the prometheus-k8s deployment for excessive memory usage.
	// Enable when you care about per-mount NFS client statistics.
	// +optional
	MountStats NodeExporterCollectorMountStatsConfig `json:"mountStats,omitempty,omitzero"`
	// ksmd configures the ksmd collector, which collects statistics from the kernel same-page
	// merger daemon.
	// ksmd is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable on nodes where KSM is in use and you want visibility into merging activity.
	// +optional
	Ksmd NodeExporterCollectorKSMDConfig `json:"ksmd,omitempty,omitzero"`
	// processes configures the processes collector, which collects statistics from processes and
	// threads running in the system.
	// processes is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable for process/thread-level insight; can be expensive on busy nodes.
	// +optional
	Processes NodeExporterCollectorProcessesConfig `json:"processes,omitempty,omitzero"`
	// systemd configures the systemd collector, which collects statistics on the systemd daemon
	// and its managed services.
	// systemd is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enabling this collector with a long list of selected units may produce metrics with high
	// cardinality. If you enable this collector, closely monitor the prometheus-k8s deployment
	// for excessive memory usage.
	// Enable when you need metrics for specific units; scope units carefully.
	// +optional
	Systemd NodeExporterCollectorSystemdConfig `json:"systemd,omitempty,omitzero"`
	// softirqs configures the softirqs collector, which exposes detailed softirq statistics
	// from /proc/softirqs.
	// softirqs is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is disabled.
	// Enable when you need visibility into kernel softirq processing across CPUs.
	// +optional
	Softirqs NodeExporterCollectorSoftirqsConfig `json:"softirqs,omitempty,omitzero"`
}

// NodeExporterCollectorCpufreqConfig provides configuration for the cpufreq collector
// of the node-exporter agent. The cpufreq collector collects CPU frequency statistics.
// It is disabled by default.
type NodeExporterCollectorCpufreqConfig struct {
	// collectionPolicy declares whether the cpufreq collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the cpufreq collector is active and CPU frequency statistics are collected.
	// When set to "DoNotCollect", the cpufreq collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorTcpStatConfig provides configuration for the tcpstat collector
// of the node-exporter agent. The tcpstat collector collects TCP connection statistics.
// It is disabled by default.
type NodeExporterCollectorTcpStatConfig struct {
	// collectionPolicy declares whether the tcpstat collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the tcpstat collector is active and TCP connection statistics are collected.
	// When set to "DoNotCollect", the tcpstat collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorEthtoolConfig provides configuration for the ethtool collector
// of the node-exporter agent. The ethtool collector collects ethernet device statistics.
// It is disabled by default.
type NodeExporterCollectorEthtoolConfig struct {
	// collectionPolicy declares whether the ethtool collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the ethtool collector is active and ethernet device statistics are collected.
	// When set to "DoNotCollect", the ethtool collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorNetDevConfig provides configuration for the netdev collector
// of the node-exporter agent. The netdev collector collects network device statistics
// such as bytes, packets, errors, and drops per device.
// It is enabled by default.
type NodeExporterCollectorNetDevConfig struct {
	// collectionPolicy declares whether the netdev collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the netdev collector is active and network device statistics are collected.
	// When set to "DoNotCollect", the netdev collector is inactive and the corresponding metrics become unavailable.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorNetClassConfig provides configuration for the netclass collector
// of the node-exporter agent. The netclass collector collects information about network devices
// such as network speed, MTU, and carrier status.
// It is enabled by default.
// When collectionPolicy is DoNotCollect, the collect field must not be set.
// +kubebuilder:validation:XValidation:rule="has(self.collectionPolicy) && self.collectionPolicy == 'Collect' ? true : !has(self.collect)",message="collect is forbidden when collectionPolicy is not Collect"
// +union
type NodeExporterCollectorNetClassConfig struct {
	// collectionPolicy declares whether the netclass collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the netclass collector is active and network class information is collected.
	// When set to "DoNotCollect", the netclass collector is inactive and the corresponding metrics become unavailable.
	// When set to "DoNotCollect", the collect field must not be set.
	// +unionDiscriminator
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
	// collect contains configuration options that apply only when the netclass collector is actively collecting metrics
	// (i.e. when collectionPolicy is Collect).
	// collect is optional and may be omitted even when collectionPolicy is Collect.
	// collect may only be set when collectionPolicy is Collect.
	// When set, at least one field must be specified within collect.
	// +unionMember
	// +optional
	Collect NodeExporterCollectorNetClassCollectConfig `json:"collect,omitzero,omitempty"`
}

// NodeExporterCollectorNetClassCollectConfig holds configuration options for the netclass collector
// when it is actively collecting metrics. At least one field must be specified.
// +kubebuilder:validation:MinProperties=1
type NodeExporterCollectorNetClassCollectConfig struct {
	// statsGatherer selects which implementation the netclass collector uses to gather statistics (sysfs or netlink).
	// statsGatherer is optional.
	// Valid values are "Sysfs" and "Netlink".
	// When set to "Netlink", the netlink implementation is used; when set to "Sysfs", the sysfs implementation is used.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default,
	// which is subject to change over time. The current default is Netlink.
	// +optional
	StatsGatherer NodeExporterNetclassStatsGatherer `json:"statsGatherer,omitempty"`
}

// NodeExporterCollectorBuddyInfoConfig provides configuration for the buddyinfo collector
// of the node-exporter agent. The buddyinfo collector collects statistics about memory fragmentation
// from the node_buddyinfo_blocks metric using data from /proc/buddyinfo.
// It is disabled by default.
type NodeExporterCollectorBuddyInfoConfig struct {
	// collectionPolicy declares whether the buddyinfo collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the buddyinfo collector is active and memory fragmentation statistics are collected.
	// When set to "DoNotCollect", the buddyinfo collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorMountStatsConfig provides configuration for the mountstats collector
// of the node-exporter agent. The mountstats collector collects statistics about NFS volume I/O activities.
// It is disabled by default.
// Enabling this collector may produce metrics with high cardinality. If you enable this
// collector, closely monitor the prometheus-k8s deployment for excessive memory usage.
type NodeExporterCollectorMountStatsConfig struct {
	// collectionPolicy declares whether the mountstats collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the mountstats collector is active and NFS volume I/O statistics are collected.
	// When set to "DoNotCollect", the mountstats collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorKSMDConfig provides configuration for the ksmd collector
// of the node-exporter agent. The ksmd collector collects statistics from the kernel
// same-page merger daemon.
// It is disabled by default.
type NodeExporterCollectorKSMDConfig struct {
	// collectionPolicy declares whether the ksmd collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the ksmd collector is active and kernel same-page merger statistics are collected.
	// When set to "DoNotCollect", the ksmd collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorProcessesConfig provides configuration for the processes collector
// of the node-exporter agent. The processes collector collects statistics from processes and threads
// running in the system.
// It is disabled by default.
type NodeExporterCollectorProcessesConfig struct {
	// collectionPolicy declares whether the processes collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the processes collector is active and process/thread statistics are collected.
	// When set to "DoNotCollect", the processes collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// NodeExporterCollectorSystemdConfig provides configuration for the systemd collector
// of the node-exporter agent. The systemd collector collects statistics on the systemd daemon
// and its managed services.
// It is disabled by default.
// Enabling this collector with a long list of selected units may produce metrics with high
// cardinality. If you enable this collector, closely monitor the prometheus-k8s deployment
// for excessive memory usage.
// When collectionPolicy is DoNotCollect, the collect field must not be set.
// +kubebuilder:validation:XValidation:rule="has(self.collectionPolicy) && self.collectionPolicy == 'Collect' ? true : !has(self.collect)",message="collect is forbidden when collectionPolicy is not Collect"
// +union
type NodeExporterCollectorSystemdConfig struct {
	// collectionPolicy declares whether the systemd collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the systemd collector is active and systemd unit statistics are collected.
	// When set to "DoNotCollect", the systemd collector is inactive and the collect field must not be set.
	// +unionDiscriminator
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
	// collect contains configuration options that apply only when the systemd collector is actively collecting metrics
	// (i.e. when collectionPolicy is Collect).
	// collect is optional and may be omitted even when collectionPolicy is Collect.
	// collect may only be set when collectionPolicy is Collect.
	// When set, at least one field must be specified within collect.
	// +unionMember
	// +optional
	Collect NodeExporterCollectorSystemdCollectConfig `json:"collect,omitzero,omitempty"`
}

// NodeExporterCollectorSystemdCollectConfig holds configuration options for the systemd collector
// when it is actively collecting metrics. At least one field must be specified.
// +kubebuilder:validation:MinProperties=1
type NodeExporterCollectorSystemdCollectConfig struct {
	// units is a list of regular expression patterns that match systemd units to be included
	// by the systemd collector.
	// units is optional.
	// By default, the list is empty, so the collector exposes no metrics for systemd units.
	// Each entry is a regular expression pattern and must be at least 1 character and at most 1024 characters.
	// Maximum length for this list is 50.
	// Minimum length for this list is 1.
	// Entries in this list must be unique.
	// +kubebuilder:validation:MaxItems=50
	// +kubebuilder:validation:MinItems=1
	// +listType=set
	// +optional
	Units []NodeExporterSystemdUnit `json:"units,omitempty"`
}

// NodeExporterSystemdUnit is a string that is interpreted as a Go regular expression
// pattern by the controller to match systemd unit names.
// Invalid regular expressions will cause a controller-level error at runtime.
// Must be at least 1 character and at most 1024 characters.
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=1024
type NodeExporterSystemdUnit string

// NodeExporterCollectorSoftirqsConfig provides configuration for the softirqs collector
// of the node-exporter agent. The softirqs collector exposes detailed softirq statistics
// from /proc/softirqs.
// It is disabled by default.
type NodeExporterCollectorSoftirqsConfig struct {
	// collectionPolicy declares whether the softirqs collector collects metrics.
	// This field is required.
	// Valid values are "Collect" and "DoNotCollect".
	// When set to "Collect", the softirqs collector is active and softirq statistics are collected.
	// When set to "DoNotCollect", the softirqs collector is inactive.
	// +required
	CollectionPolicy NodeExporterCollectorCollectionPolicy `json:"collectionPolicy,omitempty"`
}

// MonitoringPluginConfig provides configuration options for the monitoring plugin
// that runs as a dynamic plugin of the OpenShift web console.
// The monitoring plugin provides the monitoring UI in the OpenShift web console
// for visualizing metrics, alerts, and dashboards.
// At least one field must be specified; an empty monitoringPluginConfig object is not allowed.
// +kubebuilder:validation:MinProperties=1
type MonitoringPluginConfig struct {
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the monitoring-plugin container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 10m
	//    - name: memory
	//      request: 50Mi
	//
	// When specified, resources must contain at least 1 entry and must not exceed 5 entries.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines the tolerations required for the monitoring-plugin Pods.
	// This field is optional.
	//
	// When omitted, the monitoring-plugin Pods will not have any tolerations, which
	// means they will only be scheduled on nodes with no taints.
	// When specified, tolerations must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +listType=atomic
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how monitoring-plugin Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// When specified, this list must contain at least 1 entry and must not exceed 10 entries.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// UserDefinedMonitoring config for user-defined projects.
type UserDefinedMonitoring struct {
	// mode defines the different configurations of UserDefinedMonitoring
	// Valid values are Disabled and NamespaceIsolated
	// Disabled disables monitoring for user-defined projects. This restricts the default monitoring stack, installed in the openshift-monitoring project, to monitor only platform namespaces, which prevents any custom monitoring configurations or resources from being applied to user-defined namespaces.
	// NamespaceIsolated enables monitoring for user-defined projects with namespace-scoped tenancy. This ensures that metrics, alerts, and monitoring data are isolated at the namespace level.
	// The current default value is `Disabled`.
	// +required
	// +kubebuilder:validation:Enum=Disabled;NamespaceIsolated
	Mode UserDefinedMode `json:"mode"`
}

// UserDefinedMode specifies mode for UserDefine Monitoring
// +enum
type UserDefinedMode string

const (
	// UserDefinedDisabled disables monitoring for user-defined projects. This restricts the default monitoring stack, installed in the openshift-monitoring project, to monitor only platform namespaces, which prevents any custom monitoring configurations or resources from being applied to user-defined namespaces.
	UserDefinedDisabled UserDefinedMode = "Disabled"
	// UserDefinedNamespaceIsolated enables monitoring for user-defined projects with namespace-scoped tenancy. This ensures that metrics, alerts, and monitoring data are isolated at the namespace level.
	UserDefinedNamespaceIsolated UserDefinedMode = "NamespaceIsolated"
)

// alertmanagerConfig provides configuration options for the default Alertmanager instance
// that runs in the `openshift-monitoring` namespace. Use this configuration to control
// whether the default Alertmanager is deployed, how it logs, and how its pods are scheduled.
// +kubebuilder:validation:XValidation:rule="self.deploymentMode == 'CustomConfig' ? has(self.customConfig) : !has(self.customConfig)",message="customConfig is required when deploymentMode is CustomConfig, and forbidden otherwise"
type AlertmanagerConfig struct {
	// deploymentMode determines whether the default Alertmanager instance should be deployed
	// as part of the monitoring stack.
	// Allowed values are Disabled, DefaultConfig, and CustomConfig.
	// When set to Disabled, the Alertmanager instance will not be deployed.
	// When set to DefaultConfig, the platform will deploy Alertmanager with default settings.
	// When set to CustomConfig, the Alertmanager will be deployed with custom configuration.
	//
	// +unionDiscriminator
	// +required
	DeploymentMode AlertManagerDeployMode `json:"deploymentMode,omitempty"`

	// customConfig must be set when deploymentMode is CustomConfig, and must be unset otherwise.
	// When set to CustomConfig, the Alertmanager will be deployed with custom configuration.
	// +optional
	CustomConfig AlertmanagerCustomConfig `json:"customConfig,omitempty,omitzero"`
}

// AlertmanagerCustomConfig represents the configuration for a custom Alertmanager deployment.
// alertmanagerCustomConfig provides configuration options for the default Alertmanager instance
// that runs in the `openshift-monitoring` namespace. Use this configuration to control
// whether the default Alertmanager is deployed, how it logs, and how its pods are scheduled.
// +kubebuilder:validation:MinProperties=1
type AlertmanagerCustomConfig struct {
	// logLevel defines the verbosity of logs emitted by Alertmanager.
	// This field allows users to control the amount and severity of logs generated, which can be useful
	// for debugging issues or reducing noise in production environments.
	// Allowed values are Error, Warn, Info, and Debug.
	// When set to Error, only errors will be logged.
	// When set to Warn, both warnings and errors will be logged.
	// When set to Info, general information, warnings, and errors will all be logged.
	// When set to Debug, detailed debugging information will be logged.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	// The current default value is `Info`.
	// +optional
	LogLevel LogLevel `json:"logLevel,omitempty"`
	// nodeSelector defines the nodes on which the Pods are scheduled
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the Alertmanager container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 4m
	//      limit: null
	//    - name: memory
	//      request: 40Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// secrets defines a list of secrets that need to be mounted into the Alertmanager.
	// The secrets must reside within the same namespace as the Alertmanager object.
	// They will be added as volumes named secret-<secret-name> and mounted at
	// /etc/alertmanager/secrets/<secret-name> within the 'alertmanager' container of
	// the Alertmanager Pods.
	//
	// These secrets can be used to authenticate Alertmanager with endpoint receivers.
	// For example, you can use secrets to:
	// - Provide certificates for TLS authentication with receivers that require private CA certificates
	// - Store credentials for Basic HTTP authentication with receivers that require password-based auth
	// - Store any other authentication credentials needed by your alert receivers
	//
	// This field is optional.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries in this list must be unique.
	// +optional
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=set
	Secrets []SecretName `json:"secrets,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how Alertmanager Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
	// volumeClaimTemplate defines persistent storage for Alertmanager. Use this setting to
	// configure the persistent volume claim, including storage class and volume size.
	// If omitted, the Pod uses ephemeral storage and alert data will not persist
	// across restarts.
	// +optional
	VolumeClaimTemplate *v1.PersistentVolumeClaim `json:"volumeClaimTemplate,omitempty,omitzero"`
}

// AlertManagerDeployMode defines the deployment state of the platform Alertmanager instance.
//
// Possible values:
// - "Disabled": The Alertmanager instance will not be deployed.
// - "DefaultConfig": The Alertmanager instance will be deployed with default settings.
// - "CustomConfig": The Alertmanager instance will be deployed with custom configuration.
// +kubebuilder:validation:Enum=Disabled;DefaultConfig;CustomConfig
type AlertManagerDeployMode string

const (
	// AlertManagerModeDisabled means the Alertmanager instance will not be deployed.
	AlertManagerDeployModeDisabled AlertManagerDeployMode = "Disabled"
	// AlertManagerModeDefaultConfig means the Alertmanager instance will be deployed with default settings.
	AlertManagerDeployModeDefaultConfig AlertManagerDeployMode = "DefaultConfig"
	// AlertManagerModeCustomConfig means the Alertmanager instance will be deployed with custom configuration.
	AlertManagerDeployModeCustomConfig AlertManagerDeployMode = "CustomConfig"
)

// LogLevel defines the verbosity of logs emitted by Alertmanager.
// Valid values are Error, Warn, Info and Debug.
// +kubebuilder:validation:Enum=Error;Warn;Info;Debug
type LogLevel string

const (
	// LogLevelError only errors will be logged.
	LogLevelError LogLevel = "Error"
	// LogLevelWarn, both warnings and errors will be logged.
	LogLevelWarn LogLevel = "Warn"
	// LogLevelInfo, general information, warnings, and errors will all be logged.
	LogLevelInfo LogLevel = "Info"
	// LogLevelDebug, detailed debugging information will be logged.
	LogLevelDebug LogLevel = "Debug"
)

// MaxItems on []ContainerResource fields is kept at 5 to stay within the
// Kubernetes CRD CEL validation cost budget (StaticEstimatedCRDCostLimit).
// The quantity() CEL function has a high fixed estimated cost per invocation,
// and the limit-vs-request comparison rule is costed per maxItems per location.
// With multiple structs in ClusterMonitoringSpec embedding []ContainerResource,
// maxItems > 5 causes the total estimated rule cost to exceed the budget.

// ContainerResource defines a single resource requirement for a container.
// +kubebuilder:validation:XValidation:rule="has(self.request) || has(self.limit)",message="at least one of request or limit must be set"
// +kubebuilder:validation:XValidation:rule="!(has(self.request) && has(self.limit)) || quantity(self.limit).compareTo(quantity(self.request)) >= 0",message="limit must be greater than or equal to request"
type ContainerResource struct {
	// name of the resource (e.g. "cpu", "memory", "hugepages-2Mi").
	// This field is required.
	// name must consist only of alphanumeric characters, `-`, `_` and `.` and must start and end with an alphanumeric character.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.qualifiedName().validate(self).hasValue()",message="name must consist only of alphanumeric characters, `-`, `_` and `.` and must start and end with an alphanumeric character"
	Name string `json:"name,omitempty"`

	// request is the minimum amount of the resource required (e.g. "2Mi", "1Gi").
	// This field is optional.
	// When limit is specified, request cannot be greater than limit.
	// The value must be greater than 0 when specified.
	// +optional
	// +kubebuilder:validation:XIntOrString
	// +kubebuilder:validation:MaxLength=20
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="quantity(self).isGreaterThan(quantity('0'))",message="request must be a positive, non-zero quantity"
	Request resource.Quantity `json:"request,omitempty"`

	// limit is the maximum amount of the resource allowed (e.g. "2Mi", "1Gi").
	// This field is optional.
	// When request is specified, limit cannot be less than request.
	// The value must be greater than 0 when specified.
	// +optional
	// +kubebuilder:validation:XIntOrString
	// +kubebuilder:validation:MaxLength=20
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="quantity(self).isGreaterThan(quantity('0'))",message="limit must be a positive, non-zero quantity"
	Limit resource.Quantity `json:"limit,omitempty"`
}

// SecretName is a type that represents the name of a Secret in the same namespace.
// It must be at most 253 characters in length.
// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character."
// +kubebuilder:validation:MaxLength=63
type SecretName string

// MetricsServerConfig provides configuration options for the Metrics Server instance
// that runs in the `openshift-monitoring` namespace. Use this configuration to control
// how the Metrics Server instance is deployed, how it logs, and how its pods are scheduled.
// +kubebuilder:validation:MinProperties=1
type MetricsServerConfig struct {
	// audit defines the audit configuration used by the Metrics Server instance.
	// audit is optional.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	//The current default sets audit.profile to Metadata
	// +optional
	Audit Audit `json:"audit,omitempty,omitzero"`
	// nodeSelector defines the nodes on which the Pods are scheduled
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// verbosity defines the verbosity of log messages for Metrics Server.
	// Valid values are Errors, Info, Trace, TraceAll and omitted.
	// When set to Errors, only critical messages and errors are logged.
	// When set to Info, only basic information messages are logged.
	// When set to Trace, information useful for general debugging is logged.
	// When set to TraceAll, detailed information about metric scraping is logged.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	// The current default value is `Errors`
	// +optional
	Verbosity VerbosityLevel `json:"verbosity,omitempty,omitzero"`
	// resources defines the compute resource requests and limits for the Metrics Server container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 4m
	//      limit: null
	//    - name: memory
	//      request: 40Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// topologySpreadConstraints defines rules for how Metrics Server Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// PrometheusOperatorConfig provides configuration options for the Prometheus Operator instance
// Use this configuration to control how the Prometheus Operator instance is deployed, how it logs, and how its pods are scheduled.
// +kubebuilder:validation:MinProperties=1
type PrometheusOperatorConfig struct {
	// logLevel defines the verbosity of logs emitted by Prometheus Operator.
	// This field allows users to control the amount and severity of logs generated, which can be useful
	// for debugging issues or reducing noise in production environments.
	// Allowed values are Error, Warn, Info, and Debug.
	// When set to Error, only errors will be logged.
	// When set to Warn, both warnings and errors will be logged.
	// When set to Info, general information, warnings, and errors will all be logged.
	// When set to Debug, detailed debugging information will be logged.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	// The current default value is `Info`.
	// +optional
	LogLevel LogLevel `json:"logLevel,omitempty"`
	// nodeSelector defines the nodes on which the Pods are scheduled
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the Prometheus Operator container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 4m
	//      limit: null
	//    - name: memory
	//      request: 40Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how Prometheus Operator Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// PrometheusOperatorAdmissionWebhookConfig provides configuration options for the admission webhook
// component of Prometheus Operator that runs in the `openshift-monitoring` namespace. The admission
// webhook validates PrometheusRule and AlertmanagerConfig objects, mutates PrometheusRule annotations,
// and converts AlertmanagerConfig objects between API versions.
// +kubebuilder:validation:MinProperties=1
type PrometheusOperatorAdmissionWebhookConfig struct {
	// resources defines the compute resource requests and limits for the
	// prometheus-operator-admission-webhook container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 5m
	//      limit: null
	//    - name: memory
	//      request: 30Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// topologySpreadConstraints defines rules for how admission webhook Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// PrometheusConfig provides configuration options for the Prometheus instance.
// Use this configuration to control
// Prometheus deployment, pod scheduling, resource allocation, retention policies, and external integrations.
// +kubebuilder:validation:MinProperties=1
type PrometheusConfig struct {
	// additionalAlertmanagerConfigs configures additional Alertmanager instances that receive alerts from
	// the Prometheus component. This is useful for organizations that need to:
	//   - Send alerts to external monitoring systems (like PagerDuty, Slack, or custom webhooks)
	//   - Route different types of alerts to different teams or systems
	//   - Integrate with existing enterprise alerting infrastructure
	//   - Maintain separate alert routing for compliance or organizational requirements
	// When omitted, no additional Alertmanager instances are configured (default behavior).
	// When provided, at least one configuration must be specified (minimum 1, maximum 10 items).
	// Entries must have unique names (name is the list key).
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +listType=map
	// +listMapKey=name
	AdditionalAlertmanagerConfigs []AdditionalAlertmanagerConfig `json:"additionalAlertmanagerConfigs,omitempty"`
	// enforcedBodySizeLimitBytes enforces a body size limit (in bytes) for Prometheus scraped metrics.
	// If a scraped target's body response is larger than the limit, the scrape will fail.
	// This helps protect Prometheus from targets that return excessively large responses.
	// The value is specified in bytes (e.g., 4194304 for 4MB, 1073741824 for 1GB).
	// When omitted, the Cluster Monitoring Operator automatically calculates an appropriate
	// limit based on cluster capacity. Set an explicit value to override the automatic calculation.
	// Minimum value is 10240 (10kB).
	// Maximum value is 1073741824 (1GB).
	// +kubebuilder:validation:Minimum=10240
	// +kubebuilder:validation:Maximum=1073741824
	// +optional
	EnforcedBodySizeLimitBytes int64 `json:"enforcedBodySizeLimitBytes,omitempty"`
	// externalLabels defines labels to be attached to time series and alerts
	// when communicating with external systems such as federation, remote storage,
	// and Alertmanager. These labels are not stored with metrics on disk; they are
	// only added when data leaves Prometheus (e.g., during federation queries,
	// remote write, or alert notifications).
	// At least 1 label must be specified when set, with a maximum of 50 labels allowed.
	// Each label key must be unique within this list.
	// When omitted, no external labels are applied.
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=50
	// +listType=map
	// +listMapKey=key
	ExternalLabels []Label `json:"externalLabels,omitempty"`
	// logLevel defines the verbosity of logs emitted by Prometheus.
	// This field allows users to control the amount and severity of logs generated, which can be useful
	// for debugging issues or reducing noise in production environments.
	// Allowed values are Error, Warn, Info, and Debug.
	// When set to Error, only errors will be logged.
	// When set to Warn, both warnings and errors will be logged.
	// When set to Info, general information, warnings, and errors will all be logged.
	// When set to Debug, detailed debugging information will be logged.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, that is subject to change over time.
	// The current default value is `Info`.
	// +optional
	LogLevel LogLevel `json:"logLevel,omitempty"`
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least one key-value pair (minimum of 1)
	// and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// queryLogFile specifies the file to which PromQL queries are logged.
	// This setting can be either a filename, in which
	// case the queries are saved to an `emptyDir` volume
	// at `/var/log/prometheus`, or a full path to a location where
	// an `emptyDir` volume will be mounted and the queries saved.
	// Writing to `/dev/stderr`, `/dev/stdout` or `/dev/null` is supported, but
	// writing to any other `/dev/` path is not supported. Relative paths are
	// also not supported.
	// By default, PromQL queries are not logged.
	// Must be an absolute path starting with `/` or a simple filename without path separators.
	// Must not contain consecutive slashes, end with a slash, or include '..' path traversal.
	// Must contain only alphanumeric characters, '.', '_', '-', or '/'.
	// Must be between 1 and 255 characters in length.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=255
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9._/-]+$')",message="must contain only alphanumeric characters, '.', '_', '-', or '/'"
	// +kubebuilder:validation:XValidation:rule="self.startsWith('/') || !self.contains('/')",message="must be an absolute path starting with '/' or a simple filename without '/'"
	// +kubebuilder:validation:XValidation:rule="!self.startsWith('/dev/') || self in ['/dev/stdout', '/dev/stderr', '/dev/null']",message="only /dev/stdout, /dev/stderr, and /dev/null are allowed as /dev/ paths"
	// +kubebuilder:validation:XValidation:rule="!self.contains('//') && !self.endsWith('/') && !self.contains('..')",message="must not contain '//', end with '/', or contain '..'"
	QueryLogFile string `json:"queryLogFile,omitempty"`
	// remoteWrite defines the remote write configuration, including URL, authentication, and relabeling settings.
	// Remote write allows Prometheus to send metrics it collects to external long-term storage systems.
	// When omitted, no remote write endpoints are configured.
	// When provided, at least one configuration must be specified (minimum 1, maximum 10 items).
	// Entries must have unique names (name is the list key).
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +listType=map
	// +listMapKey=name
	// +optional
	RemoteWrite []RemoteWriteSpec `json:"remoteWrite,omitempty"`
	// resources defines the compute resource requests and limits for the Prometheus container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 4m
	//      limit: null
	//    - name: memory
	//      request: 40Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// retention configures how long Prometheus retains metrics data and how much storage it can use.
	// When omitted, the platform chooses reasonable defaults (currently 15 days retention, no size limit).
	// +optional
	Retention Retention `json:"retention,omitempty,omitzero"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10
	// Minimum length for this list is 1
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how Prometheus Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
	// collectionProfile defines the metrics collection profile that Prometheus uses to collect
	// metrics from the platform components. Supported values are `Full` or
	// `Minimal`. In the `Full` profile (default), Prometheus collects all
	// metrics that are exposed by the platform components. In the `Minimal`
	// profile, Prometheus only collects metrics necessary for the default
	// platform alerts, recording rules, telemetry and console dashboards.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is `Full`.
	// +optional
	CollectionProfile CollectionProfile `json:"collectionProfile,omitempty"`
	// volumeClaimTemplate defines persistent storage for Prometheus. Use this setting to
	// configure the persistent volume claim, including storage class and volume size.
	// If omitted, the Pod uses ephemeral storage and Prometheus data will not persist
	// across restarts.
	// +optional
	VolumeClaimTemplate *v1.PersistentVolumeClaim `json:"volumeClaimTemplate,omitempty,omitzero"`
}

// AlertmanagerScheme defines the URL scheme to use when communicating with Alertmanager instances.
// +kubebuilder:validation:Enum=HTTP;HTTPS
type AlertmanagerScheme string

const (
	AlertmanagerSchemeHTTP  AlertmanagerScheme = "HTTP"
	AlertmanagerSchemeHTTPS AlertmanagerScheme = "HTTPS"
)

// AdditionalAlertmanagerConfig represents configuration for additional Alertmanager instances.
// The `AdditionalAlertmanagerConfig` resource defines settings for how a
// component communicates with additional Alertmanager instances.
type AdditionalAlertmanagerConfig struct {
	// name is a unique identifier for this Alertmanager configuration entry.
	// The name must be a valid DNS subdomain (RFC 1123): lowercase alphanumeric characters,
	// hyphens, or periods, and must start and end with an alphanumeric character.
	// Minimum length is 1 character (empty string is invalid).
	// Maximum length is 253 characters.
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character."
	// +required
	Name string `json:"name,omitempty"`
	// authorization configures the authentication method for Alertmanager connections.
	// Supports bearer token authentication. When omitted, no authentication is used.
	// +optional
	Authorization AuthorizationConfig `json:"authorization,omitempty,omitzero"`
	// pathPrefix defines an optional URL path prefix to prepend to the Alertmanager API endpoints.
	// For example, if your Alertmanager is behind a reverse proxy at "/alertmanager/",
	// set this to "/alertmanager" so requests go to "/alertmanager/api/v1/alerts" instead of "/api/v1/alerts".
	// This is commonly needed when Alertmanager is deployed behind ingress controllers or load balancers.
	// When no prefix is needed, omit this field; do not set it to "/" as that would produce paths with double slashes (e.g. "//api/v1/alerts").
	// Must start with "/", must not end with "/", and must not be exactly "/".
	// Must not contain query strings ("?") or fragments ("#").
	// +kubebuilder:validation:MaxLength=255
	// +kubebuilder:validation:MinLength=2
	// +kubebuilder:validation:XValidation:rule="self.startsWith('/')",message="pathPrefix must start with '/'"
	// +kubebuilder:validation:XValidation:rule="!self.endsWith('/')",message="pathPrefix must not end with '/'"
	// +kubebuilder:validation:XValidation:rule="self != '/'",message="pathPrefix must not be '/' (would produce double slashes in request path); omit for no prefix"
	// +kubebuilder:validation:XValidation:rule="!self.contains('?') && !self.contains('#')",message="pathPrefix must not contain '?' or '#'"
	// +optional
	PathPrefix string `json:"pathPrefix,omitempty"`
	// scheme defines the URL scheme to use when communicating with Alertmanager
	// instances.
	// Possible values are `HTTP` or `HTTPS`.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The current default value is `HTTP`.
	// +optional
	Scheme AlertmanagerScheme `json:"scheme,omitempty"`
	// staticConfigs is a list of statically configured Alertmanager endpoints in the form
	// of `<host>:<port>`. Each entry must be a valid hostname, IPv4 address, or IPv6 address
	// (in brackets) followed by a colon and a valid port number (1-65535).
	// Examples: "alertmanager.example.com:9093", "192.168.1.100:9093", "[::1]:9093"
	// At least one endpoint must be specified (minimum 1, maximum 10 endpoints).
	// Each entry must be unique and non-empty (empty string is invalid).
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=255
	// +kubebuilder:validation:items:XValidation:rule="isURL('http://' + self) && size(url('http://' + self).getHostname()) > 0 && size(url('http://' + self).getPort()) > 0 && int(url('http://' + self).getPort()) >= 1 && int(url('http://' + self).getPort()) <= 65535",message="must be a valid 'host:port' where host is a DNS name, IPv4, or IPv6 address (in brackets), and port is 1-65535"
	// +listType=set
	// +required
	StaticConfigs []string `json:"staticConfigs,omitempty"`
	// timeoutSeconds defines the timeout in seconds for requests to Alertmanager.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// Currently the default is 10 seconds.
	// Minimum value is 1 second.
	// Maximum value is 600 seconds (10 minutes).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=600
	// +optional
	TimeoutSeconds int32 `json:"timeoutSeconds,omitempty"`
	// tlsConfig defines the TLS settings to use for Alertmanager connections.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	TLSConfig TLSConfig `json:"tlsConfig,omitempty,omitzero"`
}

// Label represents a key/value pair for external labels.
type Label struct {
	// key is the name of the label.
	// Prometheus supports UTF-8 label names, so any valid UTF-8 string is allowed.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MaxLength=128
	// +kubebuilder:validation:MinLength=1
	Key string `json:"key,omitempty"`
	// value is the value of the label.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MaxLength=128
	// +kubebuilder:validation:MinLength=1
	Value string `json:"value,omitempty"`
}

// RemoteWriteSpec represents configuration for remote write endpoints.
type RemoteWriteSpec struct {
	// url is the URL of the remote write endpoint.
	// Must be a valid URL with http or https scheme and a non-empty hostname.
	// Query parameters, fragments, and user information (e.g. user:password@host) are not allowed.
	// Empty string is invalid. Must be between 1 and 2048 characters in length.
	// +required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="must be a valid URL"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getScheme() == 'http' || url(self).getScheme() == 'https'",message="must use http or https scheme"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || size(url(self).getHostname()) > 0",message="must have a non-empty hostname"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getQuery().size() == 0",message="query parameters are not allowed"
	// +kubebuilder:validation:XValidation:rule="!self.matches('.*#.*')",message="fragments are not allowed"
	// +kubebuilder:validation:XValidation:rule="!self.matches('.*@.*')",message="user information (e.g. user:password@host) is not allowed"
	URL string `json:"url,omitempty"`
	// name is a required identifier for this remote write configuration (name is the list key for the remoteWrite list).
	// This name is used in metrics and logging to differentiate remote write queues.
	// Must contain only alphanumeric characters, hyphens, and underscores.
	// Must be between 1 and 63 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9_-]+$')",message="must contain only alphanumeric characters, hyphens, and underscores"
	Name string `json:"name,omitempty"`
	// authorization defines the authorization method for the remote write endpoint.
	// When omitted, no authorization is performed.
	// When set, type must be one of BearerToken, BasicAuth, OAuth2, SigV4, SafeAuthorization, or ServiceAccount; the corresponding nested config must be set (ServiceAccount has no config).
	// +optional
	AuthorizationConfig RemoteWriteAuthorization `json:"authorization,omitzero"`
	// headers specifies the custom HTTP headers to be sent along with each remote write request.
	// Sending custom headers makes the configuration of a proxy in between optional and helps the
	// receiver recognize the given source better.
	// Clients MAY allow users to send custom HTTP headers; they MUST NOT allow users to configure
	// them in such a way as to send reserved headers. Headers set by Prometheus cannot be overwritten.
	// When omitted, no custom headers are sent.
	// Maximum of 50 headers can be specified. Each header name must be unique.
	// Each header name must contain only alphanumeric characters, hyphens, and underscores, and must not be a reserved Prometheus header (Host, Authorization, Content-Encoding, Content-Type, X-Prometheus-Remote-Write-Version, User-Agent, Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization, WWW-Authenticate).
	// +optional
	// +kubebuilder:validation:MinItems=0
	// +kubebuilder:validation:MaxItems=50
	// +kubebuilder:validation:items:XValidation:rule="self.name.matches('^[a-zA-Z0-9_-]+$')",message="header name must contain only alphanumeric characters, hyphens, and underscores"
	// +kubebuilder:validation:items:XValidation:rule="!self.name.matches('(?i)^(host|authorization|content-encoding|content-type|x-prometheus-remote-write-version|user-agent|connection|keep-alive|proxy-authenticate|proxy-authorization|www-authenticate)$')",message="header name must not be a reserved Prometheus header (Host, Authorization, Content-Encoding, Content-Type, X-Prometheus-Remote-Write-Version, User-Agent, Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization, WWW-Authenticate)"
	// +listType=map
	// +listMapKey=name
	Headers []PrometheusRemoteWriteHeader `json:"headers,omitempty"`
	// metadataConfig configures the sending of series metadata to remote storage.
	// When omitted, no metadata is sent.
	// When set to sendPolicy: Default, metadata is sent using platform-chosen defaults (e.g. send interval 30 seconds).
	// When set to sendPolicy: Custom, metadata is sent using the settings in the custom field (e.g. custom.sendIntervalSeconds).
	// +optional
	MetadataConfig MetadataConfig `json:"metadataConfig,omitempty,omitzero"`
	// proxyUrl defines an optional proxy URL.
	// If the cluster-wide proxy is enabled, it replaces the proxyUrl setting.
	// The cluster-wide proxy supports both HTTP and HTTPS proxies, with HTTPS taking precedence.
	// When omitted, no proxy is used.
	// Must be a valid URL with http or https scheme.
	// Must be between 1 and 2048 characters in length.
	// +optional
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="isURL(self) && (url(self).getScheme() == 'http' || url(self).getScheme() == 'https')",message="must be a valid URL with http or https scheme"
	ProxyURL string `json:"proxyUrl,omitempty"`
	// queueConfig allows tuning configuration for remote write queue parameters.
	// When omitted, default queue configuration is used.
	// +optional
	QueueConfig QueueConfig `json:"queueConfig,omitempty,omitzero"`
	// remoteTimeoutSeconds defines the timeout in seconds for requests to the remote write endpoint.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// Minimum value is 1 second.
	// Maximum value is 600 seconds (10 minutes).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=600
	RemoteTimeoutSeconds int32 `json:"remoteTimeoutSeconds,omitempty"`
	// exemplarsMode controls whether exemplars are sent via remote write.
	// Valid values are "Send", "DoNotSend" and omitted.
	// When set to "Send", Prometheus is configured to store a maximum of 100,000 exemplars in memory and send them with remote write.
	// Note that this setting only applies to user-defined monitoring. It is not applicable to default in-cluster monitoring.
	// When omitted or set to "DoNotSend", exemplars are not sent.
	// +optional
	ExemplarsMode ExemplarsMode `json:"exemplarsMode,omitempty"`
	// tlsConfig defines TLS authentication settings for the remote write endpoint.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// +optional
	TLSConfig TLSConfig `json:"tlsConfig,omitempty,omitzero"`
	// writeRelabelConfigs is a list of relabeling rules to apply before sending data to the remote endpoint.
	// When omitted, no relabeling is performed and all metrics are sent as-is.
	// Minimum of 1 and maximum of 10 relabeling rules can be specified.
	// Each rule must have a unique name.
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +listType=map
	// +listMapKey=name
	WriteRelabelConfigs []RelabelConfig `json:"writeRelabelConfigs,omitempty"`
}

// PrometheusRemoteWriteHeader defines a custom HTTP header for remote write requests.
// The header name must not be one of the reserved headers set by Prometheus (Host, Authorization, Content-Encoding, Content-Type, X-Prometheus-Remote-Write-Version, User-Agent, Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization, WWW-Authenticate).
// Header names must contain only case-insensitive alphanumeric characters, hyphens (-), and underscores (_); other characters (e.g. emoji) are rejected by validation.
// Validation is enforced on the Headers field in RemoteWriteSpec.
type PrometheusRemoteWriteHeader struct {
	// name is the HTTP header name. Must not be a reserved header (see type documentation).
	// Must contain only alphanumeric characters, hyphens, and underscores; invalid characters are rejected. Must be between 1 and 256 characters.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	Name string `json:"name,omitempty"`
	// value is the HTTP header value. Must be at most 4096 characters.
	// +required
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=4096
	Value *string `json:"value,omitempty"`
}

// BasicAuth defines basic authentication settings for the remote write endpoint URL.
type BasicAuth struct {
	// username defines the secret reference containing the username for basic authentication.
	// The secret must exist in the openshift-monitoring namespace.
	// +required
	Username SecretKeySelector `json:"username,omitzero,omitempty"`
	// password defines the secret reference containing the password for basic authentication.
	// The secret must exist in the openshift-monitoring namespace.
	// +required
	Password SecretKeySelector `json:"password,omitzero,omitempty"`
}

// RemoteWriteAuthorizationType defines the authorization method for remote write endpoints.
// +kubebuilder:validation:Enum=BearerToken;BasicAuth;OAuth2;SigV4;SafeAuthorization;ServiceAccount
type RemoteWriteAuthorizationType string

const (
	// RemoteWriteAuthorizationTypeBearerToken indicates bearer token from a secret.
	RemoteWriteAuthorizationTypeBearerToken RemoteWriteAuthorizationType = "BearerToken"
	// RemoteWriteAuthorizationTypeBasicAuth indicates HTTP basic authentication.
	RemoteWriteAuthorizationTypeBasicAuth RemoteWriteAuthorizationType = "BasicAuth"
	// RemoteWriteAuthorizationTypeOAuth2 indicates OAuth2 client credentials.
	RemoteWriteAuthorizationTypeOAuth2 RemoteWriteAuthorizationType = "OAuth2"
	// RemoteWriteAuthorizationTypeSigV4 indicates AWS Signature Version 4.
	RemoteWriteAuthorizationTypeSigV4 RemoteWriteAuthorizationType = "SigV4"
	// RemoteWriteAuthorizationTypeSafeAuthorization indicates authorization from a secret (Prometheus SafeAuthorization pattern).
	// The secret key contains the credentials (e.g. a Bearer token). Use the safeAuthorization field.
	RemoteWriteAuthorizationTypeSafeAuthorization RemoteWriteAuthorizationType = "SafeAuthorization"
	// RemoteWriteAuthorizationTypeServiceAccount indicates use of the pod's service account token for machine identity.
	// No additional field is required; the operator configures the token path.
	RemoteWriteAuthorizationTypeServiceAccount RemoteWriteAuthorizationType = "ServiceAccount"
)

// RemoteWriteAuthorization defines the authorization method for a remote write endpoint.
// Exactly one of the nested configs must be set according to the type discriminator.
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'BearerToken' ? has(self.bearerToken) : !has(self.bearerToken)",message="bearerToken is required when type is BearerToken, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'BasicAuth' ? has(self.basicAuth) : !has(self.basicAuth)",message="basicAuth is required when type is BasicAuth, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'OAuth2' ? has(self.oauth2) : !has(self.oauth2)",message="oauth2 is required when type is OAuth2, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'SigV4' ? has(self.sigv4) : !has(self.sigv4)",message="sigv4 is required when type is SigV4, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'SafeAuthorization' ? has(self.safeAuthorization) : !has(self.safeAuthorization)",message="safeAuthorization is required when type is SafeAuthorization, and forbidden otherwise"
// +union
type RemoteWriteAuthorization struct {
	// type specifies the authorization method to use.
	// Allowed values are BearerToken, BasicAuth, OAuth2, SigV4, SafeAuthorization, ServiceAccount.
	//
	// When set to BearerToken, the bearer token is read from a Secret referenced by the bearerToken field.
	//
	// When set to BasicAuth, HTTP basic authentication is used; the basicAuth field (username and password from Secrets) must be set.
	//
	// When set to OAuth2, OAuth2 client credentials flow is used; the oauth2 field (clientId, clientSecret, tokenUrl) must be set.
	//
	// When set to SigV4, AWS Signature Version 4 is used for authentication; the sigv4 field must be set.
	//
	// When set to SafeAuthorization, credentials are read from a single Secret key (Prometheus SafeAuthorization pattern). The secret key typically contains a Bearer token. Use the safeAuthorization field.
	//
	// When set to ServiceAccount, the pod's service account token is used for machine identity. No additional field is required; the operator configures the token path.
	// +unionDiscriminator
	// +required
	Type RemoteWriteAuthorizationType `json:"type,omitempty"`
	// safeAuthorization defines the secret reference containing the credentials for authentication (e.g. Bearer token).
	// Required when type is "SafeAuthorization", and forbidden otherwise. Maps to Prometheus SafeAuthorization. The secret must exist in the openshift-monitoring namespace.
	// +unionMember
	// +optional
	SafeAuthorization *v1.SecretKeySelector `json:"safeAuthorization,omitempty"`
	// bearerToken defines the secret reference containing the bearer token.
	// Required when type is "BearerToken", and forbidden otherwise.
	// +unionMember
	// +optional
	BearerToken SecretKeySelector `json:"bearerToken,omitempty,omitzero"`
	// basicAuth defines HTTP basic authentication credentials.
	// Required when type is "BasicAuth", and forbidden otherwise.
	// +unionMember
	// +optional
	BasicAuth BasicAuth `json:"basicAuth,omitempty,omitzero"`
	// oauth2 defines OAuth2 client credentials authentication.
	// Required when type is "OAuth2", and forbidden otherwise.
	// +unionMember
	// +optional
	OAuth2 OAuth2 `json:"oauth2,omitempty,omitzero"`
	// sigv4 defines AWS Signature Version 4 authentication.
	// Required when type is "SigV4", and forbidden otherwise.
	// +unionMember
	// +optional
	Sigv4 Sigv4 `json:"sigv4,omitempty,omitzero"`
}

// MetadataConfigSendPolicy defines whether to send metadata with platform defaults or with custom settings.
// +kubebuilder:validation:Enum=Default;Custom
type MetadataConfigSendPolicy string

const (
	// MetadataConfigSendPolicyDefault indicates metadata is sent using platform-chosen defaults (e.g. send interval 30 seconds).
	MetadataConfigSendPolicyDefault MetadataConfigSendPolicy = "Default"
	// MetadataConfigSendPolicyCustom indicates metadata is sent using the settings in the custom field.
	MetadataConfigSendPolicyCustom MetadataConfigSendPolicy = "Custom"
)

// MetadataConfig defines whether and how to send series metadata to remote write storage.
// +kubebuilder:validation:XValidation:rule="self.sendPolicy == 'Default' ? self.custom.sendIntervalSeconds == 0 : true",message="custom is forbidden when sendPolicy is Default"
type MetadataConfig struct {
	// sendPolicy specifies whether to send metadata and how it is configured.
	// Default: send metadata using platform-chosen defaults (e.g. send interval 30 seconds).
	// Custom: send metadata using the settings in the custom field.
	// +required
	SendPolicy MetadataConfigSendPolicy `json:"sendPolicy,omitempty"`
	// custom defines custom metadata send settings. Required when sendPolicy is Custom (must have at least one property), and forbidden when sendPolicy is Default.
	// +optional
	Custom MetadataConfigCustom `json:"custom,omitempty,omitzero"`
}

// MetadataConfigCustom defines custom settings for sending series metadata when sendPolicy is Custom.
// At least one property must be set when sendPolicy is Custom (e.g. sendIntervalSeconds).
// +kubebuilder:validation:MinProperties=1
type MetadataConfigCustom struct {
	// sendIntervalSeconds is the interval in seconds at which metadata is sent.
	// When omitted, the platform chooses a reasonable default (e.g. 30 seconds).
	// Minimum value is 1 second. Maximum value is 86400 seconds (24 hours).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=86400
	SendIntervalSeconds int32 `json:"sendIntervalSeconds,omitempty"`
}

// OAuth2 defines OAuth2 authentication settings for the remote write endpoint.
type OAuth2 struct {
	// clientId defines the secret reference containing the OAuth2 client ID.
	// The secret must exist in the openshift-monitoring namespace.
	// +required
	ClientID SecretKeySelector `json:"clientId,omitzero,omitempty"`
	// clientSecret defines the secret reference containing the OAuth2 client secret.
	// The secret must exist in the openshift-monitoring namespace.
	// +required
	ClientSecret SecretKeySelector `json:"clientSecret,omitzero,omitempty"`
	// tokenUrl is the URL to fetch the token from.
	// Must be a valid URL with http or https scheme.
	// Must be between 1 and 2048 characters in length.
	// +required
	// +kubebuilder:validation:MaxLength=2048
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:XValidation:rule="isURL(self)",message="must be a valid URL"
	// +kubebuilder:validation:XValidation:rule="!isURL(self) || url(self).getScheme() == 'http' || url(self).getScheme() == 'https'",message="must use http or https scheme"
	TokenURL string `json:"tokenUrl,omitempty"`
	// scopes is a list of OAuth2 scopes to request.
	// When omitted, no scopes are requested.
	// Maximum of 20 scopes can be specified.
	// Each scope must be between 1 and 256 characters.
	// +optional
	// +kubebuilder:validation:MinItems=0
	// +kubebuilder:validation:MaxItems=20
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=256
	// +listType=atomic
	Scopes []string `json:"scopes,omitempty"`
	// endpointParams defines additional parameters to append to the token URL.
	// When omitted, no additional parameters are sent.
	// Maximum of 20 parameters can be specified. Entries must have unique names (name is the list key).
	// +optional
	// +kubebuilder:validation:MinItems=0
	// +kubebuilder:validation:MaxItems=20
	// +listType=map
	// +listMapKey=name
	EndpointParams []OAuth2EndpointParam `json:"endpointParams,omitempty"`
}

// OAuth2EndpointParam defines a name/value parameter for the OAuth2 token URL.
type OAuth2EndpointParam struct {
	// name is the parameter name. Must be between 1 and 256 characters.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=256
	Name string `json:"name,omitempty"`
	// value is the optional parameter value. When omitted, the query parameter is applied as ?name (no value).
	// When set (including to the empty string), it is applied as ?name=value. Empty string may be used when the
	// external system expects a parameter with an empty value (e.g. ?parameter="").
	// Must be between 0 and 2048 characters when present (aligned with common URL length recommendations).
	// +optional
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=2048
	Value *string `json:"value,omitempty"`
}

// QueueConfig allows tuning configuration for remote write queue parameters.
// Configure this when you need to control throughput, backpressure, or retry behavior—for example to avoid overloading the remote endpoint, to reduce memory usage, or to tune for high-cardinality workloads. Consider capacity, maxShards, and batchSendDeadlineSeconds for throughput; minBackoffMilliseconds and maxBackoffMilliseconds for retries; and rateLimitedAction when the remote returns HTTP 429.
// +kubebuilder:validation:MinProperties=1
type QueueConfig struct {
	// capacity is the number of samples to buffer per shard before we start dropping them.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is 10000.
	// Minimum value is 1.
	// Maximum value is 1000000.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1000000
	Capacity int32 `json:"capacity,omitempty"`
	// maxShards is the maximum number of shards, i.e. amount of concurrency.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is 200.
	// Minimum value is 1.
	// Maximum value is 10000.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=10000
	MaxShards int32 `json:"maxShards,omitempty"`
	// minShards is the minimum number of shards, i.e. amount of concurrency.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is 1.
	// Minimum value is 1.
	// Maximum value is 10000.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=10000
	MinShards int32 `json:"minShards,omitempty"`
	// maxSamplesPerSend is the maximum number of samples per send.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is 1000.
	// Minimum value is 1.
	// Maximum value is 100000.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100000
	MaxSamplesPerSend int32 `json:"maxSamplesPerSend,omitempty"`
	// batchSendDeadlineSeconds is the maximum time in seconds a sample will wait in buffer before being sent.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// Minimum value is 1 second.
	// Maximum value is 3600 seconds (1 hour).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=3600
	BatchSendDeadlineSeconds int32 `json:"batchSendDeadlineSeconds,omitempty"`
	// minBackoffMilliseconds is the minimum retry delay in milliseconds.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// Minimum value is 1 millisecond.
	// Maximum value is 3600000 milliseconds (1 hour).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=3600000
	MinBackoffMilliseconds int32 `json:"minBackoffMilliseconds,omitempty"`
	// maxBackoffMilliseconds is the maximum retry delay in milliseconds.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// Minimum value is 1 millisecond.
	// Maximum value is 3600000 milliseconds (1 hour).
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=3600000
	MaxBackoffMilliseconds int32 `json:"maxBackoffMilliseconds,omitempty"`
	// rateLimitedAction controls what to do when the remote write endpoint returns HTTP 429 (Too Many Requests).
	// When omitted, no retries are performed on rate limit responses.
	// When set to "Retry", Prometheus will retry such requests using the backoff settings above.
	// Valid value when set is "Retry".
	// +optional
	RateLimitedAction RateLimitedAction `json:"rateLimitedAction,omitempty"`
}

// Sigv4 defines AWS Signature Version 4 authentication settings.
// At least one of region, accessKey/secretKey, profile, or roleArn must be set so the platform can perform authentication.
// +kubebuilder:validation:MinProperties=1
type Sigv4 struct {
	// region is the AWS region.
	// When omitted, the region is derived from the environment or instance metadata.
	// Must be between 1 and 128 characters.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	Region string `json:"region,omitempty"`
	// accessKey defines the secret reference containing the AWS access key ID.
	// The secret must exist in the openshift-monitoring namespace.
	// When omitted, the access key is derived from the environment or instance metadata.
	// +optional
	AccessKey SecretKeySelector `json:"accessKey,omitempty,omitzero"`
	// secretKey defines the secret reference containing the AWS secret access key.
	// The secret must exist in the openshift-monitoring namespace.
	// When omitted, the secret key is derived from the environment or instance metadata.
	// +optional
	SecretKey SecretKeySelector `json:"secretKey,omitempty,omitzero"`
	// profile is the named AWS profile used to authenticate.
	// When omitted, the default profile is used.
	// Must be between 1 and 128 characters.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	Profile string `json:"profile,omitempty"`
	// roleArn is the AWS Role ARN, an alternative to using AWS API keys.
	// When omitted, API keys are used for authentication.
	// Must be a valid AWS ARN format (e.g., "arn:aws:iam::123456789012:role/MyRole").
	// Must be between 1 and 512 characters.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=512
	// +kubebuilder:validation:XValidation:rule=`self.startsWith('arn:aws') && self.matches('^arn:aws(-[a-z]+)?:iam::[0-9]{12}:role/.+$')`,message="must be a valid AWS IAM role ARN (e.g., arn:aws:iam::123456789012:role/MyRole)"
	RoleArn string `json:"roleArn,omitempty"`
}

// RelabelConfig represents a relabeling rule.
type RelabelConfig struct {
	// name is a unique identifier for this relabel configuration.
	// Must contain only alphanumeric characters, hyphens, and underscores.
	// Must be between 1 and 63 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9_-]+$')",message="must contain only alphanumeric characters, hyphens, and underscores"
	Name string `json:"name,omitempty"`

	// sourceLabels specifies which label names to extract from each series for this relabeling rule.
	// The values of these labels are joined together using the configured separator,
	// and the resulting string is then matched against the regular expression.
	// If a referenced label does not exist on a series, Prometheus substitutes an empty string.
	// When omitted, the rule operates without extracting source labels (useful for actions like labelmap).
	// Minimum of 1 and maximum of 10 source labels can be specified, each between 1 and 128 characters.
	// Each entry must be unique.
	// Label names beginning with "__" (two underscores) are reserved for internal Prometheus use and are not allowed.
	// Label names SHOULD start with a letter (a-z, A-Z) or underscore (_), followed by zero or more letters, digits (0-9), or underscores for best compatibility.
	// While Prometheus supports UTF-8 characters in label names (since v3.0.0), using the recommended character set
	// ensures better compatibility with the wider ecosystem (tooling, third-party instrumentation, etc.).
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=128
	// +kubebuilder:validation:items:XValidation:rule="!self.startsWith('__')",message="label names beginning with '__' (two underscores) are reserved for internal Prometheus use and are not allowed"
	// +listType=set
	SourceLabels []string `json:"sourceLabels,omitempty"`

	// separator is the character sequence used to join source label values.
	// Common examples: ";", ",", "::", "|||".
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is ";".
	// Must be between 1 and 5 characters in length when specified.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=5
	Separator string `json:"separator,omitempty"`

	// regex is the regular expression to match against the concatenated source label values.
	// Must be a valid RE2 regular expression (https://github.com/google/re2/wiki/Syntax).
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is "(.*)" to match everything.
	// Must be between 1 and 1000 characters in length when specified.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=1000
	Regex string `json:"regex,omitempty"`

	// action defines the action to perform on the matched labels and its configuration.
	// Exactly one action-specific configuration must be specified based on the action type.
	// +required
	Action RelabelActionConfig `json:"action,omitzero"`
}

// RelabelActionConfig represents the action to perform and its configuration.
// Exactly one action-specific configuration must be specified based on the action type.
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Replace' ? has(self.replace) : !has(self.replace)",message="replace is required when type is Replace, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'HashMod' ? has(self.hashMod) : !has(self.hashMod)",message="hashMod is required when type is HashMod, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Lowercase' ? has(self.lowercase) : !has(self.lowercase)",message="lowercase is required when type is Lowercase, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'Uppercase' ? has(self.uppercase) : !has(self.uppercase)",message="uppercase is required when type is Uppercase, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'KeepEqual' ? has(self.keepEqual) : !has(self.keepEqual)",message="keepEqual is required when type is KeepEqual, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'DropEqual' ? has(self.dropEqual) : !has(self.dropEqual)",message="dropEqual is required when type is DropEqual, and forbidden otherwise"
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'LabelMap' ? has(self.labelMap) : !has(self.labelMap)",message="labelMap is required when type is LabelMap, and forbidden otherwise"
// +union
type RelabelActionConfig struct {
	// type specifies the action to perform on the matched labels.
	// Allowed values are Replace, Lowercase, Uppercase, Keep, Drop, KeepEqual, DropEqual, HashMod, LabelMap, LabelDrop, LabelKeep.
	//
	// When set to Replace, regex is matched against the concatenated source_labels; target_label is set to replacement with match group references (${1}, ${2}, ...) substituted. If regex does not match, no replacement takes place.
	//
	// When set to Lowercase, the concatenated source_labels are mapped to their lower case. Requires Prometheus >= v2.36.0.
	//
	// When set to Uppercase, the concatenated source_labels are mapped to their upper case. Requires Prometheus >= v2.36.0.
	//
	// When set to Keep, targets for which regex does not match the concatenated source_labels are dropped.
	//
	// When set to Drop, targets for which regex matches the concatenated source_labels are dropped.
	//
	// When set to KeepEqual, targets for which the concatenated source_labels do not match target_label are dropped. Requires Prometheus >= v2.41.0.
	//
	// When set to DropEqual, targets for which the concatenated source_labels do match target_label are dropped. Requires Prometheus >= v2.41.0.
	//
	// When set to HashMod, target_label is set to the modulus of a hash of the concatenated source_labels.
	//
	// When set to LabelMap, regex is matched against all source label names (not just source_labels); matching label values are copied to new names given by replacement with ${1}, ${2}, ... substituted.
	//
	// When set to LabelDrop, regex is matched against all label names; any label that matches is removed.
	//
	// When set to LabelKeep, regex is matched against all label names; any label that does not match is removed.
	// +required
	// +unionDiscriminator
	Type RelabelAction `json:"type,omitempty"`

	// replace configures the Replace action.
	// Required when type is Replace, and forbidden otherwise.
	// +unionMember
	// +optional
	Replace ReplaceActionConfig `json:"replace,omitempty,omitzero"`

	// hashMod configures the HashMod action.
	// Required when type is HashMod, and forbidden otherwise.
	// +unionMember
	// +optional
	HashMod HashModActionConfig `json:"hashMod,omitempty,omitzero"`

	// labelMap configures the LabelMap action.
	// Required when type is LabelMap, and forbidden otherwise.
	// +unionMember
	// +optional
	LabelMap LabelMapActionConfig `json:"labelMap,omitempty,omitzero"`

	// lowercase configures the Lowercase action.
	// Required when type is Lowercase, and forbidden otherwise.
	// Requires Prometheus >= v2.36.0.
	// +unionMember
	// +optional
	Lowercase LowercaseActionConfig `json:"lowercase,omitempty,omitzero"`

	// uppercase configures the Uppercase action.
	// Required when type is Uppercase, and forbidden otherwise.
	// Requires Prometheus >= v2.36.0.
	// +unionMember
	// +optional
	Uppercase UppercaseActionConfig `json:"uppercase,omitempty,omitzero"`

	// keepEqual configures the KeepEqual action.
	// Required when type is KeepEqual, and forbidden otherwise.
	// Requires Prometheus >= v2.41.0.
	// +unionMember
	// +optional
	KeepEqual KeepEqualActionConfig `json:"keepEqual,omitempty,omitzero"`

	// dropEqual configures the DropEqual action.
	// Required when type is DropEqual, and forbidden otherwise.
	// Requires Prometheus >= v2.41.0.
	// +unionMember
	// +optional
	DropEqual DropEqualActionConfig `json:"dropEqual,omitempty,omitzero"`
}

// ReplaceActionConfig configures the Replace action.
// Regex is matched against the concatenated source_labels; target_label is set to replacement with match group references (${1}, ${2}, ...) substituted. No replacement if regex does not match.
type ReplaceActionConfig struct {
	// targetLabel is the label name where the replacement result is written.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`

	// replacement is the value written to target_label when regex matches; match group references (${1}, ${2}, ...) are substituted.
	// Required when using the Replace action so the intended behavior is explicit and the platform does not need to apply defaults.
	// Use "$1" for the first capture group, "$2" for the second, etc. Use an empty string ("") to explicitly clear the target label value.
	// Must be between 0 and 255 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=0
	// +kubebuilder:validation:MaxLength=255
	Replacement *string `json:"replacement,omitempty"`
}

// HashModActionConfig configures the HashMod action.
// target_label is set to the modulus of a hash of the concatenated source_labels (target = hash % modulus).
type HashModActionConfig struct {
	// targetLabel is the label name where the hash modulus result is written.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`

	// modulus is the divisor applied to the hash of the concatenated source label values (target = hash % modulus).
	// Required when using the HashMod action so the intended behavior is explicit.
	// Must be between 1 and 1000000.
	// +required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1000000
	Modulus int64 `json:"modulus,omitempty"`
}

// LowercaseActionConfig configures the Lowercase action.
// Maps the concatenated source_labels to their lower case and writes to target_label.
// Requires Prometheus >= v2.36.0.
type LowercaseActionConfig struct {
	// targetLabel is the label name where the lower-cased value is written.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`
}

// UppercaseActionConfig configures the Uppercase action.
// Maps the concatenated source_labels to their upper case and writes to target_label.
// Requires Prometheus >= v2.36.0.
type UppercaseActionConfig struct {
	// targetLabel is the label name where the upper-cased value is written.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`
}

// KeepEqualActionConfig configures the KeepEqual action.
// Drops targets for which the concatenated source_labels do not match the value of target_label.
// Requires Prometheus >= v2.41.0.
type KeepEqualActionConfig struct {
	// targetLabel is the label name whose value is compared to the concatenated source_labels; targets that do not match are dropped.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`
}

// DropEqualActionConfig configures the DropEqual action.
// Drops targets for which the concatenated source_labels do match the value of target_label.
// Requires Prometheus >= v2.41.0.
type DropEqualActionConfig struct {
	// targetLabel is the label name whose value is compared to the concatenated source_labels; targets that match are dropped.
	// Must be between 1 and 128 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	TargetLabel string `json:"targetLabel,omitempty"`
}

// LabelMapActionConfig configures the LabelMap action.
// Regex is matched against all source label names (not just source_labels). Matching label values are copied to new label names given by replacement, with match group references (${1}, ${2}, ...) substituted.
type LabelMapActionConfig struct {
	// replacement is the template for new label names; match group references (${1}, ${2}, ...) are substituted from the matched label name.
	// Required when using the LabelMap action so the intended behavior is explicit and the platform does not need to apply defaults.
	// Use "$1" for the first capture group, "$2" for the second, etc.
	// Must be between 1 and 255 characters in length. Empty string is invalid as it would produce invalid label names.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=255
	Replacement string `json:"replacement,omitempty"`
}

// TLSConfig represents TLS configuration for Alertmanager connections.
// At least one TLS configuration option must be specified.
// For mutual TLS (mTLS), both cert and key must be specified together, or both omitted.
// +kubebuilder:validation:MinProperties=1
// +kubebuilder:validation:XValidation:rule="(has(self.cert) && has(self.key)) || (!has(self.cert) && !has(self.key))",message="cert and key must both be specified together for mutual TLS, or both be omitted"
type TLSConfig struct {
	// ca is an optional CA certificate to use for TLS connections.
	// When omitted, the system's default CA bundle is used.
	// +optional
	CA SecretKeySelector `json:"ca,omitempty,omitzero"`
	// cert is an optional client certificate to use for mutual TLS connections.
	// When omitted, no client certificate is presented.
	// +optional
	Cert SecretKeySelector `json:"cert,omitempty,omitzero"`
	// key is an optional client key to use for mutual TLS connections.
	// When omitted, no client key is used.
	// +optional
	Key SecretKeySelector `json:"key,omitempty,omitzero"`
	// serverName is an optional server name to use for TLS connections.
	// When specified, must be a valid DNS subdomain as per RFC 1123.
	// When omitted, the server name is derived from the URL.
	// Must be between 1 and 253 characters in length.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="must be a valid DNS subdomain (lowercase alphanumeric characters, '-' or '.', start and end with alphanumeric)"
	ServerName string `json:"serverName,omitempty"`
	// certificateVerification determines the policy for TLS certificate verification.
	// Allowed values are "Verify" (performs certificate verification, secure) and "SkipVerify" (skips verification, insecure).
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is "Verify".
	// +optional
	CertificateVerification CertificateVerificationType `json:"certificateVerification,omitempty"`
}

// CertificateVerificationType defines the TLS certificate verification policy.
// +kubebuilder:validation:Enum=Verify;SkipVerify
type CertificateVerificationType string

const (
	// CertificateVerificationVerify performs certificate verification (secure, recommended).
	CertificateVerificationVerify CertificateVerificationType = "Verify"
	// CertificateVerificationSkipVerify skips certificate verification (insecure, use with caution).
	CertificateVerificationSkipVerify CertificateVerificationType = "SkipVerify"
)

// AuthorizationType defines the type of authentication to use.
// +kubebuilder:validation:Enum=BearerToken
type AuthorizationType string

const (
	// AuthorizationTypeBearerToken indicates bearer token authentication.
	AuthorizationTypeBearerToken AuthorizationType = "BearerToken"
)

// AuthorizationConfig defines the authentication method for Alertmanager connections.
// +kubebuilder:validation:XValidation:rule="has(self.type) && self.type == 'BearerToken' ? has(self.bearerToken) : !has(self.bearerToken)",message="bearerToken is required when type is BearerToken"
// +union
type AuthorizationConfig struct {
	// type specifies the authentication type to use.
	// Valid value is "BearerToken" (bearer token authentication).
	// When set to BearerToken, the bearerToken field must be specified.
	// +unionDiscriminator
	// +required
	Type AuthorizationType `json:"type,omitempty"`
	// bearerToken defines the secret reference containing the bearer token.
	// Required when type is "BearerToken", and forbidden otherwise.
	// The secret must exist in the openshift-monitoring namespace.
	// +optional
	BearerToken SecretKeySelector `json:"bearerToken,omitempty,omitzero"`
}

// SecretKeySelector selects a key of a Secret in the `openshift-monitoring` namespace.
// +structType=atomic
type SecretKeySelector struct {
	// name is the name of the secret in the `openshift-monitoring` namespace to select from.
	// Must be a valid Kubernetes secret name (lowercase alphanumeric, '-' or '.', start/end with alphanumeric).
	// Must be between 1 and 253 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="!format.dns1123Subdomain().validate(self).hasValue()",message="must be a valid secret name (lowercase alphanumeric characters, '-' or '.', start and end with alphanumeric)"
	Name string `json:"name,omitempty"`
	// key is the key of the secret to select from.
	// Must consist of alphanumeric characters, '-', '_', or '.'.
	// Must be between 1 and 253 characters in length.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:XValidation:rule="self.matches('^[a-zA-Z0-9._-]+$')",message="must contain only alphanumeric characters, '-', '_', or '.'"
	Key string `json:"key,omitempty"`
}

// Retention configures how long Prometheus retains metrics data and how much storage it can use.
// +kubebuilder:validation:MinProperties=1
type Retention struct {
	// durationInDays specifies how many days Prometheus will retain metrics data.
	// Prometheus automatically deletes data older than this duration.
	// When omitted, this means no opinion and the platform is left to choose a reasonable default, which is subject to change over time.
	// The default value is 15.
	// Minimum value is 1 day.
	// Maximum value is 365 days (1 year).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=365
	// +optional
	DurationInDays int32 `json:"durationInDays,omitempty"`
	// sizeInGiB specifies the maximum storage size in gibibytes (GiB) that Prometheus
	// can use for data blocks and the write-ahead log (WAL).
	// When the limit is reached, Prometheus will delete oldest data first.
	// When omitted, no size limit is enforced and Prometheus uses available PersistentVolume capacity.
	// Minimum value is 1 GiB.
	// Maximum value is 16384 GiB (16 TiB).
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=16384
	// +optional
	SizeInGiB int32 `json:"sizeInGiB,omitempty"`
}

// RelabelAction defines the action to perform in a relabeling rule.
// +kubebuilder:validation:Enum=Replace;Keep;Drop;HashMod;LabelMap;LabelDrop;LabelKeep;Lowercase;Uppercase;KeepEqual;DropEqual
type RelabelAction string

const (
	// RelabelActionReplace: match regex against concatenated source_labels; set target_label to replacement with ${1}, ${2}, ... substituted. No replacement if regex does not match.
	RelabelActionReplace RelabelAction = "Replace"
	// RelabelActionLowercase: map the concatenated source_labels to their lower case.
	RelabelActionLowercase RelabelAction = "Lowercase"
	// RelabelActionUppercase: map the concatenated source_labels to their upper case.
	RelabelActionUppercase RelabelAction = "Uppercase"
	// RelabelActionKeep: drop targets for which regex does not match the concatenated source_labels.
	RelabelActionKeep RelabelAction = "Keep"
	// RelabelActionDrop: drop targets for which regex matches the concatenated source_labels.
	RelabelActionDrop RelabelAction = "Drop"
	// RelabelActionKeepEqual: drop targets for which the concatenated source_labels do not match target_label.
	RelabelActionKeepEqual RelabelAction = "KeepEqual"
	// RelabelActionDropEqual: drop targets for which the concatenated source_labels do match target_label.
	RelabelActionDropEqual RelabelAction = "DropEqual"
	// RelabelActionHashMod: set target_label to the modulus of a hash of the concatenated source_labels.
	RelabelActionHashMod RelabelAction = "HashMod"
	// RelabelActionLabelMap: match regex against all source label names; copy matching label values to new names given by replacement with ${1}, ${2}, ... substituted.
	RelabelActionLabelMap RelabelAction = "LabelMap"
	// RelabelActionLabelDrop: match regex against all label names; any label that matches is removed.
	RelabelActionLabelDrop RelabelAction = "LabelDrop"
	// RelabelActionLabelKeep: match regex against all label names; any label that does not match is removed.
	RelabelActionLabelKeep RelabelAction = "LabelKeep"
)

// CollectionProfile defines the metrics collection profile for Prometheus.
// +kubebuilder:validation:Enum=Full;Minimal
type CollectionProfile string

const (
	// CollectionProfileFull means Prometheus collects all metrics that are exposed by the platform components.
	CollectionProfileFull CollectionProfile = "Full"
	// CollectionProfileMinimal means Prometheus only collects metrics necessary for the default
	// platform alerts, recording rules, telemetry and console dashboards.
	CollectionProfileMinimal CollectionProfile = "Minimal"
)

// TelemeterClientConfig provides configuration options for the Telemeter Client component
// that runs in the `openshift-monitoring` namespace. The Telemeter Client collects selected
// monitoring metrics and forwards them to Red Hat for telemetry purposes.
// At least one field must be specified.
// +kubebuilder:validation:MinProperties=1
type TelemeterClientConfig struct {
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the Telemeter Client container.
	// This includes CPU, memory and HugePages constraints to help control scheduling and resource usage.
	// When not specified, defaults are used by the platform. Requests cannot exceed limits.
	// This field is optional.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 1m
	//      limit: null
	//    - name: memory
	//      request: 40Mi
	//      limit: null
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how Telemeter Client Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Default is empty list.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// ThanosQuerierConfig provides configuration options for the Thanos Querier component
// that runs in the `openshift-monitoring` namespace.
// At least one field must be specified; an empty thanosQuerierConfig object is not allowed.
// +kubebuilder:validation:MinProperties=1
type ThanosQuerierConfig struct {
	// nodeSelector defines the nodes on which the Pods are scheduled.
	// nodeSelector is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default value is `kubernetes.io/os: linux`.
	// When specified, nodeSelector must contain at least 1 entry and must not contain more than 10 entries.
	// +optional
	// +kubebuilder:validation:MinProperties=1
	// +kubebuilder:validation:MaxProperties=10
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// resources defines the compute resource requests and limits for the Thanos Querier container.
	// resources is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Requests cannot exceed limits.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
	// This is a simplified API that maps to Kubernetes ResourceRequirements.
	// The current default values are:
	//   resources:
	//    - name: cpu
	//      request: 5m
	//    - name: memory
	//      request: 12Mi
	// Maximum length for this list is 5.
	// Minimum length for this list is 1.
	// Each resource name must be unique within this list.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=5
	// +kubebuilder:validation:MinItems=1
	Resources []ContainerResource `json:"resources,omitempty"`
	// tolerations defines tolerations for the pods.
	// tolerations is optional.
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=atomic
	// +optional
	Tolerations []v1.Toleration `json:"tolerations,omitempty"`
	// topologySpreadConstraints defines rules for how Thanos Querier Pods should be distributed
	// across topology domains such as zones, nodes, or other user-defined labels.
	// topologySpreadConstraints is optional.
	// This helps improve high availability and resource efficiency by avoiding placing
	// too many replicas in the same failure domain.
	//
	// When omitted, this means no opinion and the platform is left to choose a default, which is subject to change over time.
	// This field maps directly to the `topologySpreadConstraints` field in the Pod spec.
	// Defaults are empty/unset.
	// Maximum length for this list is 10.
	// Minimum length for this list is 1.
	// Entries must have unique topologyKey and whenUnsatisfiable pairs.
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:MinItems=1
	// +listType=map
	// +listMapKey=topologyKey
	// +listMapKey=whenUnsatisfiable
	// +optional
	TopologySpreadConstraints []v1.TopologySpreadConstraint `json:"topologySpreadConstraints,omitempty"`
}

// AuditProfile defines the audit log level for the Metrics Server.
// +kubebuilder:validation:Enum=None;Metadata;Request;RequestResponse
type AuditProfile string

const (
	// AuditProfileNone disables audit logging
	AuditProfileNone AuditProfile = "None"
	// AuditProfileMetadata logs request metadata (requesting user, timestamp, resource, verb, etc.) but not request or response body
	AuditProfileMetadata AuditProfile = "Metadata"
	// AuditProfileRequest logs event metadata and request body but not response body
	AuditProfileRequest AuditProfile = "Request"
	// AuditProfileRequestResponse logs event metadata, request and response bodies
	AuditProfileRequestResponse AuditProfile = "RequestResponse"
)

// VerbosityLevel defines the verbosity of log messages for Metrics Server.
// +kubebuilder:validation:Enum=Errors;Info;Trace;TraceAll
type VerbosityLevel string

const (
	// VerbosityLevelErrors means only critical messages and errors are logged.
	VerbosityLevelErrors VerbosityLevel = "Errors"
	// VerbosityLevelInfo means basic informational messages are logged.
	VerbosityLevelInfo VerbosityLevel = "Info"
	// VerbosityLevelTrace means extended information useful for general debugging is logged.
	VerbosityLevelTrace VerbosityLevel = "Trace"
	// VerbosityLevelTraceAll means detailed information about metric scraping operations is logged.
	VerbosityLevelTraceAll VerbosityLevel = "TraceAll"
)

// ExemplarsMode defines whether exemplars are sent via remote write.
// +kubebuilder:validation:Enum=Send;DoNotSend
type ExemplarsMode string

const (
	// ExemplarsModeSend means exemplars are sent via remote write.
	ExemplarsModeSend ExemplarsMode = "Send"
	// ExemplarsModeDoNotSend means exemplars are not sent via remote write.
	ExemplarsModeDoNotSend ExemplarsMode = "DoNotSend"
)

// RateLimitedAction defines what to do when the remote write endpoint returns HTTP 429 (Too Many Requests).
// Omission of this field means do not retry. When set, the only valid value is Retry.
// +kubebuilder:validation:Enum=Retry
type RateLimitedAction string

const (
	// RateLimitedActionRetry means requests will be retried on HTTP 429 responses.
	RateLimitedActionRetry RateLimitedAction = "Retry"
)

// Audit profile configurations
type Audit struct {
	// profile is a required field for configuring the audit log level of the Kubernetes Metrics Server.
	// Allowed values are None, Metadata, Request, or RequestResponse.
	// When set to None, audit logging is disabled and no audit events are recorded.
	// When set to Metadata, only request metadata (such as requesting user, timestamp, resource, verb, etc.) is logged, but not the request or response body.
	// When set to Request, event metadata and the request body are logged, but not the response body.
	// When set to RequestResponse, event metadata, request body, and response body are all logged, providing the most detailed audit information.
	//
	// See: https://kubernetes.io/docs/tasks/debug-application-cluster/audit/#audit-policy
	// for more information about auditing and log levels.
	// +required
	Profile AuditProfile `json:"profile,omitempty"`
}
