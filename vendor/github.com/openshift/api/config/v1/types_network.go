package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Network holds cluster-wide information about Network. The canonical name is `cluster`. It is used to configure the desired network configuration, such as: IP address pools for services/pod IPs, network plugin, etc.
// Please view network.spec for an explanation on what applies when configuring this resource.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/470
// +openshift:compatibility-gen:level=1
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=config-operator,operatorOrdering=01
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=networks,scope=Cluster
type Network struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration.
	// As a general rule, this SHOULD NOT be read directly. Instead, you should
	// consume the NetworkStatus, as it indicates the currently deployed configuration.
	// Currently, most spec fields are immutable after installation. Please view the individual ones for further details on each.
	// +kubebuilder:validation:Required
	// +required
	Spec NetworkSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status NetworkStatus `json:"status"`
}

// NetworkSpec is the desired network configuration.
// As a general rule, this SHOULD NOT be read directly. Instead, you should
// consume the NetworkStatus, as it indicates the currently deployed configuration.
// Currently, most spec fields are immutable after installation. Please view the individual ones for further details on each.
// +openshift:validation:FeatureGateAwareXValidation:featureGate=NetworkDiagnosticsConfig,rule="!has(self.networkDiagnostics) || !has(self.networkDiagnostics.mode) || self.networkDiagnostics.mode!='Disabled' || !has(self.networkDiagnostics.sourcePlacement) && !has(self.networkDiagnostics.targetPlacement)",message="cannot set networkDiagnostics.sourcePlacement and networkDiagnostics.targetPlacement when networkDiagnostics.mode is Disabled"
type NetworkSpec struct {
	// IP address pool to use for pod IPs.
	// This field is immutable after installation.
	ClusterNetwork []ClusterNetworkEntry `json:"clusterNetwork"`

	// IP address pool for services.
	// Currently, we only support a single entry here.
	// This field is immutable after installation.
	ServiceNetwork []string `json:"serviceNetwork"`

	// NetworkType is the plugin that is to be deployed (e.g. OpenShiftSDN).
	// This should match a value that the cluster-network-operator understands,
	// or else no networking will be installed.
	// Currently supported values are:
	// - OpenShiftSDN
	// This field is immutable after installation.
	NetworkType string `json:"networkType"`

	// externalIP defines configuration for controllers that
	// affect Service.ExternalIP. If nil, then ExternalIP is
	// not allowed to be set.
	// +optional
	ExternalIP *ExternalIPConfig `json:"externalIP,omitempty"`

	// The port range allowed for Services of type NodePort.
	// If not specified, the default of 30000-32767 will be used.
	// Such Services without a NodePort specified will have one
	// automatically allocated from this range.
	// This parameter can be updated after the cluster is
	// installed.
	// +kubebuilder:validation:Pattern=`^([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])-([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$`
	ServiceNodePortRange string `json:"serviceNodePortRange,omitempty"`

	// networkDiagnostics defines network diagnostics configuration.
	//
	// Takes precedence over spec.disableNetworkDiagnostics in network.operator.openshift.io.
	// If networkDiagnostics is not specified or is empty,
	// and the spec.disableNetworkDiagnostics flag in network.operator.openshift.io is set to true,
	// the network diagnostics feature will be disabled.
	//
	// +optional
	// +openshift:enable:FeatureGate=NetworkDiagnosticsConfig
	NetworkDiagnostics NetworkDiagnostics `json:"networkDiagnostics"`
}

// NetworkStatus is the current network configuration.
type NetworkStatus struct {
	// IP address pool to use for pod IPs.
	ClusterNetwork []ClusterNetworkEntry `json:"clusterNetwork,omitempty"`

	// IP address pool for services.
	// Currently, we only support a single entry here.
	ServiceNetwork []string `json:"serviceNetwork,omitempty"`

	// NetworkType is the plugin that is deployed (e.g. OpenShiftSDN).
	NetworkType string `json:"networkType,omitempty"`

	// ClusterNetworkMTU is the MTU for inter-pod networking.
	ClusterNetworkMTU int `json:"clusterNetworkMTU,omitempty"`

	// Migration contains the cluster network migration configuration.
	Migration *NetworkMigration `json:"migration,omitempty"`

	// conditions represents the observations of a network.config current state.
	// Known .status.conditions.type are: "NetworkTypeMigrationInProgress", "NetworkTypeMigrationMTUReady",
	// "NetworkTypeMigrationTargetCNIAvailable", "NetworkTypeMigrationTargetCNIInUse",
	// "NetworkTypeMigrationOriginalCNIPurged" and "NetworkDiagnosticsAvailable"
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +openshift:enable:FeatureGate=NetworkLiveMigration
	// +openshift:enable:FeatureGate=NetworkDiagnosticsConfig
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// ClusterNetworkEntry is a contiguous block of IP addresses from which pod IPs
// are allocated.
type ClusterNetworkEntry struct {
	// The complete block for pod IPs.
	CIDR string `json:"cidr"`

	// The size (prefix) of block to allocate to each node. If this
	// field is not used by the plugin, it can be left unset.
	// +kubebuilder:validation:Minimum=0
	// +optional
	HostPrefix uint32 `json:"hostPrefix,omitempty"`
}

// ExternalIPConfig specifies some IP blocks relevant for the ExternalIP field
// of a Service resource.
type ExternalIPConfig struct {
	// policy is a set of restrictions applied to the ExternalIP field.
	// If nil or empty, then ExternalIP is not allowed to be set.
	// +optional
	Policy *ExternalIPPolicy `json:"policy,omitempty"`

	// autoAssignCIDRs is a list of CIDRs from which to automatically assign
	// Service.ExternalIP. These are assigned when the service is of type
	// LoadBalancer. In general, this is only useful for bare-metal clusters.
	// In Openshift 3.x, this was misleadingly called "IngressIPs".
	// Automatically assigned External IPs are not affected by any
	// ExternalIPPolicy rules.
	// Currently, only one entry may be provided.
	// +optional
	AutoAssignCIDRs []string `json:"autoAssignCIDRs,omitempty"`
}

// ExternalIPPolicy configures exactly which IPs are allowed for the ExternalIP
// field in a Service. If the zero struct is supplied, then none are permitted.
// The policy controller always allows automatically assigned external IPs.
type ExternalIPPolicy struct {
	// allowedCIDRs is the list of allowed CIDRs.
	AllowedCIDRs []string `json:"allowedCIDRs,omitempty"`

	// rejectedCIDRs is the list of disallowed CIDRs. These take precedence
	// over allowedCIDRs.
	// +optional
	RejectedCIDRs []string `json:"rejectedCIDRs,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type NetworkList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	Items []Network `json:"items"`
}

// NetworkMigration represents the cluster network configuration.
type NetworkMigration struct {
	// NetworkType is the target plugin that is to be deployed.
	// Currently supported values are: OpenShiftSDN, OVNKubernetes
	// +kubebuilder:validation:Enum={"OpenShiftSDN","OVNKubernetes"}
	// +optional
	NetworkType string `json:"networkType,omitempty"`

	// MTU contains the MTU migration configuration.
	// +optional
	MTU *MTUMigration `json:"mtu,omitempty"`
}

// MTUMigration contains infomation about MTU migration.
type MTUMigration struct {
	// Network contains MTU migration configuration for the default network.
	// +optional
	Network *MTUMigrationValues `json:"network,omitempty"`

	// Machine contains MTU migration configuration for the machine's uplink.
	// +optional
	Machine *MTUMigrationValues `json:"machine,omitempty"`
}

// MTUMigrationValues contains the values for a MTU migration.
type MTUMigrationValues struct {
	// To is the MTU to migrate to.
	// +kubebuilder:validation:Minimum=0
	To *uint32 `json:"to"`

	// From is the MTU to migrate from.
	// +kubebuilder:validation:Minimum=0
	// +optional
	From *uint32 `json:"from,omitempty"`
}

// NetworkDiagnosticsMode is an enumeration of the available network diagnostics modes
// Valid values are "", "All", "Disabled".
// +kubebuilder:validation:Enum:="";All;Disabled
type NetworkDiagnosticsMode string

const (
	// NetworkDiagnosticsNoOpinion means that the user has no opinion and the platform is left
	// to choose reasonable default. The current default is All and is a subject to change over time.
	NetworkDiagnosticsNoOpinion NetworkDiagnosticsMode = ""
	// NetworkDiagnosticsAll means that all network diagnostics checks are enabled
	NetworkDiagnosticsAll NetworkDiagnosticsMode = "All"
	// NetworkDiagnosticsDisabled means that network diagnostics is disabled
	NetworkDiagnosticsDisabled NetworkDiagnosticsMode = "Disabled"
)

// NetworkDiagnostics defines network diagnostics configuration

type NetworkDiagnostics struct {
	// mode controls the network diagnostics mode
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is All.
	//
	// +optional
	Mode NetworkDiagnosticsMode `json:"mode"`

	// sourcePlacement controls the scheduling of network diagnostics source deployment
	//
	// See NetworkDiagnosticsSourcePlacement for more details about default values.
	//
	// +optional
	SourcePlacement NetworkDiagnosticsSourcePlacement `json:"sourcePlacement"`

	// targetPlacement controls the scheduling of network diagnostics target daemonset
	//
	// See NetworkDiagnosticsTargetPlacement for more details about default values.
	//
	// +optional
	TargetPlacement NetworkDiagnosticsTargetPlacement `json:"targetPlacement"`
}

// NetworkDiagnosticsSourcePlacement defines node scheduling configuration network diagnostics source components
type NetworkDiagnosticsSourcePlacement struct {
	// nodeSelector is the node selector applied to network diagnostics components
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is `kubernetes.io/os: linux`.
	//
	// +optional
	NodeSelector map[string]string `json:"nodeSelector"`

	// tolerations is a list of tolerations applied to network diagnostics components
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is an empty list.
	//
	// +optional
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations"`
}

// NetworkDiagnosticsTargetPlacement defines node scheduling configuration network diagnostics target components
type NetworkDiagnosticsTargetPlacement struct {
	// nodeSelector is the node selector applied to network diagnostics components
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is `kubernetes.io/os: linux`.
	//
	// +optional
	NodeSelector map[string]string `json:"nodeSelector"`

	// tolerations is a list of tolerations applied to network diagnostics components
	//
	// When omitted, this means the user has no opinion and the platform is left
	// to choose reasonable defaults. These defaults are subject to change over time.
	// The current default is `- operator: "Exists"` which means that all taints are tolerated.
	//
	// +optional
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations"`
}
