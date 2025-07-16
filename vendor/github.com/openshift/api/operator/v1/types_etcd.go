package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=etcds,scope=Cluster,categories=coreoperators
// +kubebuilder:subresource:status
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/752
// +openshift:file-pattern=cvoRunLevel=0000_12,operatorName=etcd,operatorOrdering=01

// Etcd provides information to configure an operator to manage etcd.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Etcd struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// +required
	Spec EtcdSpec `json:"spec"`
	// +optional
	Status EtcdStatus `json:"status"`
}

type EtcdSpec struct {
	StaticPodOperatorSpec `json:",inline"`
	// HardwareSpeed allows user to change the etcd tuning profile which configures
	// the latency parameters for heartbeat interval and leader election timeouts
	// allowing the cluster to tolerate longer round-trip-times between etcd members.
	// Valid values are "", "Standard" and "Slower".
	//	"" means no opinion and the platform is left to choose a reasonable default
	//	which is subject to change without notice.
	// +openshift:enable:FeatureGate=HardwareSpeed
	// +optional
	HardwareSpeed ControlPlaneHardwareSpeed `json:"controlPlaneHardwareSpeed"`

	// backendQuotaGiB sets the etcd backend storage size limit in gibibytes.
	// The value should be an integer not less than 8 and not more than 32.
	// When not specified, the default value is 8.
	// +kubebuilder:default:=8
	// +kubebuilder:validation:Minimum=8
	// +kubebuilder:validation:Maximum=32
	// +kubebuilder:validation:XValidation:rule="self>=oldSelf",message="etcd backendQuotaGiB may not be decreased"
	// +openshift:enable:FeatureGate=EtcdBackendQuota
	// +default=8
	// +optional
	BackendQuotaGiB int32 `json:"backendQuotaGiB,omitempty"`
}

type EtcdStatus struct {
	StaticPodOperatorStatus `json:",inline"`
	// +optional
	HardwareSpeed ControlPlaneHardwareSpeed `json:"controlPlaneHardwareSpeed"`
}

const (
	// StandardHardwareSpeed provides the normal tolerances for hardware speed and latency.
	//	Currently sets (values subject to change at any time):
	//		ETCD_HEARTBEAT_INTERVAL: 100ms
	// 	ETCD_LEADER_ELECTION_TIMEOUT: 1000ms
	StandardHardwareSpeed ControlPlaneHardwareSpeed = "Standard"
	// SlowerHardwareSpeed provides more tolerance for slower hardware and/or higher latency networks.
	// Sets (values subject to change):
	//		ETCD_HEARTBEAT_INTERVAL: 5x Standard
	// 	ETCD_LEADER_ELECTION_TIMEOUT: 2.5x Standard
	SlowerHardwareSpeed ControlPlaneHardwareSpeed = "Slower"
)

// ControlPlaneHardwareSpeed declares valid hardware speed tolerance levels
// +enum
// +kubebuilder:validation:Enum:="";Standard;Slower
type ControlPlaneHardwareSpeed string

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeAPISOperatorConfigList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type EtcdList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// items contains the items
	Items []Etcd `json:"items"`
}
