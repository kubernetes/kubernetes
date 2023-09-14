package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Etcd provides information to configure an operator to manage etcd.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Etcd struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	// +kubebuilder:validation:Required
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
	// +kubebuilder:validation:Optional
	// +openshift:enable:FeatureSets=CustomNoUpgrade;TechPreviewNoUpgrade
	// +optional
	HardwareSpeed ControlPlaneHardwareSpeed `json:"controlPlaneHardwareSpeed"`
}

type EtcdStatus struct {
	StaticPodOperatorStatus `json:",inline"`
	HardwareSpeed           ControlPlaneHardwareSpeed `json:"controlPlaneHardwareSpeed"`
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

	// Items contains the items
	Items []Etcd `json:"items"`
}
