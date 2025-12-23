package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

type PodNetworkHealthSpec struct {
	SourcePod string `json:"sourcePod"`
	TargetPod string `json:"targetPod"`
}

type PodNetworkHealthStatus struct {
	Reachable bool `json:"reachable,omitempty"`
	LatencyMs int64 `json:"latencyMs,omitempty"`
	LastCheck metav1.Time `json:"lastCheck,omitempty"`
}

type PodNetworkHealth struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   PodNetworkHealthSpec   `json:"spec,omitempty"`
	Status PodNetworkHealthStatus `json:"status,omitempty"`
}
