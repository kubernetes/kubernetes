package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	intstr "k8s.io/apimachinery/pkg/util/intstr"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourceTopology is a specification for a Foo resource
type NodeResourceTopology struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	TopologyPolicy []string `json:"topologyPolicies"`
	Zones          ZoneMap  `json:"zones"`
}

// Zone is the spec for a NodeResourceTopology resource
type Zone struct {
	Type       string          `json:"type"`
	Parent     string          `json:"parent,omitempty"`
	Costs      map[string]int  `json:"costs,omitempty"`
	Attributes map[string]int  `json:"attributes,omitempty"`
	Resources  ResourceInfoMap `json:"resources,omitempty"`
}

type ZoneMap map[string]Zone
type ResourceInfoMap map[string]ResourceInfo

type ResourceInfo struct {
	Allocatable intstr.IntOrString `json:"allocatable"`
	Capacity    intstr.IntOrString `json:"capacity"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeResourceTopologyList is a list of NodeResourceTopology resources
type NodeResourceTopologyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []NodeResourceTopology `json:"items"`
}
