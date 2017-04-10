package benchmark

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
		"k8s.io/kubernetes/pkg/api/v1"
)

// High Level Configuration for all predicates and priorities.
type schedulerPerfConfig struct {
	NodeAffinity nodeAffinity
	InterpodAffinity interpodAffinity
}

// interpodAffinity priority configuration details.
type interpodAffinity struct {
	Enabled     bool
	Operator    metav1.LabelSelectorOperator
	affinityKey string
	Labels      map[string]string
	TopologyKey string
}

// nodeAffinity priority configuration details.
type nodeAffinity struct {
	Enabled         bool   //If not enabled, node affinity is disabled.
	numGroups       int    // the % of nodes and pods that should match. Higher # -> smaller performance deficit at scale.
	nodeAffinityKey string // the number of labels needed to match.  Higher # -> larger performance deficit at scale.
	Operator        v1.NodeSelectorOperator
}

