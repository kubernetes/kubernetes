package benchmark

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
		"k8s.io/kubernetes/pkg/api/v1"
)

// High Level Configuration for all predicates and priorities.
type schedulerPerfConfig struct {
	NodeAffinity *nodeAffinity
	InterPodAffinity *interPodAffinity
}

// interPodAffinity priority configuration details.
type interPodAffinity struct {
	labelSelectorOperator    metav1.LabelSelectorOperator	// LabelSelector operator like IN, EXISTS etc.
	affinityKey string				// Selectors key
	Labels      map[string]string			// Labels on pods.
	TopologyKey string				// Topology Key
}

// nodeAffinity priority configuration details.
type nodeAffinity struct {
	numGroups       int				// number of Node-Pod sets with Pods NodeAffinity matching given Nodes.
	nodeAffinityKey string				// Node Selection Key.
	Operator        v1.NodeSelectorOperator		// Operator like IN, EXISTS etc.
}

