package podnodeconstraints

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodNodeConstraintsConfig is the configuration for the pod node name
// and node selector constraint plug-in. For accounts, serviceaccounts,
// and groups which lack the "pods/binding" permission, Loading this
// plugin will prevent setting NodeName on pod specs and will prevent
// setting NodeSelectors whose labels appear in the blacklist field
// "NodeSelectorLabelBlacklist"
type PodNodeConstraintsConfig struct {
	metav1.TypeMeta
	// NodeSelectorLabelBlacklist specifies a list of labels which cannot be set by entities without the "pods/binding" permission
	NodeSelectorLabelBlacklist []string
}
