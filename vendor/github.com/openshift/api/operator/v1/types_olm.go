package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OLM provides information to configure an operator to manage the OLM controllers
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
// +kubebuilder:object:root=true
// +kubebuilder:resource:path=olms,scope=Cluster
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations=include.release.openshift.io/ibm-cloud-managed=false
// +kubebuilder:metadata:annotations=include.release.openshift.io/self-managed-high-availability=true
// +openshift:api-approved.openshift.io=https://github.com/openshift/api/pull/1504
// +openshift:file-pattern=cvoRunLevel=0000_10,operatorName=operator-lifecycle-manager,operatorOrdering=01
// +openshift:enable:FeatureGate=NewOLM
// +openshift:capability=OperatorLifecycleManagerV1
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'cluster'",message="olm is a singleton, .metadata.name must be 'cluster'"
type OLM struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata"`

	//spec holds user settable values for configuration
	//+kubebuilder:validation:Required
	Spec OLMSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status OLMStatus `json:"status"`
}

type OLMSpec struct {
	OperatorSpec `json:",inline"`
}

type OLMStatus struct {
	OperatorStatus `json:",inline"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// OLMList is a collection of items
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type OLMList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata"`

	// items contains the items
	Items []OLM `json:"items"`
}
