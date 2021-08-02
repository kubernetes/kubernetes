package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Project holds cluster-wide information about Project.  The canonical name is `cluster`
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Project struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec holds user settable values for configuration
	// +kubebuilder:validation:Required
	// +required
	Spec ProjectSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status ProjectStatus `json:"status"`
}

// TemplateReference references a template in a specific namespace.
// The namespace must be specified at the point of use.
type TemplateReference struct {
	// name is the metadata.name of the referenced project request template
	Name string `json:"name"`
}

// ProjectSpec holds the project creation configuration.
type ProjectSpec struct {
	// projectRequestMessage is the string presented to a user if they are unable to request a project via the projectrequest api endpoint
	// +optional
	ProjectRequestMessage string `json:"projectRequestMessage"`

	// projectRequestTemplate is the template to use for creating projects in response to projectrequest.
	// This must point to a template in 'openshift-config' namespace. It is optional.
	// If it is not specified, a default template is used.
	//
	// +optional
	ProjectRequestTemplate TemplateReference `json:"projectRequestTemplate"`
}

type ProjectStatus struct {
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ProjectList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Project `json:"items"`
}
