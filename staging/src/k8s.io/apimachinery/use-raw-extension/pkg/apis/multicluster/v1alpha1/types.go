package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// WorkSpec defines the desired state of Work
type WorkSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Workload represents the manifest workload to be deployed on spoke cluster
	Workload WorkloadTemplate `json:"workload,omitempty"`
}

// WorkloadTemplate represents the manifest workload to be deployed on spoke cluster
type WorkloadTemplate struct {
	// Manifests represents a list of kuberenetes resources to be deployed on the spoke cluster.
	// +optional
	Manifests []runtime.RawExtension `json:"manifests,omitempty" protobuf:"bytes,2,rep,name=manifests"`
}

// WorkStatus defines the observed state of Work
type WorkStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Conditions contains the different condition statuses for this work.
	// Valid condition types are:
	// 1. Applied represents workload in Work is applied successfully on spoke cluster.
	// 2. Progressing represents workload in Work is being applied on spoke cluster.
	// 3. Available represents workload in Work exists on the spoke cluster.
	// 4. Degraded represents the current state of workload does not match the desired
	// state for a certain period.
	Conditions []StatusCondition `json:"conditions"`

	// ManifestConditions represents the conditions of each resource in work deployed on
	// spoke cluster.
	// +optional
	ManifestConditions []ManifestCondition `json:"manifestConditions,omitempty"`
}

// ResourceIdentifier provides the identifiers needed to interact with any arbitrary object.
type ResourceIdentifier struct {
	// Ordinal represents an index in manifests list, so the condition can still be linked
	// to a manifest even thougth manifest cannot be parsed successfully.
	Ordinal int `json:"ordinal,omitempty"`

	// Group is the group of the resource.
	Group string `json:"group,omitempty"`

	// Version is the version of the resource.
	Version string `json:"version,omitempty"`

	// Kind is the kind of the resource.
	Kind string `json:"kind,omitempty"`

	// Namespace is the namespace of the resource, the resource is cluster scoped if the value
	// is empty
	Namespace string `json:"namespace,omitempty"`

	// Name is the name of the resource
	Name string `json:"name,omitempty"`
}

// ManifestCondition represents the conditions of the resources deployed on
// spoke cluster
type ManifestCondition struct {
	// resourceId represents a identity of a resource linking to manifests in spec.
	// +required
	Identifier ResourceIdentifier `json:"identifier,omitempty"`

	// Conditions represents the conditions of this resource on spoke cluster
	// +required
	Conditions []StatusCondition `json:"conditions"`
}

// StatusCondition contains condition information for a work.
type StatusCondition struct {
	// Type is the type of the spoke work condition.
	// +required
	Type string `json:"type" protobuf:"bytes,1,opt,name=type"`

	// Status is the status of the condition. One of True, False, Unknown.
	// +required
	Status metav1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/apimachinery/pkg/apis/meta/v1.ConditionStatus"`

	// LastTransitionTime is the last time the condition changed from one status to another.
	// +required
	LastTransitionTime metav1.Time `json:"lastTransitionTime" protobuf:"bytes,3,opt,name=lastTransitionTime"`

	// Reason is a (brief) reason for the condition's last status change.
	// +required
	Reason string `json:"reason" protobuf:"bytes,4,opt,name=reason"`

	// Message is a human-readable message indicating details about the last status change.
	// +required
	Message string `json:"message" protobuf:"bytes,5,opt,name=message"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// Work is the Schema for the works API
type Work struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   WorkSpec   `json:"spec,omitempty"`
	Status WorkStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// WorkList contains a list of Work
type WorkList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Work `json:"items"`
}
