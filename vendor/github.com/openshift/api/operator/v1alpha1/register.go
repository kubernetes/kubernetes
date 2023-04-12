package v1alpha1

import (
	configv1 "github.com/openshift/api/config/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	GroupName     = "operator.openshift.io"
	GroupVersion  = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}
	schemeBuilder = runtime.NewSchemeBuilder(addKnownTypes, configv1.Install)
	// Install is a function which adds this version to a scheme
	Install = schemeBuilder.AddToScheme

	// SchemeGroupVersion generated code relies on this name
	// Deprecated
	SchemeGroupVersion = GroupVersion
	// AddToScheme exists solely to keep the old generators creating valid code
	// DEPRECATED
	AddToScheme = schemeBuilder.AddToScheme
)

// Resource generated code relies on this being here, but it logically belongs to the group
// DEPRECATED
func Resource(resource string) schema.GroupResource {
	return schema.GroupResource{Group: GroupName, Resource: resource}
}

func addKnownTypes(scheme *runtime.Scheme) error {
	metav1.AddToGroupVersion(scheme, GroupVersion)

	scheme.AddKnownTypes(GroupVersion,
		&GenericOperatorConfig{},
		&ImageContentSourcePolicy{},
		&ImageContentSourcePolicyList{},
	)

	return nil
}
