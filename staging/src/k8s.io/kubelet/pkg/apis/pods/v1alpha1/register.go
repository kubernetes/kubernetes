package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	// SchemeBuilder is the scheme builder for the pods API.
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	// AddToScheme adds the pods API to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)

// GroupName is the group name for the pods API.
const GroupName = "pods.kubelet.k8s.io"

// SchemeGroupVersion is the group version for the pods API.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

// addKnownTypes adds the known types to the given scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion)
	return nil
}
