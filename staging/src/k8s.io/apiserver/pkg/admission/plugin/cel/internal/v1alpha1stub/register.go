package v1alpha1stub

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the group name for this API.
const GroupName = "admissionregistration.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

// Resource takes an unqualified resource and returns a Group qualified GroupResource
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

// TODO: move SchemeBuilder with zz_generated.deepcopy.go to k8s.io/api.
// localSchemeBuilder and AddToScheme will stay in k8s.io/kubernetes.
var (
	// SchemeBuilder points to a list of functions added to Scheme.
	SchemeBuilder      = runtime.NewSchemeBuilder(addKnownTypes)
	localSchemeBuilder = &SchemeBuilder
	// AddToScheme is a common registration function for mapping packaged scoped group & version keys to a scheme.
	AddToScheme = localSchemeBuilder.AddToScheme
)

// Adds the list of known types to scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&ValidatingAdmissionPolicy{},
		&ValidatingAdmissionPolicyList{},
		&ValidatingAdmissionPolicyBinding{},
		&ValidatingAdmissionPolicyBindingList{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
