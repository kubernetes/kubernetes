package docker10

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	GroupName       = "image.openshift.io"
	LegacyGroupName = ""
)

// SchemeGroupVersion is group version used to register these objects
var (
	SchemeGroupVersion       = schema.GroupVersion{Group: GroupName, Version: "1.0"}
	LegacySchemeGroupVersion = schema.GroupVersion{Group: LegacyGroupName, Version: "1.0"}

	SchemeBuilder       = runtime.NewSchemeBuilder(addKnownTypes)
	LegacySchemeBuilder = runtime.NewSchemeBuilder(addLegacyKnownTypes)

	AddToScheme            = SchemeBuilder.AddToScheme
	AddToSchemeInCoreGroup = LegacySchemeBuilder.AddToScheme
)

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&DockerImage{},
	)
	return nil
}

func addLegacyKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(LegacySchemeGroupVersion,
		&DockerImage{},
	)
	return nil
}
