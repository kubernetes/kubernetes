package v1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	legacyGroupVersion            = schema.GroupVersion{Group: "", Version: "v1"}
	legacySchemeBuilder           = runtime.NewSchemeBuilder(addLegacyKnownTypes, corev1.AddToScheme)
	DeprecatedInstallWithoutGroup = legacySchemeBuilder.AddToScheme
)

func addLegacyKnownTypes(scheme *runtime.Scheme) error {
	types := []runtime.Object{
		&User{},
		&UserList{},
		&Identity{},
		&IdentityList{},
		&UserIdentityMapping{},
		&Group{},
		&GroupList{},
	}
	scheme.AddKnownTypes(legacyGroupVersion, types...)
	return nil
}
