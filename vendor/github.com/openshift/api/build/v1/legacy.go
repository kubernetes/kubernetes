package v1

import (
	corev1 "k8s.io/api/core/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var (
	legacyGroupVersion            = schema.GroupVersion{Group: "", Version: "v1"}
	legacySchemeBuilder           = runtime.NewSchemeBuilder(addLegacyKnownTypes, corev1.AddToScheme)
	DeprecatedInstallWithoutGroup = legacySchemeBuilder.AddToScheme
)

func addLegacyKnownTypes(scheme *runtime.Scheme) error {
	types := []runtime.Object{
		&Build{},
		&BuildList{},
		&BuildConfig{},
		&BuildConfigList{},
		&BuildLog{},
		&BuildRequest{},
		&BuildLogOptions{},
		&BinaryBuildRequestOptions{},
	}
	scheme.AddKnownTypes(legacyGroupVersion, types...)
	return nil
}
