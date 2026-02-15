package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

var SchemeGroupVersion = metav1.GroupVersion{
	Group:   "networking.k8s.io",
	Version: "v1alpha1",
}

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&PodNetworkHealth{},
	)
	return nil
}
