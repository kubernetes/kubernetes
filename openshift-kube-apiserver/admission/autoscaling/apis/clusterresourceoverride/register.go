package clusterresourceoverride

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var GroupVersion = schema.GroupVersion{Group: "autoscaling.openshift.io", Version: runtime.APIVersionInternal}

var (
	schemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	Install       = schemeBuilder.AddToScheme
)

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(GroupVersion,
		&ClusterResourceOverrideConfig{},
	)
	return nil
}

func (obj *ClusterResourceOverrideConfig) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
