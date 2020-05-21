package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride"
)

func (obj *ClusterResourceOverrideConfig) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }

var GroupVersion = schema.GroupVersion{Group: "autoscaling.openshift.io", Version: "v1"}

var (
	schemeBuilder = runtime.NewSchemeBuilder(
		addKnownTypes,
		clusterresourceoverride.Install,
	)
	Install = schemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(GroupVersion,
		&ClusterResourceOverrideConfig{},
	)
	return nil
}
