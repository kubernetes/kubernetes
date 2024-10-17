package v1

import (
	"fmt"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/utils/ptr"
)

// PodApplyConfigurationNew is a prototype/example for the future PodApplyConfiguration after
// applyconfiguration-gen is updated to make apply configurations generated for runtime.Object
// implement meta.Object and the new (proposed ) runtime.ObjectApplyConfiguration interface.
//
// NOTE: The current suggested implementations are just minimal viable products, and should be
// reviewed and optimized further when the generator is updated.
type PodApplyConfigurationNew struct {
	PodApplyConfiguration
}

func (p PodApplyConfigurationNew) SetGroupVersionKind(kind schema.GroupVersionKind) {
	p.WithAPIVersion(kind.GroupVersion().String())
}

func (p PodApplyConfigurationNew) GroupVersionKind() schema.GroupVersionKind {
	groupVersion, _ := schema.ParseGroupVersion(ptr.Deref(p.APIVersion, ""))
	return groupVersion.WithKind(ptr.Deref(p.Kind, ""))
}

func (p PodApplyConfigurationNew) GetObjectKind() schema.ObjectKind {
	return p
}

func (p PodApplyConfigurationNew) Object() *corev1.Pod {
	buffer, err := json.Marshal(p)
	if err != nil {
		panic(fmt.Errorf("encoding %T as JSON: %v", p, err))
	}
	obj := &corev1.Pod{}
	if err := json.Unmarshal(buffer, obj); err != nil {
		panic(fmt.Errorf("decoding %T from JSON: %v", obj, err))
	}
	return obj
}

func (p PodApplyConfigurationNew) GetNamespace() *string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.Namespace
}

func (p PodApplyConfigurationNew) GetGenerateName() *string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.GenerateName
}

func (p PodApplyConfigurationNew) GetUID() *types.UID {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.UID
}

func (p PodApplyConfigurationNew) GetResourceVersion() *string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.ResourceVersion
}

func (p PodApplyConfigurationNew) GetGeneration() *int64 {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.Generation
}

func (p PodApplyConfigurationNew) GetCreationTimestamp() *metav1.Time {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.CreationTimestamp
}

func (p PodApplyConfigurationNew) GetDeletionTimestamp() *metav1.Time {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.DeletionTimestamp
}

func (p PodApplyConfigurationNew) GetDeletionGracePeriodSeconds() *int64 {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.DeletionGracePeriodSeconds
}

func (p PodApplyConfigurationNew) GetLabels() map[string]string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.Labels
}

func (p PodApplyConfigurationNew) GetAnnotations() map[string]string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.Annotations
}

func (p PodApplyConfigurationNew) GetFinalizers() []string {
	p.ensureObjectMetaApplyConfigurationExists()
	return p.ObjectMetaApplyConfiguration.Finalizers
}

// The following vars are just a temporary check that types satisfy the new interfaces correctly
var _ runtime.ObjectApplyConfiguration[*PodApplyConfiguration, *corev1.Pod] = &PodApplyConfigurationNew{}
var _ metav1.ObjectApplyConfiguration[*PodApplyConfiguration] = &PodApplyConfigurationNew{}
