package v1

import (
	"fmt"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	v1 "k8s.io/client-go/applyconfigurations/meta/v1"
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

func (p PodApplyConfigurationNew) GetNamespace() string {
	if p.ObjectMetaApplyConfiguration == nil {
		return ""
	}
	return ptr.Deref(p.Namespace, "")
}

func (p PodApplyConfigurationNew) SetNamespace(namespace string) {
	p.WithNamespace(namespace)
}

func (p PodApplyConfigurationNew) GetName() string {
	if p.ObjectMetaApplyConfiguration == nil {
		return ""
	}
	return ptr.Deref(p.Name, "")
}

func (p PodApplyConfigurationNew) SetName(name string) {
	p.WithName(name)
}

func (p PodApplyConfigurationNew) GetGenerateName() string {
	if p.ObjectMetaApplyConfiguration == nil {
		return ""
	}
	return ptr.Deref(p.GenerateName, "")
}

func (p PodApplyConfigurationNew) SetGenerateName(name string) {
	p.WithGenerateName(name)
}

func (p PodApplyConfigurationNew) GetUID() types.UID {
	if p.ObjectMetaApplyConfiguration == nil {
		return ""
	}
	return ptr.Deref(p.UID, "")
}

func (p PodApplyConfigurationNew) SetUID(uid types.UID) {
	p.WithUID(uid)
}

func (p PodApplyConfigurationNew) GetResourceVersion() string {
	if p.ObjectMetaApplyConfiguration == nil {
		return ""
	}
	return ptr.Deref(p.ResourceVersion, "")
}

func (p PodApplyConfigurationNew) SetResourceVersion(version string) {
	p.WithResourceVersion(version)
}

func (p PodApplyConfigurationNew) GetGeneration() int64 {
	if p.ObjectMetaApplyConfiguration == nil {
		return 0
	}
	return ptr.Deref(p.Generation, 0)
}

func (p PodApplyConfigurationNew) SetGeneration(generation int64) {
	p.WithGeneration(generation)
}

func (p PodApplyConfigurationNew) GetSelfLink() string {
	panic("not implemented")
}

func (p PodApplyConfigurationNew) SetSelfLink(selfLink string) {
	panic("not implemented")
}

func (p PodApplyConfigurationNew) GetCreationTimestamp() metav1.Time {
	if p.ObjectMetaApplyConfiguration == nil {
		return metav1.Time{}
	}
	return ptr.Deref(p.CreationTimestamp, metav1.Time{})
}

func (p PodApplyConfigurationNew) SetCreationTimestamp(timestamp metav1.Time) {
	p.WithCreationTimestamp(timestamp)
}

func (p PodApplyConfigurationNew) GetDeletionTimestamp() *metav1.Time {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	return p.DeletionTimestamp
}

func (p PodApplyConfigurationNew) SetDeletionTimestamp(timestamp *metav1.Time) {
	if timestamp != nil {
		p.WithDeletionTimestamp(*timestamp)
	}
}

func (p PodApplyConfigurationNew) GetDeletionGracePeriodSeconds() *int64 {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	return p.DeletionGracePeriodSeconds
}

func (p PodApplyConfigurationNew) SetDeletionGracePeriodSeconds(i *int64) {
	if i != nil {
		p.WithDeletionGracePeriodSeconds(*i)
	}
}

func (p PodApplyConfigurationNew) GetLabels() map[string]string {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	return p.Labels
}

func (p PodApplyConfigurationNew) SetLabels(labels map[string]string) {
	p.WithLabels(labels)
}

func (p PodApplyConfigurationNew) GetAnnotations() map[string]string {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	return p.Annotations
}

func (p PodApplyConfigurationNew) SetAnnotations(annotations map[string]string) {
	p.WithAnnotations(annotations)
}

func (p PodApplyConfigurationNew) GetFinalizers() []string {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	return p.Finalizers
}

func (p PodApplyConfigurationNew) SetFinalizers(finalizers []string) {
	p.WithFinalizers(finalizers...)
}

func (p PodApplyConfigurationNew) GetOwnerReferences() []metav1.OwnerReference {
	if p.ObjectMetaApplyConfiguration == nil {
		return nil
	}
	ownerReferences := make([]metav1.OwnerReference, len(p.OwnerReferences))
	for i, or := range p.OwnerReferences {
		// The following code should be replaceable by or.Object()
		ownerReferences[i] = metav1.OwnerReference{
			APIVersion:         ptr.Deref(or.APIVersion, ""),
			Kind:               ptr.Deref(or.Kind, ""),
			Name:               ptr.Deref(or.Name, ""),
			UID:                ptr.Deref(or.UID, ""),
			Controller:         or.Controller,
			BlockOwnerDeletion: or.BlockOwnerDeletion,
		}
	}
	return ownerReferences
}

func (p PodApplyConfigurationNew) SetOwnerReferences(references []metav1.OwnerReference) {
	ownerReferences := make([]*v1.OwnerReferenceApplyConfiguration, len(references))
	for i, or := range references {
		// The following code should be replaceable by or.Object()
		ownerReferences[i] = v1.OwnerReference().
			WithAPIVersion(or.APIVersion).
			WithKind(or.Kind).
			WithName(or.Name).
			WithUID(or.UID)
		ownerReferences[i].Controller = or.Controller
		ownerReferences[i].BlockOwnerDeletion = or.BlockOwnerDeletion
	}
	p.WithOwnerReferences(ownerReferences...)
}

func (p PodApplyConfigurationNew) GetManagedFields() []metav1.ManagedFieldsEntry {
	//TODO implement me
	panic("implement me")
}

func (p PodApplyConfigurationNew) SetManagedFields(managedFields []metav1.ManagedFieldsEntry) {
	//TODO implement me
	panic("implement me")
}

var _ metav1.Object = &PodApplyConfigurationNew{}
var _ runtime.ObjectApplyConfiguration[*corev1.Pod] = &PodApplyConfigurationNew{}
