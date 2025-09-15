/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

// TODO: move this, Object, List, and Type to a different package
type ObjectMetaAccessor interface {
	GetObjectMeta() Object
}

// Object lets you work with object metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field (Name, UID, Namespace on lists) will be a no-op and return
// a default value.
type Object interface {
	GetNamespace() string
	SetNamespace(namespace string)
	GetName() string
	SetName(name string)
	GetGenerateName() string
	SetGenerateName(name string)
	GetUID() types.UID
	SetUID(uid types.UID)
	GetResourceVersion() string
	SetResourceVersion(version string)
	GetGeneration() int64
	SetGeneration(generation int64)
	GetSelfLink() string
	SetSelfLink(selfLink string)
	GetCreationTimestamp() Time
	SetCreationTimestamp(timestamp Time)
	GetDeletionTimestamp() *Time
	SetDeletionTimestamp(timestamp *Time)
	GetDeletionGracePeriodSeconds() *int64
	SetDeletionGracePeriodSeconds(*int64)
	GetLabels() map[string]string
	SetLabels(labels map[string]string)
	GetAnnotations() map[string]string
	SetAnnotations(annotations map[string]string)
	GetFinalizers() []string
	SetFinalizers(finalizers []string)
	GetOwnerReferences() []OwnerReference
	SetOwnerReferences([]OwnerReference)
	GetManagedFields() []ManagedFieldsEntry
	SetManagedFields(managedFields []ManagedFieldsEntry)
}

// ListMetaAccessor retrieves the list interface from an object
type ListMetaAccessor interface {
	GetListMeta() ListInterface
}

// Common lets you work with core metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field will be a no-op and return a default value.
// TODO: move this, and TypeMeta and ListMeta, to a different package
type Common interface {
	GetResourceVersion() string
	SetResourceVersion(version string)
	GetSelfLink() string
	SetSelfLink(selfLink string)
}

// ListInterface lets you work with list metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field will be a no-op and return a default value.
// TODO: move this, and TypeMeta and ListMeta, to a different package
type ListInterface interface {
	GetResourceVersion() string
	SetResourceVersion(version string)
	GetSelfLink() string
	SetSelfLink(selfLink string)
	GetContinue() string
	SetContinue(c string)
	GetRemainingItemCount() *int64
	SetRemainingItemCount(c *int64)
}

// Type exposes the type and APIVersion of versioned or internal API objects.
// TODO: move this, and TypeMeta and ListMeta, to a different package
type Type interface {
	GetAPIVersion() string
	SetAPIVersion(version string)
	GetKind() string
	SetKind(kind string)
}

var _ ListInterface = &ListMeta{}

func (meta *ListMeta) GetResourceVersion() string        { return meta.ResourceVersion }
func (meta *ListMeta) SetResourceVersion(version string) { meta.ResourceVersion = version }
func (meta *ListMeta) GetSelfLink() string               { return meta.SelfLink }
func (meta *ListMeta) SetSelfLink(selfLink string)       { meta.SelfLink = selfLink }
func (meta *ListMeta) GetContinue() string               { return meta.Continue }
func (meta *ListMeta) SetContinue(c string)              { meta.Continue = c }
func (meta *ListMeta) GetRemainingItemCount() *int64     { return meta.RemainingItemCount }
func (meta *ListMeta) SetRemainingItemCount(c *int64)    { meta.RemainingItemCount = c }

func (obj *TypeMeta) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func (obj *ListMeta) GetListMeta() ListInterface { return obj }

func (obj *ObjectMeta) GetObjectMeta() Object { return obj }

// Namespace implements metav1.Object for any object with an ObjectMeta typed field. Allows
// fast, direct access to metadata fields for API objects.
func (meta *ObjectMeta) GetNamespace() string                { return meta.Namespace }
func (meta *ObjectMeta) SetNamespace(namespace string)       { meta.Namespace = namespace }
func (meta *ObjectMeta) GetName() string                     { return meta.Name }
func (meta *ObjectMeta) SetName(name string)                 { meta.Name = name }
func (meta *ObjectMeta) GetGenerateName() string             { return meta.GenerateName }
func (meta *ObjectMeta) SetGenerateName(generateName string) { meta.GenerateName = generateName }
func (meta *ObjectMeta) GetUID() types.UID                   { return meta.UID }
func (meta *ObjectMeta) SetUID(uid types.UID)                { meta.UID = uid }
func (meta *ObjectMeta) GetResourceVersion() string          { return meta.ResourceVersion }
func (meta *ObjectMeta) SetResourceVersion(version string)   { meta.ResourceVersion = version }
func (meta *ObjectMeta) GetGeneration() int64                { return meta.Generation }
func (meta *ObjectMeta) SetGeneration(generation int64)      { meta.Generation = generation }
func (meta *ObjectMeta) GetSelfLink() string                 { return meta.SelfLink }
func (meta *ObjectMeta) SetSelfLink(selfLink string)         { meta.SelfLink = selfLink }
func (meta *ObjectMeta) GetCreationTimestamp() Time          { return meta.CreationTimestamp }
func (meta *ObjectMeta) SetCreationTimestamp(creationTimestamp Time) {
	meta.CreationTimestamp = creationTimestamp
}
func (meta *ObjectMeta) GetDeletionTimestamp() *Time { return meta.DeletionTimestamp }
func (meta *ObjectMeta) SetDeletionTimestamp(deletionTimestamp *Time) {
	meta.DeletionTimestamp = deletionTimestamp
}
func (meta *ObjectMeta) GetDeletionGracePeriodSeconds() *int64 {
	return meta.DeletionGracePeriodSeconds
}
func (meta *ObjectMeta) SetDeletionGracePeriodSeconds(deletionGracePeriodSeconds *int64) {
	meta.DeletionGracePeriodSeconds = deletionGracePeriodSeconds
}
func (meta *ObjectMeta) GetLabels() map[string]string                 { return meta.Labels }
func (meta *ObjectMeta) SetLabels(labels map[string]string)           { meta.Labels = labels }
func (meta *ObjectMeta) GetAnnotations() map[string]string            { return meta.Annotations }
func (meta *ObjectMeta) SetAnnotations(annotations map[string]string) { meta.Annotations = annotations }
func (meta *ObjectMeta) GetFinalizers() []string                      { return meta.Finalizers }
func (meta *ObjectMeta) SetFinalizers(finalizers []string)            { meta.Finalizers = finalizers }
func (meta *ObjectMeta) GetOwnerReferences() []OwnerReference         { return meta.OwnerReferences }
func (meta *ObjectMeta) SetOwnerReferences(references []OwnerReference) {
	meta.OwnerReferences = references
}
func (meta *ObjectMeta) GetManagedFields() []ManagedFieldsEntry { return meta.ManagedFields }
func (meta *ObjectMeta) SetManagedFields(managedFields []ManagedFieldsEntry) {
	meta.ManagedFields = managedFields
}
