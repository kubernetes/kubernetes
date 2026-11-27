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
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
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

// APIResult is an interface that represents the result of an API operation.
type APIResult interface {
	// Get returns the result as a runtime.Object, or an error if the operation failed.
	// If Error() returns a non-nil error, Get will return nil, err.
	Get() (runtime.Object, error)

	// StatusCode returns the HTTP status code of the response and any error encountered.
	// If Error() returns a non-nil error that prevented a response from being read (such as a network failure),
	// StatusCode returns 0, err.
	// If the error was returned from the server with an HTTP status code, StatusCode returns status, err.
	// If the operation succeeded, StatusCode returns status, nil.
	StatusCode() (int, error)

	// Error returns the error executing the request, or nil if no error occurred.
	Error() error
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

// ShardedListInterface can be implemented by list types to indicate that they
// represent a sharded subset of the full collection rather than the complete list.
type ShardedListInterface interface {
	GetShardInfo() *ShardInfo
	SetShardInfo(*ShardInfo)
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
func (meta *ListMeta) GetShardInfo() *ShardInfo          { return meta.ShardInfo }
func (meta *ListMeta) SetShardInfo(s *ShardInfo)         { meta.ShardInfo = s }

func (obj *TypeMeta) GetObjectKind() schema.ObjectKind { return obj }

// SetGroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) GroupVersionKind() schema.GroupVersionKind {
	if o, ok := typeMetaOverrides.Load(obj); ok {
		typeMeta := o.(typeMetaOverride).typeMeta
		return typeMeta.GroupVersionKind()
	}
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

var (
	// typeMetaOverrides maps TypeMeta pointers to typeOverride structs values
	// which contain the actual TypeMeta while an object embedding the
	// original TypeMeta instance is being encoded.
	//
	// A sync.Map is here because in practice there is no contention on they keys
	// ("The Map type is optimized for ... (2) when multiple goroutines read, write, and overwrite
	// entries for disjoint sets of keys").
	typeMetaOverrides sync.Map
)

// typeMetaOverride contains the refcount (= number of active encoding operations)
// and actual TypeMeta for those operations. Doesn't use pointers because the
// entire struct must be comparable.
type typeMetaOverride struct {
	refcount int
	typeMeta TypeMeta
}

// OverrideVersionForProtobufEncoding satisfies the ProtobufVersionOverrider interface for all objects that embed TypeMeta
func (obj *TypeMeta) OverrideVersionForProtobufEncoding(gvk schema.GroupVersionKind) func() {
	// The most common case is that an object is being encoded once,
	// so typically we do one LoadOrStore and one CompareAndDelete.
	//
	// But we have to support encoding it in parallel, too, so
	// each override must bump the refcount and only the last one may
	// remove the override entry. This is achieved by comparing
	// and retrying in case of a conflict.
	//
	// Theoretically it could happen that two concurrent encoding operations
	// want to use different GVK for the same object. This is handled
	// by comparing the desired TypeMeta as part of the typeMetaOverride,
	// which forces the loosing operation to busy loop until it can store
	// its override - not ideal, but seems so unlikely that it doesn't
	// warant adding additional code (like exponential backoff).
	want := typeMetaOverride{refcount: 1}
	want.typeMeta.APIVersion, want.typeMeta.Kind = gvk.ToAPIVersionAndKind()
	for {
		if want.refcount == 1 {
			// Try adding a new entry. Fails if added by someone else already.
			if actual, loaded := typeMetaOverrides.LoadOrStore(obj, want); loaded {
				// Try again with new refcount.
				want.refcount = actual.(typeMetaOverride).refcount + 1
				continue
			}
			// Done with insertion.
			break
		} else {
			// Increase refcount, but only if we bump it exactly by one.
			if swapped := typeMetaOverrides.CompareAndSwap(obj, typeMetaOverride{refcount: want.refcount - 1, typeMeta: want.typeMeta}, want); swapped {
				// Done with increment.
				break
			}
			if old, ok := typeMetaOverrides.Load(obj); ok {
				// Someone else changed the refcount in the meantime, try again with new value.
				want.refcount = old.(typeMetaOverride).refcount + 1
				continue
			}
			// Gone in the meantime, insert anew.
			want.refcount = 1
		}
	}

	return func() {
		for {
			if want.refcount == 1 {
				if deleted := typeMetaOverrides.CompareAndDelete(obj, want); deleted {
					// Done.
					break
				}
			} else {
				if swapped := typeMetaOverrides.CompareAndSwap(obj, want, typeMetaOverride{refcount: want.refcount - 1, typeMeta: want.typeMeta}); swapped {
					// Done with decrement.
					break
				}
			}
			if old, ok := typeMetaOverrides.Load(obj); ok {
				// Someone else changed the refcount in the meantime, try again with new value.
				want.refcount = old.(typeMetaOverride).refcount
				continue
			}
			// It's gone?! Shouldn't happen.
			break
		}
	}
}

var _ runtime.ProtobufVersionOverrider = &TypeMeta{}

func (m *TypeMeta) Marshal() (dAtA []byte, err error) {
	if o, ok := typeMetaOverrides.Load(m); ok {
		override := o.(typeMetaOverride).typeMeta
		return override.marshal()
	}
	return m.marshal()
}

func (m *TypeMeta) MarshalTo(dAtA []byte) (int, error) {
	if o, ok := typeMetaOverrides.Load(m); ok {
		override := o.(typeMetaOverride).typeMeta
		return override.marshalTo(dAtA)
	}
	return m.marshalTo(dAtA)
}

func (m *TypeMeta) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	if o, ok := typeMetaOverrides.Load(m); ok {
		override := o.(typeMetaOverride).typeMeta
		return override.marshalToSizedBuffer(dAtA)
	}
	return m.marshalToSizedBuffer(dAtA)
}

func (m *TypeMeta) Size() (n int) {
	if o, ok := typeMetaOverrides.Load(m); ok {
		override := o.(typeMetaOverride).typeMeta
		return override.size()
	}
	return m.size()
}

// The approach above for overriding the TypeMeta in the marshal function for protobuf
// does not work for JSON: objects embedding TypeMeta inherit MarshalJSON from
// TypeMeta and (in contrast to protobuf) don't provide their own method, so
// encoding objects would only encode TypeMeta and none of the other other fields
// of the object.

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
