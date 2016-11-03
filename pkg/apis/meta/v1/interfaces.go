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
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

// ListMetaAccessor retrieves the list interface from an object
type ListMetaAccessor interface {
	GetListMeta() ListAccessor
}

// List lets you work with list metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field will be a no-op and return a default value.
type ListAccessor interface {
	GetResourceVersion() string
	SetResourceVersion(version string)
	GetSelfLink() string
	SetSelfLink(selfLink string)
}

// Type exposes the type and APIVersion of versioned or internal API objects.
type TypeAccessor interface {
	GetAPIVersion() string
	SetAPIVersion(version string)
	GetKind() string
	SetKind(kind string)
}

// ResourceVersioner provides methods for setting and retrieving
// the resource version from an API object.
type ResourceVersioner interface {
	SetResourceVersion(obj runtime.Object, version string) error
	ResourceVersion(obj runtime.Object) (string, error)
}

// SelfLinker provides methods for setting and retrieving the SelfLink field of an API object.
type SelfLinker interface {
	SetSelfLink(obj runtime.Object, selfLink string) error
	SelfLink(obj runtime.Object) (string, error)

	// Knowing Name is sometimes necessary to use a SelfLinker.
	Name(obj runtime.Object) (string, error)
	// Knowing Namespace is sometimes necessary to use a SelfLinker
	Namespace(obj runtime.Object) (string, error)
}

// ObjectMetaAccessor is implemented by "objects".
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
	GetSelfLink() string
	SetSelfLink(selfLink string)
	GetCreationTimestamp() Time
	SetCreationTimestamp(timestamp Time)
	GetDeletionTimestamp() *Time
	SetDeletionTimestamp(timestamp *Time)
	GetLabels() map[string]string
	SetLabels(labels map[string]string)
	GetAnnotations() map[string]string
	SetAnnotations(annotations map[string]string)
	GetFinalizers() []string
	SetFinalizers(finalizers []string)
	GetOwnerReferences() []OwnerReference
	SetOwnerReferences([]OwnerReference)
	GetClusterName() string
	SetClusterName(clusterName string)
}

// MetadataAccessor lets you work with object and list metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field (Name, UID, Namespace on lists) will be a no-op and return
// a default value.
//
// MetadataAccessor exposes Interface in a way that can be used with multiple objects.
type MetadataAccessor interface {
	APIVersion(obj runtime.Object) (string, error)
	SetAPIVersion(obj runtime.Object, version string) error

	Kind(obj runtime.Object) (string, error)
	SetKind(obj runtime.Object, kind string) error

	Namespace(obj runtime.Object) (string, error)
	SetNamespace(obj runtime.Object, namespace string) error

	Name(obj runtime.Object) (string, error)
	SetName(obj runtime.Object, name string) error

	GenerateName(obj runtime.Object) (string, error)
	SetGenerateName(obj runtime.Object, name string) error

	UID(obj runtime.Object) (types.UID, error)
	SetUID(obj runtime.Object, uid types.UID) error

	SelfLink(obj runtime.Object) (string, error)
	SetSelfLink(obj runtime.Object, selfLink string) error

	Labels(obj runtime.Object) (map[string]string, error)
	SetLabels(obj runtime.Object, labels map[string]string) error

	Annotations(obj runtime.Object) (map[string]string, error)
	SetAnnotations(obj runtime.Object, annotations map[string]string) error

	ResourceVersioner
}
