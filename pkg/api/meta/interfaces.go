/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package meta

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

// VersionInterfaces contains the interfaces one should use for dealing with types of a particular version.
type VersionInterfaces struct {
	runtime.ObjectConvertor
	MetadataAccessor
}

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
	GetCreationTimestamp() unversioned.Time
	SetCreationTimestamp(timestamp unversioned.Time)
	GetDeletionTimestamp() *unversioned.Time
	SetDeletionTimestamp(timestamp *unversioned.Time)
	GetLabels() map[string]string
	SetLabels(labels map[string]string)
	GetAnnotations() map[string]string
	SetAnnotations(annotations map[string]string)
}

// List lets you work with list metadata from any of the versioned or
// internal API objects. Attempting to set or retrieve a field on an object that does
// not support that field will be a no-op and return a default value.
type List interface {
	GetResourceVersion() string
	SetResourceVersion(version string)
	GetSelfLink() string
	SetSelfLink(selfLink string)
}

// Type exposes the type and APIVersion of versioned or internal API objects.
type Type interface {
	GetAPIVersion() string
	SetAPIVersion(version string)
	GetKind() string
	SetKind(kind string)
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

	runtime.ResourceVersioner
}

type RESTScopeName string

const (
	RESTScopeNameNamespace RESTScopeName = "namespace"
	RESTScopeNameRoot      RESTScopeName = "root"
)

// RESTScope contains the information needed to deal with REST resources that are in a resource hierarchy
type RESTScope interface {
	// Name of the scope
	Name() RESTScopeName
	// ParamName is the optional name of the parameter that should be inserted in the resource url
	// If empty, no param will be inserted
	ParamName() string
	// ArgumentName is the optional name that should be used for the variable holding the value.
	ArgumentName() string
	// ParamDescription is the optional description to use to document the parameter in api documentation
	ParamDescription() string
}

// RESTMapping contains the information needed to deal with objects of a specific
// resource and kind in a RESTful manner.
type RESTMapping struct {
	// Resource is a string representing the name of this resource as a REST client would see it
	Resource string

	GroupVersionKind unversioned.GroupVersionKind

	// Scope contains the information needed to deal with REST Resources that are in a resource hierarchy
	Scope RESTScope

	runtime.ObjectConvertor
	MetadataAccessor
}

// RESTMapper allows clients to map resources to kind, and map kind and version
// to interfaces for manipulating those objects. It is primarily intended for
// consumers of Kubernetes compatible REST APIs as defined in docs/devel/api-conventions.md.
//
// The Kubernetes API provides versioned resources and object kinds which are scoped
// to API groups. In other words, kinds and resources should not be assumed to be
// unique across groups.
//
// TODO(caesarxuchao): Add proper multi-group support so that kinds & resources are
// scoped to groups. See http://issues.k8s.io/12413 and http://issues.k8s.io/10009.
type RESTMapper interface {
	// KindFor takes a partial resource and returns back the single match.  Returns an error if there are multiple matches
	KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error)

	// KindsFor takes a partial resource and returns back the list of potential kinds in priority order
	KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error)

	// ResourceFor takes a partial resource and returns back the single match.  Returns an error if there are multiple matches
	ResourceFor(input unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error)

	// ResourcesFor takes a partial resource and returns back the list of potential resource in priority order
	ResourcesFor(input unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error)

	RESTMapping(gk unversioned.GroupKind, versions ...string) (*RESTMapping, error)

	AliasesForResource(resource string) ([]string, bool)
	ResourceSingularizer(resource string) (singular string, err error)
}

var _ Object = &runtime.Unstructured{}
