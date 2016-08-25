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

package announced

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// GroupMetaFactoryArgs contains the group-level args of a GroupMetaFactory.
type GroupMetaFactoryArgs struct {
	GroupName              string
	VersionPreferenceOrder []string
	ImportPrefix           string

	RootScopedKinds sets.String // nil is allowed
	IgnoredKinds    sets.String // nil is allowed

	// These will be added to the scheme with the internal version of the
	// group.
	InternalObjects []runtime.Object

	// This for any fixup, optimizations, etc that needs to be done to the
	// scheme on account of the internal objects, so for example maybe you
	// want to register deep copy functions etc. May be nil if nothing
	// needs to be done.
	InternalSchemeAdjustment *runtime.SchemeBuilder
}

// AddInternalObjectsToScheme adds all the internal objects to the scheme,
// including any adjustments (deep copy, etc).
func (g *GroupMetaFactoryArgs) AddInternalObjectsToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(g.SchemeGroupVersion(), g.InternalObjects...)
	if g.InternalSchemeAdjustment == nil {
		return nil
	}
	return g.InternalSchemeAdjustment.VersionedAddToScheme(scheme, g.SchemeGroupVersion())
}

// SchemeGroupVersion returns the version that will be used to register
// objects, e.g. batch/__internal.
func (g *GroupMetaFactoryArgs) SchemeGroupVersion() unversioned.GroupVersion {
	return unversioned.GroupVersion{
		Group:   g.GroupName,
		Version: runtime.APIVersionInternal,
	}
}

// Kind constructs a GroupKind for objects from this group.
func (g *GroupMetaFactoryArgs) Kind(kind string) unversioned.GroupKind {
	return g.SchemeGroupVersion().WithKind(kind).GroupKind()
}

// Resource constructs a GroupResource for objects from this group.
func (g *GroupMetaFactoryArgs) Resource(resource string) unversioned.GroupResource {
	return g.SchemeGroupVersion().WithResource(resource).GroupResource()
}

var (
	_ = KindResourceConstructor(&GroupMetaFactoryArgs{})
	_ = KindResourceConstructor(&GroupVersionFactoryArgs{})
)
