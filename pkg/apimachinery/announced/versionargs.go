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
)

// GroupVersionFactoryArgs contains all the per-version parts of a GroupMetaFactory.
type GroupVersionFactoryArgs struct {
	GroupName   string
	VersionName string

	// May be nil if no adjustment is necessary.
	SchemeAdjustment *runtime.SchemeBuilder

	// All objects in this version.
	VersionedObjects []runtime.Object
}

// AddToScheme adds versioned objects to the scheme, along with any adjustment
// (conversion functions, etc).
func (g *GroupVersionFactoryArgs) AddToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(g.SchemeGroupVersion(), g.VersionedObjects...)
	if g.SchemeAdjustment == nil {
		return nil
	}
	return g.SchemeAdjustment.VersionedAddToScheme(scheme, g.SchemeGroupVersion())
}

// SchemeGroupVersion returns the version that will be used to register
// objects, e.g. batch/v1.
func (g *GroupVersionFactoryArgs) SchemeGroupVersion() unversioned.GroupVersion {
	return unversioned.GroupVersion{
		Group:   g.GroupName,
		Version: g.VersionName,
	}
}

// Kind constructs a GroupKind for objects from this group.
func (g *GroupVersionFactoryArgs) Kind(kind string) unversioned.GroupKind {
	return g.SchemeGroupVersion().WithKind(kind).GroupKind()
}

// Resource constructs a GroupResource for objects from this group.
func (g *GroupVersionFactoryArgs) Resource(resource string) unversioned.GroupResource {
	return g.SchemeGroupVersion().WithResource(resource).GroupResource()
}

var (
	_ = KindResourceConstructor(&GroupVersionFactoryArgs{})
)
