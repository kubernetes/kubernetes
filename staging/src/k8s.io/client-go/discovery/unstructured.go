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

package discovery

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// UnstructuredObjectTyper provides a runtime.ObjectTyper implmentation for
// runtime.Unstructured object based on discovery information.
type UnstructuredObjectTyper struct {
	registered map[schema.GroupVersionKind]bool
}

// NewUnstructuredObjectTyper returns a runtime.ObjectTyper for
// unstructred objects based on discovery information.
func NewUnstructuredObjectTyper(groupResources []*APIGroupResources) *UnstructuredObjectTyper {
	dot := &UnstructuredObjectTyper{registered: make(map[schema.GroupVersionKind]bool)}
	for _, group := range groupResources {
		for _, discoveryVersion := range group.Group.Versions {
			resources, ok := group.VersionedResources[discoveryVersion.Version]
			if !ok {
				continue
			}

			gv := schema.GroupVersion{Group: group.Group.Name, Version: discoveryVersion.Version}
			for _, resource := range resources {
				dot.registered[gv.WithKind(resource.Kind)] = true
			}
		}
	}
	return dot
}

// ObjectKinds returns a slice of one element with the group,version,kind of the
// provided object, or an error if the object is not runtime.Unstructured or
// has no group,version,kind information. unversionedType will always be false
// because runtime.Unstructured object should always have group,version,kind
// information set.
func (d *UnstructuredObjectTyper) ObjectKinds(obj runtime.Object) (gvks []schema.GroupVersionKind, unversionedType bool, err error) {
	if _, ok := obj.(runtime.Unstructured); !ok {
		return nil, false, fmt.Errorf("type %T is invalid for dynamic object typer", obj)
	}
	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, false, runtime.NewMissingKindErr("unstructured object has no kind")
	}
	if len(gvk.Version) == 0 {
		return nil, false, runtime.NewMissingVersionErr("unstructured object has no version")
	}
	return []schema.GroupVersionKind{gvk}, false, nil
}

// Recognizes returns true if the provided group,version,kind was in the
// discovery information.
func (d *UnstructuredObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return d.registered[gvk]
}

var _ runtime.ObjectTyper = &UnstructuredObjectTyper{}
