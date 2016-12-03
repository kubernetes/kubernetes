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

package dynamic

import (
	"fmt"

	"k8s.io/client-go/pkg/api/meta"
	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/runtime/schema"
)

// VersionInterfaces provides an object converter and metadata
// accessor appropriate for use with unstructured objects.
func VersionInterfaces(schema.GroupVersion) (*meta.VersionInterfaces, error) {
	return &meta.VersionInterfaces{
		ObjectConvertor:  &runtime.UnstructuredObjectConverter{},
		MetadataAccessor: meta.NewAccessor(),
	}, nil
}

// NewDiscoveryRESTMapper returns a RESTMapper based on discovery information.
func NewDiscoveryRESTMapper(resources []*metav1.APIResourceList, versionFunc meta.VersionInterfacesFunc) (*meta.DefaultRESTMapper, error) {
	rm := meta.NewDefaultRESTMapper(nil, versionFunc)
	for _, resourceList := range resources {
		gv, err := schema.ParseGroupVersion(resourceList.GroupVersion)
		if err != nil {
			return nil, err
		}

		for _, resource := range resourceList.APIResources {
			gvk := gv.WithKind(resource.Kind)
			scope := meta.RESTScopeRoot
			if resource.Namespaced {
				scope = meta.RESTScopeNamespace
			}
			rm.Add(gvk, scope)
		}
	}
	return rm, nil
}

// ObjectTyper provides an ObjectTyper implementation for
// runtime.Unstructured object based on discovery information.
type ObjectTyper struct {
	registered map[schema.GroupVersionKind]bool
}

// NewObjectTyper constructs an ObjectTyper from discovery information.
func NewObjectTyper(resources []*metav1.APIResourceList) (runtime.ObjectTyper, error) {
	ot := &ObjectTyper{registered: make(map[schema.GroupVersionKind]bool)}
	for _, resourceList := range resources {
		gv, err := schema.ParseGroupVersion(resourceList.GroupVersion)
		if err != nil {
			return nil, err
		}

		for _, resource := range resourceList.APIResources {
			ot.registered[gv.WithKind(resource.Kind)] = true
		}
	}
	return ot, nil
}

// ObjectKinds returns a slice of one element with the
// group,version,kind of the provided object, or an error if the
// object is not *runtime.Unstructured or has no group,version,kind
// information.
func (ot *ObjectTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	if _, ok := obj.(*runtime.Unstructured); !ok {
		return nil, false, fmt.Errorf("type %T is invalid for dynamic object typer", obj)
	}
	return []schema.GroupVersionKind{obj.GetObjectKind().GroupVersionKind()}, false, nil
}

// Recognizes returns true if the provided group,version,kind was in
// the discovery information.
func (ot *ObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return ot.registered[gvk]
}
