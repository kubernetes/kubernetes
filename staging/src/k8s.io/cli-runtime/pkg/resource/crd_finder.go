/*
Copyright 2018 The Kubernetes Authors.

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

package resource

import (
	"fmt"
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// CRDGetter is a function that can download the list of GVK for all
// CRDs.
type CRDGetter func() ([]schema.GroupKind, error)

func CRDFromDynamic(client dynamic.Interface) CRDGetter {
	return func() ([]schema.GroupKind, error) {
		list, err := client.Resource(schema.GroupVersionResource{
			Group:    "apiextensions.k8s.io",
			Version:  "v1beta1",
			Resource: "customresourcedefinitions",
		}).List(metav1.ListOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to list CRDs: %v", err)
		}
		if list == nil {
			return nil, nil
		}

		gks := []schema.GroupKind{}

		// We need to parse the list to get the gvk, I guess that's fine.
		for _, crd := range (*list).Items {
			// Look for group, version, and kind
			group, _, _ := unstructured.NestedString(crd.Object, "spec", "group")
			kind, _, _ := unstructured.NestedString(crd.Object, "spec", "names", "kind")

			gks = append(gks, schema.GroupKind{
				Group: group,
				Kind:  kind,
			})
		}

		return gks, nil
	}
}

// CRDFinder keeps a cache of known CRDs and finds a given GVK in the
// list.
type CRDFinder interface {
	HasCRD(gvk schema.GroupKind) (bool, error)
}

func NewCRDFinder(getter CRDGetter) CRDFinder {
	return &crdFinder{
		getter: getter,
	}
}

type crdFinder struct {
	getter CRDGetter
	cache  *[]schema.GroupKind
}

func (f *crdFinder) cacheCRDs() error {
	if f.cache != nil {
		return nil
	}

	list, err := f.getter()
	if err != nil {
		return err
	}
	f.cache = &list
	return nil
}

func (f *crdFinder) findCRD(gvk schema.GroupKind) bool {
	for _, crd := range *f.cache {
		if reflect.DeepEqual(gvk, crd) {
			return true
		}
	}
	return false
}

func (f *crdFinder) HasCRD(gvk schema.GroupKind) (bool, error) {
	if err := f.cacheCRDs(); err != nil {
		return false, err
	}
	return f.findCRD(gvk), nil
}
