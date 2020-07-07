/*
Copyright 2020 The Kubernetes Authors.

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

package kubernetes

import (
	"bytes"
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
)

// NewUnstructuredList decodes YAML or JSON data into a flattened list of Kubernetes resources,
// it will attempt parsing all manner of manifests, such as YAML multi docs and any list types
func NewUnstructuredList(data []byte) (*unstructured.UnstructuredList, error) {
	list := new(unstructured.UnstructuredList)
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewBuffer(data), 4096)

	for {
		obj := new(unstructured.Unstructured)
		if err := decoder.Decode(obj); err != nil {
			if err == io.EOF {
				return list, nil
			}
			return nil, err
		}
		list.Items = append(list.Items, *obj)
	}
}

// AppendFlattened will append newItem to list; making sure that raw newItem is decoded
// and flattened with another list; it drops empty lists
func AppendFlattened(list *unstructured.UnstructuredList, newItem *unstructured.Unstructured) error {
	if newItem.Object == nil {
		return nil
	}
	gvk := newItem.GetObjectKind().GroupVersionKind()
	// IsList() checks if object has 'items', but it will ignore
	// something like `{"apiVersion": "v1", "kind": "List"}`, which
	// is still a list and should be treated as such
	if newItem.IsList() || strings.HasSuffix(gvk.Kind, "List") {
		innerList, err := newItem.ToList()
		if err != nil {
			return err
		}
		for _, item := range innerList.Items {
			if err := AppendFlattened(list, &item); err != nil {
				return err
			}
		}
		return nil
	}
	list.Items = append(list.Items, *newItem)
	return nil
}

// Flatten will ensure given list is fully flattened
func Flatten(list *unstructured.UnstructuredList) (*unstructured.UnstructuredList, error) {
	newList := new(unstructured.UnstructuredList)
	for _, item := range list.Items {
		if err := AppendFlattened(newList, &item); err != nil {
			return nil, err
		}
	}
	return newList, nil
}
