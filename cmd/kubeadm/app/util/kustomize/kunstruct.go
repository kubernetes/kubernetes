/*
Copyright 2019 The Kubernetes Authors.

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

package kustomize

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// UnstructuredSlice is a slice of Unstructured objects.
// Unstructured objects are used to represent both resources and patches of any group/version/kind.
type UnstructuredSlice []*unstructured.Unstructured

// NewUnstructuredSliceFromFiles returns a ResMap given a resource path slice.
// This func use a Loader to mimic the behavior of kubectl kustomize, and most specifically support for reading from
// a local git repository like git@github.com:someOrg/someRepo.git or https://github.com/someOrg/someRepo?ref=someHash
func NewUnstructuredSliceFromFiles(loader ifc.Loader, paths []string) (UnstructuredSlice, error) {
	var result UnstructuredSlice
	for _, path := range paths {
		content, err := loader.Load(path)
		if err != nil {
			return nil, errors.Wrapf(err, "load from path %q failed", path)
		}
		res, err := NewUnstructuredSliceFromBytes(content)
		if err != nil {
			return nil, errors.Wrapf(err, "convert %q to Unstructured failed", path)
		}

		result = append(result, res...)
	}
	return result, nil
}

// NewUnstructuredSliceFromBytes returns a slice of Unstructured.
// This functions handles all the nuances of Kubernetes yaml (e.g. many yaml
// documents in one file, List of objects)
func NewUnstructuredSliceFromBytes(in []byte) (UnstructuredSlice, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(in), 1024)
	var result UnstructuredSlice
	var err error
	// Parse all the yaml documents in the file
	for err == nil || isEmptyYamlError(err) {
		var u unstructured.Unstructured
		err = decoder.Decode(&u)
		// if the yaml document is a valid unstructured object
		if err == nil {
			// it the unstructured object is empty, move to the next
			if len(u.Object) == 0 {
				continue
			}

			// validate the object has kind, metadata.name as required by Kustomize
			if err := validate(u); err != nil {
				return nil, err
			}

			// if the document is a list of objects
			if strings.HasSuffix(u.GetKind(), "List") {
				// for each item in the list of objects
				if err := u.EachListItem(func(item runtime.Object) error {
					// Marshal the object
					itemJSON, err := json.Marshal(item)
					if err != nil {
						return err
					}

					// Get the UnstructuredSlice for the item
					itemU, err := NewUnstructuredSliceFromBytes(itemJSON)
					if err != nil {
						return err
					}

					// append the UnstructuredSlice for the item to the UnstructuredSlice
					result = append(result, itemU...)

					return nil
				}); err != nil {
					return nil, err
				}

				continue
			}

			// append the object to the UnstructuredSlice
			result = append(result, &u)
		}
	}
	if err != io.EOF {
		return nil, err
	}
	return result, nil
}

// FilterResource returns all the Unstructured items in the UnstructuredSlice corresponding to a given resource
func (rs *UnstructuredSlice) FilterResource(gvk schema.GroupVersionKind, namespace, name string) UnstructuredSlice {
	var result UnstructuredSlice
	for _, r := range *rs {
		if r.GroupVersionKind() == gvk &&
			r.GetNamespace() == namespace &&
			r.GetName() == name {
			result = append(result, r)
		}
	}
	return result
}

// validate validates that u has kind and name
// except for kind `List`, which doesn't require a name
func validate(u unstructured.Unstructured) error {
	kind := u.GetKind()
	if kind == "" {
		return errors.New("missing kind in object")
	} else if strings.HasSuffix(kind, "List") {
		return nil
	}
	if u.GetName() == "" {
		return errors.New("missing metadata.name in object")
	}
	return nil
}

func isEmptyYamlError(err error) bool {
	return strings.Contains(err.Error(), "is missing in 'null'")
}
