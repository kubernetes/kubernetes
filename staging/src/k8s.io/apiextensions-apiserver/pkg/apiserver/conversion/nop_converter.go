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

package conversion

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// nopConverter is a converter that only sets the apiVersion fields, but does not real conversion.
type nopConverter struct {
}

var _ crConverterInterface = &nopConverter{}

// ConvertToVersion converts in object to the given gv in place and returns the same `in` object.
func (c *nopConverter) Convert(in runtime.Object, targetGV schema.GroupVersion) (runtime.Object, error) {
	// Run the converter on the list items instead of list itself
	if list, ok := in.(*unstructured.UnstructuredList); ok {
		for i := range list.Items {
			list.Items[i].SetGroupVersionKind(targetGV.WithKind(list.Items[i].GroupVersionKind().Kind))
		}
	}
	in.GetObjectKind().SetGroupVersionKind(targetGV.WithKind(in.GetObjectKind().GroupVersionKind().Kind))
	return in, nil
}
