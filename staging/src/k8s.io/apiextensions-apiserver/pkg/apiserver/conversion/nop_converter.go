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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// nopConverter is a converter that only sets the apiVersion fields, but does not real conversion. It supports fields selectors.
type nopConverter struct {
}

var _ crdConverterInterface = &nopConverter{}

func (c *nopConverter) ConvertCustomResource(obj runtime.Unstructured, target runtime.GroupVersioner) (bool, error) {
	kind := obj.GetObjectKind().GroupVersionKind()
	gvk, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{kind})
	if !ok {
		// TODO: should this be a typed error?
		return false, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", kind, target)
	}
	obj.GetObjectKind().SetGroupVersionKind(gvk)
	return false, nil
}
