/*
Copyright 2017 The Kubernetes Authors.

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

package parse

import "k8s.io/kubernetes/pkg/kubectl/apply"

// primitiveElement builds a new primitiveElement from a PrimitiveItem
func (v ElementBuildingVisitor) primitiveElement(item *primitiveItem) (*apply.PrimitiveElement, error) {
	meta := apply.FieldMetaImpl{Name: item.Name}
	return &apply.PrimitiveElement{
		FieldMetaImpl:  meta,
		RawElementData: item.RawElementData,
	}, nil
}
