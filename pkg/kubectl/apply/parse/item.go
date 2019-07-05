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

import (
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubernetes/pkg/kubectl/apply"
)

// Item wraps values from 3 sources (recorded, local, remote).
// The values are not collated
type Item interface {
	// CreateElement merges the values in the item into a combined Element
	CreateElement(ItemVisitor) (apply.Element, error)
}

// primitiveItem contains a recorded, local, and remote value
type primitiveItem struct {
	Name      string
	Primitive *proto.Primitive

	apply.RawElementData
}

func (i *primitiveItem) CreateElement(v ItemVisitor) (apply.Element, error) {
	return v.CreatePrimitiveElement(i)
}

func (i *primitiveItem) GetMeta() proto.Schema {
	// https://golang.org/doc/faq#nil_error
	if i.Primitive != nil {
		return i.Primitive
	}
	return nil
}

// listItem contains a recorded, local, and remote list
type listItem struct {
	Name  string
	Array *proto.Array

	apply.ListElementData
}

func (i *listItem) CreateElement(v ItemVisitor) (apply.Element, error) {
	return v.CreateListElement(i)
}

func (i *listItem) GetMeta() proto.Schema {
	// https://golang.org/doc/faq#nil_error
	if i.Array != nil {
		return i.Array
	}
	return nil
}

// mapItem contains a recorded, local, and remote map
type mapItem struct {
	Name string
	Map  *proto.Map

	apply.MapElementData
}

func (i *mapItem) CreateElement(v ItemVisitor) (apply.Element, error) {
	return v.CreateMapElement(i)
}

func (i *mapItem) GetMeta() proto.Schema {
	// https://golang.org/doc/faq#nil_error
	if i.Map != nil {
		return i.Map
	}
	return nil
}

// mapItem contains a recorded, local, and remote map
type typeItem struct {
	Name string
	Type *proto.Kind

	apply.MapElementData
}

func (i *typeItem) GetMeta() proto.Schema {
	// https://golang.org/doc/faq#nil_error
	if i.Type != nil {
		return i.Type
	}
	return nil
}

func (i *typeItem) CreateElement(v ItemVisitor) (apply.Element, error) {
	return v.CreateTypeElement(i)
}

// emptyItem contains no values
type emptyItem struct {
	Name string
}

func (i *emptyItem) CreateElement(v ItemVisitor) (apply.Element, error) {
	e := &apply.EmptyElement{}
	e.Name = i.Name
	return e, nil
}
