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
	"k8s.io/kubectl/pkg/apply"
	"k8s.io/kubectl/pkg/util/openapi"
)

// ItemVisitor provides an interface for Items to Accept and call
// the Visit function that corresponds to its actual type.
type ItemVisitor interface {
	// CreatePrimitiveElement builds an Element for a primitiveItem
	CreatePrimitiveElement(*primitiveItem) (apply.Element, error)

	// CreateListElement builds an Element for a listItem
	CreateListElement(*listItem) (apply.Element, error)

	// CreateMapElement builds an Element for a mapItem
	CreateMapElement(*mapItem) (apply.Element, error)

	// CreateTypeElement builds an Element for a typeItem
	CreateTypeElement(*typeItem) (apply.Element, error)
}

// ElementBuildingVisitor creates an Elements from Items
// An Element combines the values from the Item with the field metadata.
type ElementBuildingVisitor struct {
	resources openapi.Resources
}

// CreatePrimitiveElement creates a primitiveElement
func (v ElementBuildingVisitor) CreatePrimitiveElement(item *primitiveItem) (apply.Element, error) {
	return v.primitiveElement(item)
}

// CreateListElement creates a ListElement
func (v ElementBuildingVisitor) CreateListElement(item *listItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta(), item.Name)
	if err != nil {
		return nil, err
	}
	if meta.GetFieldMergeType() == apply.MergeStrategy {
		return v.mergeListElement(meta, item)
	}
	return v.replaceListElement(meta, item)
}

// CreateMapElement creates a mapElement
func (v ElementBuildingVisitor) CreateMapElement(item *mapItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta(), item.Name)
	if err != nil {
		return nil, err
	}
	return v.mapElement(meta, item)
}

// CreateTypeElement creates a typeElement
func (v ElementBuildingVisitor) CreateTypeElement(item *typeItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta(), item.Name)
	if err != nil {
		return nil, err
	}
	return v.typeElement(meta, item)
}
