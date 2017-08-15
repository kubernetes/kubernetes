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
	"k8s.io/kubernetes/pkg/kubectl/apply"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

// ItemVisitor provides an interface for Items to Accept and call
// the Visit function that corresponds to its actual type.
type ItemVisitor interface {
	// VisitPrimitive builds an Element for a primitiveItem
	VisitPrimitive(*primitiveItem) (apply.Element, error)

	// VisitList builds an Element for a listItem
	VisitList(*listItem) (apply.Element, error)

	// VisitMap builds an Element for a mapItem
	VisitMap(*mapItem) (apply.Element, error)

	// VisitType builds an Element for a typeItem
	VisitType(*typeItem) (apply.Element, error)
}

// ElementBuildingVisitor creates an Elements from Items
// An Element combines the values from the Item with the field metadata.
type ElementBuildingVisitor struct {
	resources openapi.Resources
}

// VisitPrimitive creates a primitiveElement
func (v ElementBuildingVisitor) VisitPrimitive(item *primitiveItem) (apply.Element, error) {
	return v.primitiveElement(item)
}

// VisitList creates a ListElement
func (v ElementBuildingVisitor) VisitList(item *listItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta())
	if err != nil {
		return nil, err
	}
	if meta.GetFieldMergeType() == "merge" {
		return v.mergeListElement(meta, item)
	}
	return v.replaceListElement(meta, item)
}

// VisitMap creates a mapElement
func (v ElementBuildingVisitor) VisitMap(item *mapItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta())
	if err != nil {
		return nil, err
	}
	return v.mapElement(meta, item)
}

// VisitType creates a typeElement
func (v ElementBuildingVisitor) VisitType(item *typeItem) (apply.Element, error) {
	meta, err := getFieldMeta(item.GetMeta())
	if err != nil {
		return nil, err
	}
	return v.typeElement(meta, item)
}
