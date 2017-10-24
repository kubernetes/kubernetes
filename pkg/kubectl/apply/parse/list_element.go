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
	"fmt"

	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubernetes/pkg/kubectl/apply"
)

// Contains the heavy lifting for finding tuples of matching elements in lists based on the merge key
// and then uses the canonical order derived from the orders in the recorded, local and remote lists.

// replaceListElement builds a ListElement for a listItem.
// Uses the "merge" strategy to identify "same" elements across lists by a "merge key"
func (v ElementBuildingVisitor) mergeListElement(meta apply.FieldMetaImpl, item *listItem) (*apply.ListElement, error) {
	subtype := getSchemaType(item.Array.SubType)
	switch subtype {
	case "primitive":
		return v.doPrimitiveList(meta, item)
	case "map", "kind", "reference":
		return v.doMapList(meta, item)
	default:
		return nil, fmt.Errorf("Cannot merge lists with subtype %s", subtype)
	}
}

// doPrimitiveList merges 3 lists of primitives together
// tries to maintain ordering
func (v ElementBuildingVisitor) doPrimitiveList(meta apply.FieldMetaImpl, item *listItem) (*apply.ListElement, error) {
	result := &apply.ListElement{
		FieldMetaImpl: apply.FieldMetaImpl{
			MergeType: "merge",
			Name:      item.Name,
		},
		ListElementData: item.ListElementData,
		Values:          []apply.Element{},
	}

	// Use locally defined order, then add remote, then add recorded.
	orderedKeys := &apply.CombinedPrimitiveSlice{}

	// Locally defined items come first and retain their order
	// as defined locally
	for _, l := range item.GetLocalList() {
		orderedKeys.UpsertLocal(l)
	}
	// Mixin remote values, adding any that are not present locally
	for _, l := range item.GetRemoteList() {
		orderedKeys.UpsertRemote(l)
	}
	// Mixin recorded values, adding any that are not present locally
	// or remotely
	for _, l := range item.GetRecordedList() {
		orderedKeys.UpsertRecorded(l)
	}

	for i, l := range orderedKeys.Items {
		var s proto.Schema
		if item.Array != nil && item.Array.SubType != nil {
			s = item.Array.SubType
		}

		subitem, err := v.getItem(s, fmt.Sprintf("%d", i), l.RawElementData)

		if err != nil {
			return nil, err
		}

		// Convert the Item to an Element
		newelem, err := subitem.CreateElement(v)
		if err != nil {
			return nil, err
		}

		// Append the element to the list
		result.Values = append(result.Values, newelem)
	}

	return result, nil
}

// doMapList merges 3 lists of maps together by collating their values.
// tries to retain ordering
func (v ElementBuildingVisitor) doMapList(meta apply.FieldMetaImpl, item *listItem) (*apply.ListElement, error) {
	key := meta.GetFieldMergeKeys()
	result := &apply.ListElement{
		FieldMetaImpl: apply.FieldMetaImpl{
			MergeType: "merge",
			MergeKeys: key,
			Name:      item.Name,
		},
		ListElementData: item.ListElementData,
		Values:          []apply.Element{},
	}

	// Use locally defined order, then add remote, then add recorded.
	orderedKeys := &apply.CombinedMapSlice{}

	// Locally defined items come first and retain their order
	// as defined locally
	for _, l := range item.GetLocalList() {
		orderedKeys.UpsertLocal(key, l)
	}
	// Mixin remote values, adding any that are not present locally
	for _, l := range item.GetRemoteList() {
		orderedKeys.UpsertRemote(key, l)
	}
	// Mixin recorded values, adding any that are not present locally
	// or remotely
	for _, l := range item.GetRecordedList() {
		orderedKeys.UpsertRecorded(key, l)
	}

	for i, l := range orderedKeys.Items {
		var s proto.Schema
		if item.Array != nil && item.Array.SubType != nil {
			s = item.Array.SubType
		}
		subitem, err := v.getItem(s, fmt.Sprintf("%d", i), l.RawElementData)
		if err != nil {
			return nil, err
		}

		// Build the element fully
		newelem, err := subitem.CreateElement(v)
		if err != nil {
			return nil, err
		}

		// Append the element to the list
		result.Values = append(result.Values, newelem)
	}

	return result, nil
}

// replaceListElement builds a new ListElement from a listItem
// Uses the "replace" strategy and identify "same" elements across lists by their index
func (v ElementBuildingVisitor) replaceListElement(meta apply.FieldMetaImpl, item *listItem) (*apply.ListElement, error) {
	meta.Name = item.Name
	result := &apply.ListElement{
		FieldMetaImpl:   meta,
		ListElementData: item.ListElementData,
		Values:          []apply.Element{},
	}

	// Use the max length to iterate over the slices
	for i := 0; i < max(len(item.GetRecordedList()), len(item.GetLocalList()), len(item.GetRemoteList())); i++ {

		// Lookup the item from each list
		data := apply.RawElementData{}
		if recorded, recordedSet := boundsSafeLookup(i, item.GetRecordedList()); recordedSet {
			data.SetRecorded(recorded)
		}
		if local, localSet := boundsSafeLookup(i, item.GetLocalList()); localSet {
			data.SetLocal(local)
		}
		if remote, remoteSet := boundsSafeLookup(i, item.GetRemoteList()); remoteSet {
			data.SetRemote(remote)
		}

		// Create the Item
		var s proto.Schema
		if item.Array != nil && item.Array.SubType != nil {
			s = item.Array.SubType
		}
		subitem, err := v.getItem(s, fmt.Sprintf("%d", i), data)
		if err != nil {
			return nil, err
		}

		// Build the element
		newelem, err := subitem.CreateElement(v)
		if err != nil {
			return nil, err
		}

		// Append the element to the list
		result.Values = append(result.Values, newelem)
	}

	return result, nil
}
