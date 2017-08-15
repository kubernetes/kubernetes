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

// mapElement builds a new mapElement from a MapItem
func (v ElementBuildingVisitor) typeElement(meta apply.FieldMetaImpl, item *typeItem) (*apply.TypeElement, error) {
	result := &apply.TypeElement{
		Name: item.Name,
		//Schema:   item.MergeType,
		FieldMetaImpl: meta,
		RecordedSet:   item.RecordedSet,
		LocalSet:      item.LocalSet,
		RemoteSet:     item.RemoteSet,
		Recorded:      item.Recorded,
		Local:         item.Local,
		Remote:        item.Remote,
		Values:        map[string]apply.Element{},
	}

	// Collate each key in the type
	for _, key := range keysUnion(item.Recorded, item.Local, item.Remote) {
		var s openapi.Schema
		if item.Type != nil && item.Type.Fields != nil {
			s = item.Type.Fields[key]
		}

		recorded, recordedSet := nilSafeLookup(key, item.Recorded)
		local, localSet := nilSafeLookup(key, item.Local)
		remote, remoteSet := nilSafeLookup(key, item.Remote)

		// Create an item for the field
		field, err := v.getItem(s, key, recorded, local, remote, recordedSet, localSet, remoteSet)
		if err != nil {
			return nil, err
		}

		// Build the element for this field
		element, err := field.Accept(v)
		if err != nil {
			return nil, err
		}

		// Add the field element to the map
		result.Values[key] = element
	}
	return result, nil
}
