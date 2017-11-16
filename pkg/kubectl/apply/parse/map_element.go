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

// mapElement builds a new mapElement from a mapItem
func (v ElementBuildingVisitor) mapElement(meta apply.FieldMetaImpl, item *mapItem) (*apply.MapElement, error) {
	// Function to return schema type of the map values
	var fn schemaFn = func(string) proto.Schema {
		// All map values share the same schema
		if item.Map != nil && item.Map.SubType != nil {
			return item.Map.SubType
		}
		return nil
	}

	// Collect same fields from multiple maps into a map of elements
	values, err := v.createMapValues(fn, meta, item.MapElementData)
	if err != nil {
		return nil, err
	}

	// Return the result
	return &apply.MapElement{
		FieldMetaImpl:  meta,
		MapElementData: item.MapElementData,
		Values:         values,
	}, nil
}

// schemaFn returns the schema for a field or map value based on its name or key
type schemaFn func(key string) proto.Schema

// createMapValues combines the recorded, local and remote values from
// data into a map of elements.
func (v ElementBuildingVisitor) createMapValues(
	schemaFn schemaFn,
	meta apply.FieldMetaImpl,
	data apply.MapElementData) (map[string]apply.Element, error) {

	// Collate each key in the map
	values := map[string]apply.Element{}
	for _, key := range keysUnion(data.GetRecordedMap(), data.GetLocalMap(), data.GetRemoteMap()) {
		combined := apply.RawElementData{}
		if recorded, recordedSet := nilSafeLookup(key, data.GetRecordedMap()); recordedSet {
			combined.SetRecorded(recorded)
		}
		if local, localSet := nilSafeLookup(key, data.GetLocalMap()); localSet {
			combined.SetLocal(local)
		}
		if remote, remoteSet := nilSafeLookup(key, data.GetRemoteMap()); remoteSet {
			combined.SetRemote(remote)
		}

		// Create an item for the field
		field, err := v.getItem(schemaFn(key), key, combined)
		if err != nil {
			return nil, err
		}

		// Build the element for this field
		element, err := field.CreateElement(v)
		if err != nil {
			return nil, err
		}

		// Add the field element to the map
		values[key] = element
	}
	return values, nil
}
