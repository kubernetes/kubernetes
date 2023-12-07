/*
Copyright 2021 The Kubernetes Authors.

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

package topologycache

import (
	discovery "k8s.io/api/discovery/v1"
)

// SliceInfo stores information about EndpointSlices for the reconciliation
// process.
type SliceInfo struct {
	ServiceKey  string
	AddressType discovery.AddressType
	ToCreate    []*discovery.EndpointSlice
	ToUpdate    []*discovery.EndpointSlice
	Unchanged   []*discovery.EndpointSlice
}

func (si *SliceInfo) getTotalReadyEndpoints() int {
	totalEndpoints := 0
	for _, slice := range si.ToCreate {
		totalEndpoints += numReadyEndpoints(slice.Endpoints)
	}
	for _, slice := range si.ToUpdate {
		totalEndpoints += numReadyEndpoints(slice.Endpoints)
	}
	for _, slice := range si.Unchanged {
		totalEndpoints += numReadyEndpoints(slice.Endpoints)
	}
	return totalEndpoints
}

// getAllocatedHintsByZone sums up the allocated hints we currently have in
// unchanged slices and marks slices for update as necessary. A slice needs to
// be updated if any of the following are true:
//   - It has an endpoint without zone hints
//   - It has an endpoint hint for a zone that no longer needs any
//   - It has endpoint hints that would make the minimum allocations necessary
//     impossible with changes to slices that are already being updated or
//     created.
func (si *SliceInfo) getAllocatedHintsByZone(allocations map[string]allocation) EndpointZoneInfo {
	allocatedHintsByZone := EndpointZoneInfo{}

	// Using filtering in place to remove any endpoints that are no longer
	// unchanged (https://github.com/golang/go/wiki/SliceTricks#filter-in-place)
	j := 0
	for _, slice := range si.Unchanged {
		hintsByZone := getHintsByZone(slice, allocatedHintsByZone, allocations)
		if hintsByZone == nil {
			si.ToUpdate = append(si.ToUpdate, slice.DeepCopy())
		} else {
			si.Unchanged[j] = slice
			j++
			for zone, numHints := range hintsByZone {
				allocatedHintsByZone[zone] += numHints
			}
		}
	}

	si.Unchanged = si.Unchanged[:j]
	return allocatedHintsByZone
}
