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

package topologyheuristics

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
	// Unchanged slices are direct copies from informer cache. We need to deep
	// copy an unchanged slice before making any modifications to it so that we do
	// not modify the slice within the informer cache.
	Unchanged []*discovery.EndpointSlice
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

// getEndpointsWithMissingZone returns the number of ready endpoints that do not
// have a zone assigned.
func (si *SliceInfo) getEndpointsWithMissingZone() int {
	var slices []*discovery.EndpointSlice
	slices = append(slices, si.ToCreate...)
	slices = append(slices, si.ToUpdate...)
	slices = append(slices, si.Unchanged...)

	var result int
	for _, slice := range slices {
		for _, endpoint := range slice.Endpoints {
			if !EndpointReady(endpoint) {
				continue
			}
			if endpoint.Zone == nil || *endpoint.Zone == "" {
				result++
			}
		}
	}
	return result
}
