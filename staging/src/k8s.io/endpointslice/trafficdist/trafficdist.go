/*
Copyright 2024 The Kubernetes Authors.

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

// trafficdist handles reconciliation of hints for trafficDistribution field.
package trafficdist

import (
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
)

// ReconcileHints will reconcile hints for the given EndpointSlices.
//
// EndpointSlice resources within slicesUnchanged will not be modified.
func ReconcileHints(trafficDistribution *string, slicesToCreate, slicesToUpdate, slicesUnchanged []*discoveryv1.EndpointSlice) ([]*discoveryv1.EndpointSlice, []*discoveryv1.EndpointSlice, []*discoveryv1.EndpointSlice) {
	var h heuristic = &defaultHeuristic{}

	if trafficDistribution != nil && *trafficDistribution == corev1.ServiceTrafficDistributionPreferClose {
		h = &preferCloseHeuristic{}
	}

	// Identify the Unchanged slices that need an update because of missing or
	// incorrect zone hint.
	//
	// Uses filtering in place to remove any endpoints that are no longer
	// unchanged and need to be moved to slicesToUpdate
	// (https://github.com/golang/go/wiki/SliceTricks#filter-in-place)
	j := 0
	for _, slice := range slicesUnchanged {
		if h.needsUpdate(slice) {
			// Unchanged slices are direct copies from informer cache. We need to deep
			// copy an unchanged slice before making any modifications to it so that we do
			// not modify the slice within the informer cache.
			slicesToUpdate = append(slicesToUpdate, slice.DeepCopy())
		} else {
			slicesUnchanged[j] = slice
			j++
		}
	}
	// Truncate slicesUnchanged so it only includes slices that are still
	// unchanged.
	slicesUnchanged = slicesUnchanged[:j]

	// Add zone hints to all slices that need to be created or updated.
	for _, slice := range slicesToCreate {
		h.update(slice)
	}
	for _, slice := range slicesToUpdate {
		h.update(slice)
	}

	return slicesToCreate, slicesToUpdate, slicesUnchanged
}

type heuristic interface {
	needsUpdate(*discoveryv1.EndpointSlice) bool
	update(*discoveryv1.EndpointSlice)
}

// endpointReady returns true if an Endpoint has the Ready condition set to
// true.
func endpointReady(endpoint discoveryv1.Endpoint) bool {
	return endpoint.Conditions.Ready != nil && *endpoint.Conditions.Ready
}

// defaultHeuristic means cluster wide routing, hence it will remove any hints
// present in the EndpointSlice.
type defaultHeuristic struct {
}

// needsUpdate returns true if any endpoint in the slice has a zone hint.
func (defaultHeuristic) needsUpdate(slice *discoveryv1.EndpointSlice) bool {
	if slice == nil {
		return false
	}
	for _, endpoint := range slice.Endpoints {
		if endpoint.Hints != nil {
			return true
		}
	}
	return false
}

// update removes zone hints from all endpoints.
func (defaultHeuristic) update(slice *discoveryv1.EndpointSlice) {
	for i := range slice.Endpoints {
		slice.Endpoints[i].Hints = nil
	}
}

// preferCloseHeuristic adds
type preferCloseHeuristic struct {
}

// needsUpdate returns true if any ready endpoint in the slice has a
// missing or incorrect hint.
func (preferCloseHeuristic) needsUpdate(slice *discoveryv1.EndpointSlice) bool {
	if slice == nil {
		return false
	}
	for _, endpoint := range slice.Endpoints {
		if !endpointReady(endpoint) {
			continue
		}

		if endpoint.Zone != nil {
			// We want a zone hint.
			if endpoint.Hints == nil || len(endpoint.Hints.ForZones) != 1 || endpoint.Hints.ForZones[0].Name != *endpoint.Zone {
				// ...but either it's missing or it's incorrect
				return true
			}
		} else {
			// We don't want a zone hint.
			if endpoint.Hints != nil && len(endpoint.Hints.ForZones) > 0 {
				// ...but we have a stale hint.
				return true
			}
		}
	}
	return false
}

// update adds a same zone topology hint for all ready endpoints
func (preferCloseHeuristic) update(slice *discoveryv1.EndpointSlice) {
	for i, endpoint := range slice.Endpoints {
		if !endpointReady(endpoint) {
			continue
		}

		var forZones []discoveryv1.ForZone
		if endpoint.Zone != nil {
			forZones = []discoveryv1.ForZone{{Name: *endpoint.Zone}}
		}

		if forZones != nil {
			slice.Endpoints[i].Hints = &discoveryv1.EndpointHints{
				ForZones: forZones,
			}
		} else {
			slice.Endpoints[i].Hints = nil
		}
	}
}
