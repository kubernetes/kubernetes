/*
Copyright 2023 The Kubernetes Authors.

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
	"sync"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

// PreferZoneHeuristic is a heuristic that provides EndpointSlice topology hints
// using the [PreferZone Heuristic].
//
// [PreferZone Heuristic]: https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/2433-topology-aware-hints#preferzone-heuristic
type PreferZoneHeuristic struct {
	mu sync.Mutex
	// hintsPopulatedByService tracks whether this heuristic has been used to
	// populate hints for a given service. It helps identify scenarios like when
	// hints are being populated for the first time.
	hintsPopulatedByService sets.Set[string]
}

// PreferZoneHeuristic implements the Heuristic interface.
var _ Heuristic = &PreferZoneHeuristic{}

func NewPreferZoneHeuristic() *PreferZoneHeuristic {
	return &PreferZoneHeuristic{
		hintsPopulatedByService: sets.Set[string]{},
	}
}

// Name returns the name associated with the PreferZoneHeuristic. This will be
// matched against the topology annotation in the Service to decide if the
// Heuristic is currently active for a service.
func (t *PreferZoneHeuristic) Name() string {
	return "PreferZone"
}

// PopulateHints populates topology hints on EndpointSlices and returns
// updated lists of EndpointSlices to create and update. It also returns any
// Events that need to be recorded for the Service.
//
// Combination of [SliceInfo.ServiceKey and SliceInfo.AddressType] are used to
// identify the endpoint slices which these hints will be associated with.
func (t *PreferZoneHeuristic) PopulateHints(logger klog.Logger, si *SliceInfo) (slicesToCreate []*discoveryv1.EndpointSlice, slicesToUpdate []*discoveryv1.EndpointSlice, events []*EventBuilder) {
	// Step 1: Ensure that all endpoints within EndpointSlices have a zone
	// assigned.
	if si.getEndpointsWithMissingZone() != 0 {
		logger.Info("Endpoint found without zone specified, removing hints", "key", si.ServiceKey, "addressType", si.AddressType)
		events = append(events, &EventBuilder{
			EventType: v1.EventTypeWarning,
			Reason:    "TopologyAwareHintsDisabled",
			Message:   FormatWithAddressTypeAndHeuristicName(NoZoneSpecified, si.AddressType, t.Name()),
		})
		t.ClearCachedHints(logger, si.ServiceKey, si.AddressType)
		slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
		return slicesToCreate, slicesToUpdate, events
	}

	// Step 2: Identify the Unchanged slices that need an update because of
	// missing or incorrect zone hint.
	//
	// Using filtering in place to remove any endpoints that are no longer
	// Unchanged and need to be moved to ToUpdate
	// (https://github.com/golang/go/wiki/SliceTricks#filter-in-place)
	j := 0
	for _, slice := range si.Unchanged {
		if t.sliceNeedsUpdate(slice) {
			// Refer to the comment in SliceInfo.Unchanged to understand the need to
			// DeepCopy.
			si.ToUpdate = append(si.ToUpdate, slice.DeepCopy())
		} else {
			si.Unchanged[j] = slice
			j++
		}
	}
	// truncate si.Unchanged so it only includes slices that are still
	// unchanged.
	si.Unchanged = si.Unchanged[:j]

	// Step 3: Add same zone hints to all slices that need to be created or
	// updated.
	t.addSameZoneHint(si.ToCreate)
	t.addSameZoneHint(si.ToUpdate)

	var newlyAddedHints bool
	t.mu.Lock()
	if !t.hintsPopulatedByService.Has(si.ServiceKey) {
		t.hintsPopulatedByService.Insert(si.ServiceKey)
		newlyAddedHints = true
	}
	t.mu.Unlock()

	if newlyAddedHints {
		logger.Info("Topology Aware Hints has been enabled, adding hints.", "key", si.ServiceKey, "addressType", si.AddressType, "topology", t.Name())
		events = append(events, &EventBuilder{
			EventType: v1.EventTypeNormal,
			Reason:    "TopologyAwareHintsEnabled",
			Message:   FormatWithAddressTypeAndHeuristicName(TopologyAwareHintsEnabled, si.AddressType, t.Name()),
		})
	}

	return si.ToCreate, si.ToUpdate, events
}

// sliceNeedsUpdate returns true if any ready endpoint in the slice has a
// missing or incorrect hint.
func (t *PreferZoneHeuristic) sliceNeedsUpdate(slice *discoveryv1.EndpointSlice) bool {
	if slice == nil {
		return false
	}
	for _, endpoint := range slice.Endpoints {
		if !EndpointReady(endpoint) {
			continue
		}
		var zone string
		if endpoint.Zone != nil {
			zone = *endpoint.Zone
		}

		if endpoint.Hints == nil || len(endpoint.Hints.ForZones) != 1 || endpoint.Hints.ForZones[0].Name != zone {
			return true
		}
	}
	return false
}

// addSameZoneHint adds a same zone topology hint for all ready endpoints
func (t *PreferZoneHeuristic) addSameZoneHint(slices []*discoveryv1.EndpointSlice) {
	for _, slice := range slices {
		for i, endpoint := range slice.Endpoints {
			if !EndpointReady(endpoint) {
				endpoint.Hints = nil
				continue
			}

			var zone string
			if endpoint.Zone != nil {
				zone = *endpoint.Zone
			}
			slice.Endpoints[i].Hints = &discoveryv1.EndpointHints{ForZones: []discoveryv1.ForZone{{Name: zone}}}
		}
	}
}

// ClearCachedHints removes any cached topology hints associated with the
// [service and addrType].
func (t *PreferZoneHeuristic) ClearCachedHints(logger klog.Logger, serviceKey string, addrType discoveryv1.AddressType) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.hintsPopulatedByService.Delete(serviceKey)
}
