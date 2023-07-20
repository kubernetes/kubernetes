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
	"fmt"
	"math"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
)

// RemoveHintsFromSlices removes topology hints on EndpointSlices and returns
// updated lists of EndpointSlices to create and update.
func RemoveHintsFromSlices(si *SliceInfo) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice) {
	// Remove hints on all EndpointSlices we were already going to change.
	slices := append(si.ToCreate, si.ToUpdate...)
	for _, slice := range slices {
		for i := range slice.Endpoints {
			slice.Endpoints[i].Hints = nil
		}
	}

	// Remove hints on all unchanged EndpointSlices and mark them for update
	// if any already had hints. We use j to track the number/index of slices
	// that are still unchanged.
	j := 0
	for _, slice := range si.Unchanged {
		changed := false
		for i, endpoint := range slice.Endpoints {
			if endpoint.Hints != nil {
				// Unchanged slices are still direct copies from informer cache.
				// Need to deep copy before we make any modifications to avoid
				// accidentally changing informer cache.
				slice = slice.DeepCopy()
				slice.Endpoints[i].Hints = nil
				changed = true
			}
		}
		if changed {
			si.ToUpdate = append(si.ToUpdate, slice)
		} else {
			si.Unchanged[j] = slice
			j++
		}
	}

	// truncate si.Unchanged so it only includes slices that are still
	// unchanged.
	si.Unchanged = si.Unchanged[:j]

	return si.ToCreate, si.ToUpdate
}

// FormatWithAddressType foramts a given string by adding an addressType to the end of it.
func FormatWithAddressType(s string, addressType discovery.AddressType) string {
	return fmt.Sprintf("%s, addressType: %s", s, addressType)
}

// redistributeHints redistributes hints based in the provided EndpointSlices.
// It allocates endpoints from the provided givingZones to the provided
// receivingZones. This returns a map that represents the changes in allocated
// endpoints by zone.
func redistributeHints(logger klog.Logger, slices []*discovery.EndpointSlice, givingZones, receivingZones map[string]int) map[string]int {
	redistributions := map[string]int{}

	for _, slice := range slices {
		for i, endpoint := range slice.Endpoints {
			if !EndpointReady(endpoint) {
				endpoint.Hints = nil
				continue
			}
			if len(givingZones) == 0 || len(receivingZones) == 0 {
				return redistributions
			}
			if endpoint.Zone == nil || *endpoint.Zone == "" {
				// This should always be caught earlier in AddHints()
				logger.Info("Endpoint found without zone specified")
				continue
			}

			givingZone := *endpoint.Zone
			numToGive, ok := givingZones[givingZone]
			if ok && numToGive > 0 {
				for receivingZone, numToReceive := range receivingZones {
					if numToReceive > 0 {
						slice.Endpoints[i].Hints = &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: receivingZone}}}
						if numToGive == 1 {
							delete(givingZones, givingZone)
						} else {
							givingZones[givingZone]--
						}
						if numToReceive == 1 {
							delete(receivingZones, receivingZone)
						} else {
							receivingZones[receivingZone]--
						}

						redistributions[receivingZone]++
						redistributions[givingZone]--

						break
					}
				}
			}
		}
	}
	return redistributions
}

// getGivingAndReceivingZones returns the number of endpoints each zone should
// give to other zones along with the number of endpoints each zone should
// receive from other zones. This is calculated with the provided allocations
// (desired state) and allocatedHintsByZone (current state).
func getGivingAndReceivingZones(allocations map[string]allocation, allocatedHintsByZone map[string]int) (map[string]int, map[string]int) {
	// 1. Determine the precise number of additional endpoints each zone has
	//    (giving) or needs (receiving).
	givingZonesDesired := map[string]float64{}
	receivingZonesDesired := map[string]float64{}

	for zone, allocation := range allocations {
		allocatedHints, _ := allocatedHintsByZone[zone]
		target := allocation.desired
		if float64(allocatedHints) > target {
			givingZonesDesired[zone] = float64(allocatedHints) - target
		} else if float64(allocatedHints) < target {
			receivingZonesDesired[zone] = target - float64(allocatedHints)
		}
	}

	// 2. Convert the precise numbers needed into ints representing real
	//    endpoints given from one zone to another.
	givingZones := map[string]int{}
	receivingZones := map[string]int{}

	for {
		givingZone, numToGive := getMost(givingZonesDesired)
		receivingZone, numToReceive := getMost(receivingZonesDesired)

		// return early if any of the following are true:
		// - giving OR receiving zone are unspecified
		// - giving AND receiving zones have less than 1 endpoint left to give or receive
		// - giving OR receiving zones have less than 0.5 endpoints left to give or receive
		if givingZone == "" || receivingZone == "" || (numToGive < 1.0 && numToReceive < 1.0) || numToGive < 0.5 || numToReceive < 0.5 {
			break
		}

		givingZones[givingZone]++
		givingZonesDesired[givingZone]--
		receivingZones[receivingZone]++
		receivingZonesDesired[receivingZone]--
	}

	return givingZones, receivingZones
}

// getMost accepts a map[string]float64 and returns the string and float64 that
// represent the greatest value in this provided map. This function is not very
// efficient but it is expected that len() will rarely be greater than 2.
func getMost(zones map[string]float64) (string, float64) {
	zone := ""
	num := 0.0
	for z, n := range zones {
		if n > num {
			zone = z
			num = n
		}
	}

	return zone, num
}

// getHintsByZone returns the number of hints allocated to each zone by the
// provided EndpointSlice. This function returns nil to indicate that the
// current allocations are invalid and that the EndpointSlice needs to be
// updated. This could be caused by:
// - A hint for a zone that no longer requires any allocations.
// - An endpoint with no hints.
// - Hints that would make minimum allocations impossible.
func getHintsByZone(slice *discovery.EndpointSlice, allocatedHintsByZone EndpointZoneInfo, allocations map[string]allocation) map[string]int {
	hintsByZone := map[string]int{}
	for _, endpoint := range slice.Endpoints {
		if !EndpointReady(endpoint) {
			continue
		}
		if endpoint.Hints == nil || len(endpoint.Hints.ForZones) == 0 {
			return nil
		}

		for i := range endpoint.Hints.ForZones {
			zone := endpoint.Hints.ForZones[i].Name
			if _, ok := allocations[zone]; !ok {
				return nil
			}
			hintsByZone[zone]++
		}
	}

	for zone, numHints := range hintsByZone {
		alreadyAllocated, _ := allocatedHintsByZone[zone]
		allocation, ok := allocations[zone]
		if !ok || (numHints+alreadyAllocated) > allocation.maximum {
			return nil
		}
	}

	return hintsByZone
}

// serviceOverloaded returns true if the Service has an insufficient amount of
// endpoints for any zone.
func serviceOverloaded(ezi EndpointZoneInfo, zoneRatios map[string]float64) bool {
	if len(ezi) == 0 {
		return false
	}
	if len(zoneRatios) == 0 {
		return true
	}

	totalEndpoints := 0.0
	for _, numEndpoints := range ezi {
		totalEndpoints += float64(numEndpoints)
	}

	for zone, ratio := range zoneRatios {
		svcEndpoints, ok := ezi[zone]
		if !ok {
			return true
		}
		minEndpoints := math.Ceil(totalEndpoints * ratio * (1 / (1 + overloadThreshold)))
		if svcEndpoints < int(minEndpoints) {
			return true
		}
	}

	return false
}

// isNodeReady returns true if a node is ready; false otherwise.
func isNodeReady(node *v1.Node) bool {
	for _, c := range node.Status.Conditions {
		if c.Type == v1.NodeReady {
			return c.Status == v1.ConditionTrue
		}
	}
	return false
}

// numReadyEndpoints returns the number of Endpoints that are ready from the
// specified list.
func numReadyEndpoints(endpoints []discovery.Endpoint) int {
	var total int
	for _, ep := range endpoints {
		if ep.Conditions.Ready != nil && *ep.Conditions.Ready {
			total++
		}
	}
	return total
}

// EndpointReady returns true if an Endpoint has the Ready condition set to
// true.
func EndpointReady(endpoint discovery.Endpoint) bool {
	return endpoint.Conditions.Ready != nil && *endpoint.Conditions.Ready
}
