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
	"sync"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

const (
	// overloadThreshold represents the maximum overload any individual endpoint
	// should be exposed to.
	overloadThreshold float64 = 0.2
)

// TopologyCache tracks the distribution of Nodes and endpoints across zones.
type TopologyCache struct {
	lock                    sync.Mutex
	sufficientNodeInfo      bool
	cpuByZone               map[string]*resource.Quantity
	cpuRatiosByZone         map[string]float64
	endpointsByService      map[string]map[discovery.AddressType]EndpointZoneInfo
	hintsPopulatedByService sets.Set[string]
}

// EndpointZoneInfo tracks the distribution of endpoints across zones for a
// Service.
type EndpointZoneInfo map[string]int

// allocation describes the number of endpoints that should be allocated for a
// zone.
type allocation struct {
	minimum int
	maximum int
	desired float64
}

// NewTopologyCache initializes a new TopologyCache.
func NewTopologyCache() *TopologyCache {
	return &TopologyCache{
		cpuByZone:               map[string]*resource.Quantity{},
		cpuRatiosByZone:         map[string]float64{},
		endpointsByService:      map[string]map[discovery.AddressType]EndpointZoneInfo{},
		hintsPopulatedByService: sets.Set[string]{},
	}
}

// GetOverloadedServices returns a list of Service keys that refer to Services
// that have crossed the overload threshold for any zone.
func (t *TopologyCache) GetOverloadedServices() []string {
	t.lock.Lock()
	defer t.lock.Unlock()

	svcKeys := []string{}
	for svcKey, eziByAddrType := range t.endpointsByService {
		for _, ezi := range eziByAddrType {
			if serviceOverloaded(ezi, t.cpuRatiosByZone) {
				svcKeys = append(svcKeys, svcKey)
				break
			}
		}
	}

	return svcKeys
}

// AddHints adds or updates topology hints on EndpointSlices and returns updated
// lists of EndpointSlices to create and update.
func (t *TopologyCache) AddHints(logger klog.Logger, si *SliceInfo) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice, []*EventBuilder) {
	totalEndpoints := si.getTotalReadyEndpoints()
	allocations, allocationsEvent := t.getAllocations(totalEndpoints)
	events := []*EventBuilder{}
	if allocationsEvent != nil {
		logger.Info(allocationsEvent.Message+", removing hints", "key", si.ServiceKey, "addressType", si.AddressType)
		allocationsEvent.Message = FormatWithAddressType(allocationsEvent.Message, si.AddressType)
		events = append(events, allocationsEvent)
		t.RemoveHints(si.ServiceKey, si.AddressType)
		slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
		return slicesToCreate, slicesToUpdate, events
	}

	allocatedHintsByZone := si.getAllocatedHintsByZone(allocations)

	allocatableSlices := si.ToCreate
	for _, slice := range si.ToUpdate {
		allocatableSlices = append(allocatableSlices, slice)
	}

	// step 1: assign same-zone hints for all endpoints as a starting point.
	for _, slice := range allocatableSlices {
		for i, endpoint := range slice.Endpoints {
			if !EndpointReady(endpoint) {
				endpoint.Hints = nil
				continue
			}
			if endpoint.Zone == nil || *endpoint.Zone == "" {
				logger.Info("Endpoint found without zone specified, removing hints", "key", si.ServiceKey, "addressType", si.AddressType)
				events = append(events, &EventBuilder{
					EventType: v1.EventTypeWarning,
					Reason:    "TopologyAwareHintsDisabled",
					Message:   FormatWithAddressType(NoZoneSpecified, si.AddressType),
				})
				t.RemoveHints(si.ServiceKey, si.AddressType)
				slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
				return slicesToCreate, slicesToUpdate, events
			}

			allocatedHintsByZone[*endpoint.Zone]++
			slice.Endpoints[i].Hints = &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: *endpoint.Zone}}}
		}
	}

	// step 2. Identify which zones need to donate slices and which need more.
	givingZones, receivingZones := getGivingAndReceivingZones(allocations, allocatedHintsByZone)

	// step 3. Redistribute endpoints based on data from step 2.
	redistributions := redistributeHints(logger, allocatableSlices, givingZones, receivingZones)

	for zone, diff := range redistributions {
		allocatedHintsByZone[zone] += diff
	}

	if len(allocatedHintsByZone) == 0 {
		logger.V(2).Info("No hints allocated for zones, removing them", "key", si.ServiceKey, "addressType", si.AddressType)
		events = append(events, &EventBuilder{
			EventType: v1.EventTypeWarning,
			Reason:    "TopologyAwareHintsDisabled",
			Message:   FormatWithAddressType(NoAllocatedHintsForZones, si.AddressType),
		})
		t.RemoveHints(si.ServiceKey, si.AddressType)
		slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
		return slicesToCreate, slicesToUpdate, events
	}

	t.lock.Lock()
	defer t.lock.Unlock()
	hintsEnabled := t.hasPopulatedHintsLocked(si.ServiceKey)
	t.setHintsLocked(si.ServiceKey, si.AddressType, allocatedHintsByZone)

	// if hints were not enabled before, we publish an event to indicate we enabled them.
	if !hintsEnabled {
		logger.Info("Topology Aware Hints has been enabled, adding hints.", "key", si.ServiceKey, "addressType", si.AddressType)
		events = append(events, &EventBuilder{
			EventType: v1.EventTypeNormal,
			Reason:    "TopologyAwareHintsEnabled",
			Message:   FormatWithAddressType(TopologyAwareHintsEnabled, si.AddressType),
		})
	}
	return si.ToCreate, si.ToUpdate, events
}

// SetHints sets topology hints for the provided serviceKey and addrType in this
// cache.
func (t *TopologyCache) SetHints(serviceKey string, addrType discovery.AddressType, allocatedHintsByZone EndpointZoneInfo) {
	t.lock.Lock()
	defer t.lock.Unlock()

	t.setHintsLocked(serviceKey, addrType, allocatedHintsByZone)
}

func (t *TopologyCache) setHintsLocked(serviceKey string, addrType discovery.AddressType, allocatedHintsByZone EndpointZoneInfo) {
	_, ok := t.endpointsByService[serviceKey]
	if !ok {
		t.endpointsByService[serviceKey] = map[discovery.AddressType]EndpointZoneInfo{}
	}
	t.endpointsByService[serviceKey][addrType] = allocatedHintsByZone

	t.hintsPopulatedByService.Insert(serviceKey)
}

// RemoveHints removes topology hints for the provided serviceKey and addrType
// from this cache.
func (t *TopologyCache) RemoveHints(serviceKey string, addrType discovery.AddressType) {
	t.lock.Lock()
	defer t.lock.Unlock()

	_, ok := t.endpointsByService[serviceKey]
	if ok {
		delete(t.endpointsByService[serviceKey], addrType)
	}
	if len(t.endpointsByService[serviceKey]) == 0 {
		delete(t.endpointsByService, serviceKey)
	}
	t.hintsPopulatedByService.Delete(serviceKey)
}

// SetNodes updates the Node distribution for the TopologyCache.
func (t *TopologyCache) SetNodes(logger klog.Logger, nodes []*v1.Node) {
	cpuByZone := map[string]*resource.Quantity{}
	sufficientNodeInfo := true

	totalCPU := resource.Quantity{}

	for _, node := range nodes {
		if hasExcludedLabels(node.Labels) {
			logger.V(2).Info("Ignoring node because it has an excluded label", "node", klog.KObj(node))
			continue
		}
		if !isNodeReady(node) {
			logger.V(2).Info("Ignoring node because it is not ready", "node", klog.KObj(node))
			continue
		}

		nodeCPU := node.Status.Allocatable.Cpu()
		zone, ok := node.Labels[v1.LabelTopologyZone]

		// TODO(robscott): Figure out if there's an acceptable proportion of
		// nodes with inadequate information. The current logic means that as
		// soon as we find any node without a zone or allocatable CPU specified,
		// we bail out entirely. Bailing out at this level will make our cluster
		// wide ratios nil, which would result in slices for all Services having
		// their hints removed.
		if !ok || zone == "" || nodeCPU.IsZero() {
			cpuByZone = map[string]*resource.Quantity{}
			sufficientNodeInfo = false
			logger.Info("Can't get CPU or zone information for node", "node", klog.KObj(node))
			break
		}

		totalCPU.Add(*nodeCPU)
		if _, ok = cpuByZone[zone]; !ok {
			cpuByZone[zone] = nodeCPU
		} else {
			cpuByZone[zone].Add(*nodeCPU)
		}
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	if totalCPU.IsZero() || !sufficientNodeInfo || len(cpuByZone) < 2 {
		logger.V(2).Info("Insufficient node info for topology hints", "totalZones", len(cpuByZone), "totalCPU", totalCPU.String(), "sufficientNodeInfo", sufficientNodeInfo)
		t.sufficientNodeInfo = false
		t.cpuByZone = nil
		t.cpuRatiosByZone = nil

	} else {
		t.sufficientNodeInfo = sufficientNodeInfo
		t.cpuByZone = cpuByZone

		t.cpuRatiosByZone = map[string]float64{}
		for zone, cpu := range cpuByZone {
			t.cpuRatiosByZone[zone] = float64(cpu.MilliValue()) / float64(totalCPU.MilliValue())
		}
	}
}

// HasPopulatedHints checks whether there are populated hints for a given service in the cache.
func (t *TopologyCache) HasPopulatedHints(serviceKey string) bool {
	t.lock.Lock()
	defer t.lock.Unlock()

	return t.hasPopulatedHintsLocked(serviceKey)
}

func (t *TopologyCache) hasPopulatedHintsLocked(serviceKey string) bool {
	return t.hintsPopulatedByService.Has(serviceKey)
}

// getAllocations returns a set of minimum and maximum allocations per zone. If
// it is not possible to provide allocations that are below the overload
// threshold, a nil value will be returned.
func (t *TopologyCache) getAllocations(numEndpoints int) (map[string]allocation, *EventBuilder) {
	t.lock.Lock()
	defer t.lock.Unlock()

	// it is similar to checking !t.sufficientNodeInfo
	if t.cpuRatiosByZone == nil {
		return nil, &EventBuilder{
			EventType: v1.EventTypeWarning,
			Reason:    "TopologyAwareHintsDisabled",
			Message:   InsufficientNodeInfo,
		}
	}
	if len(t.cpuRatiosByZone) < 2 {
		return nil, &EventBuilder{
			EventType: v1.EventTypeWarning,
			Reason:    "TopologyAwareHintsDisabled",
			Message:   NodesReadyInOneZoneOnly,
		}
	}
	if len(t.cpuRatiosByZone) > numEndpoints {
		return nil, &EventBuilder{
			EventType: v1.EventTypeWarning,
			Reason:    "TopologyAwareHintsDisabled",
			Message:   fmt.Sprintf("%s (%d endpoints, %d zones)", InsufficientNumberOfEndpoints, numEndpoints, len(t.cpuRatiosByZone)),
		}
	}

	remainingMinEndpoints := numEndpoints
	minTotal := 0
	allocations := map[string]allocation{}

	for zone, ratio := range t.cpuRatiosByZone {
		desired := ratio * float64(numEndpoints)
		minimum := int(math.Ceil(desired * (1 / (1 + overloadThreshold))))
		allocations[zone] = allocation{
			minimum: minimum,
			desired: math.Max(desired, float64(minimum)),
		}
		minTotal += minimum
		remainingMinEndpoints -= minimum
		if remainingMinEndpoints < 0 {
			return nil, &EventBuilder{
				EventType: v1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   fmt.Sprintf("%s (%d endpoints, %d zones)", MinAllocationExceedsOverloadThreshold, numEndpoints, len(t.cpuRatiosByZone)),
			}
		}
	}

	for zone, allocation := range allocations {
		allocation.maximum = allocation.minimum + numEndpoints - minTotal
		allocations[zone] = allocation
	}

	return allocations, nil
}

// Nodes with any of these labels set to any value will be excluded from
// topology capacity calculations.
func hasExcludedLabels(labels map[string]string) bool {
	if len(labels) == 0 {
		return false
	}
	if _, ok := labels["node-role.kubernetes.io/control-plane"]; ok {
		return true
	}
	if _, ok := labels["node-role.kubernetes.io/master"]; ok {
		return true
	}
	return false
}
