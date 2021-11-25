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
	"math"
	"sync"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	endpointsliceutil "k8s.io/kubernetes/pkg/controller/util/endpointslice"
)

const (
	// OverloadThreshold represents the maximum overload any individual endpoint
	// should be exposed to.
	OverloadThreshold float64 = 0.2
)

// TopologyCache tracks the distribution of Nodes and endpoints across zones.
type TopologyCache struct {
	lock               sync.Mutex
	sufficientNodeInfo bool
	cpuByZone          map[string]*resource.Quantity
	cpuRatiosByZone    map[string]float64
	endpointsByService map[string]map[discovery.AddressType]EndpointZoneInfo
}

// EndpointZoneInfo tracks the distribution of endpoints across zones for a
// Service.
type EndpointZoneInfo map[string]int

// Allocation describes the number of endpoints that should be allocated for a
// zone.
type Allocation struct {
	Minimum int
	Maximum int
	Desired float64
}

// NewTopologyCache initializes a new TopologyCache.
func NewTopologyCache() *TopologyCache {
	return &TopologyCache{
		cpuByZone:          map[string]*resource.Quantity{},
		cpuRatiosByZone:    map[string]float64{},
		endpointsByService: map[string]map[discovery.AddressType]EndpointZoneInfo{},
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
func (t *TopologyCache) AddHints(si *SliceInfo) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice) {
	totalEndpoints := si.getTotalReadyEndpoints()
	allocations := t.getAllocations(totalEndpoints)

	if allocations == nil {
		klog.V(2).InfoS("Insufficient endpoints, removing hints from service", "serviceKey", si.ServiceKey)
		t.RemoveHints(si.ServiceKey, si.AddressType)
		return RemoveHintsFromSlices(si)
	}

	allocatedHintsByZone := si.getAllocatedHintsByZone(allocations)

	allocatableSlices := si.ToCreate
	for _, slice := range si.ToUpdate {
		allocatableSlices = append(allocatableSlices, slice)
	}

	// step 1: assign same-zone hints for all endpoints as a starting point.
	for _, slice := range allocatableSlices {
		for i, endpoint := range slice.Endpoints {
			if !endpointsliceutil.EndpointReady(endpoint) {
				endpoint.Hints = nil
				continue
			}
			if endpoint.Zone == nil || *endpoint.Zone == "" {
				klog.InfoS("Endpoint found without zone specified, removing hints from service", "serviceKey", si.ServiceKey)
				t.RemoveHints(si.ServiceKey, si.AddressType)
				return RemoveHintsFromSlices(si)
			}

			allocatedHintsByZone[*endpoint.Zone]++
			slice.Endpoints[i].Hints = &discovery.EndpointHints{ForZones: []discovery.ForZone{{Name: *endpoint.Zone}}}
		}
	}

	// step 2. Identify which zones need to donate slices and which need more.
	givingZones, receivingZones := getGivingAndReceivingZones(allocations, allocatedHintsByZone)

	// step 3. Redistribute endpoints based on data from step 2.
	redistributions := redistributeHints(allocatableSlices, givingZones, receivingZones)

	for zone, diff := range redistributions {
		allocatedHintsByZone[zone] += diff
	}

	t.SetHints(si.ServiceKey, si.AddressType, allocatedHintsByZone)
	return si.ToCreate, si.ToUpdate
}

// SetHints sets topology hints for the provided serviceKey and addrType in this
// cache.
func (t *TopologyCache) SetHints(serviceKey string, addrType discovery.AddressType, allocatedHintsByZone EndpointZoneInfo) {
	if len(allocatedHintsByZone) == 0 {
		klog.V(2).Infof("No hints allocated for zones, removing them from %s EndpointSlices for %s Service", addrType, serviceKey)
		t.RemoveHints(serviceKey, addrType)
		return
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	_, ok := t.endpointsByService[serviceKey]
	if !ok {
		t.endpointsByService[serviceKey] = map[discovery.AddressType]EndpointZoneInfo{}
	}
	t.endpointsByService[serviceKey][addrType] = allocatedHintsByZone
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
}

// SetNodes updates the Node distribution for the TopologyCache.
func (t *TopologyCache) SetNodes(nodes []*v1.Node) {
	cpuByZone := map[string]*resource.Quantity{}
	sufficientNodeInfo := true

	totalCPU := resource.Quantity{}

	for _, node := range nodes {
		if hasExcludedLabels(node.Labels) {
			klog.V(2).Infof("Ignoring node %s because it has an excluded label", node.Name)
			continue
		}
		if !NodeReady(node.Status) {
			klog.V(2).Infof("Ignoring node %s because it is not ready: %v", node.Name, node.Status.Conditions)
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
			klog.Warningf("Can't get CPU or zone information for %s node", node.Name)
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
		klog.V(2).Infof("Insufficient node info for topology hints (%d zones, %s CPU, %t)", len(cpuByZone), totalCPU.MilliValue(), sufficientNodeInfo)
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

// getAllocations returns a set of minimum and maximum allocations per zone. If
// it is not possible to provide allocations that are below the overload
// threshold, a nil value will be returned.
func (t *TopologyCache) getAllocations(numEndpoints int) map[string]Allocation {
	if t.cpuRatiosByZone == nil || len(t.cpuRatiosByZone) < 2 || len(t.cpuRatiosByZone) > numEndpoints {
		klog.V(2).Infof("Insufficient info to allocate endpoints (%d endpoints, %d zones)", numEndpoints, len(t.cpuRatiosByZone))
		return nil
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	remainingMinEndpoints := numEndpoints
	minTotal := 0
	allocations := map[string]Allocation{}

	for zone, ratio := range t.cpuRatiosByZone {
		desired := ratio * float64(numEndpoints)
		minimum := int(math.Ceil(desired * (1 / (1 + OverloadThreshold))))
		allocations[zone] = Allocation{
			Minimum: minimum,
			Desired: math.Max(desired, float64(minimum)),
		}
		minTotal += minimum
		remainingMinEndpoints -= minimum
		if remainingMinEndpoints < 0 {
			return nil
		}
	}

	for zone, allocation := range allocations {
		allocation.Maximum = allocation.Minimum + numEndpoints - minTotal
		allocations[zone] = allocation
	}

	return allocations
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
