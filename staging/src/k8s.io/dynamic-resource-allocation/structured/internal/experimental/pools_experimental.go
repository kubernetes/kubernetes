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

package experimental

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

func nodeMatches(node *v1.Node, nodeNameToMatch string, allNodesMatch bool, nodeSelector *v1.NodeSelector) (bool, error) {
	switch {
	case nodeNameToMatch != "":
		return node != nil && node.Name == nodeNameToMatch, nil
	case allNodesMatch:
		return true, nil
	case nodeSelector != nil:
		selector, err := nodeaffinity.NewNodeSelector(nodeSelector)
		if err != nil {
			return false, fmt.Errorf("failed to parse node selector %s: %w", nodeSelector.String(), err)
		}
		return selector.Match(node), nil
	}

	return false, nil
}

// GatherPools collects information about all resource pools which provide
// devices that are accessible from the given node.
//
// Out-dated slices are silently ignored. Pools may be incomplete (not all
// required slices available) or invalid (for example, device names not unique).
// Both is recorded in the result.
func GatherPools(ctx context.Context, logger klog.Logger, slices []*resourceapi.ResourceSlice, node *v1.Node, allocatedDeviceIDs sets.Set[DeviceID], features Features) ([]*Pool, error) {
	pools := make(map[PoolID][]*draapi.ResourceSlice)
	var slicesWithBindingConditions []*resourceapi.ResourceSlice
	allocatedDevices := make(map[PoolID][]*resourceapi.Device)

	for _, slice := range slices {
		if !features.PartitionableDevices && slice.Spec.PerDeviceNodeSelection != nil {
			continue
		}

		// Always add slices that contains SharedCounters since they aren't really associated with
		// specified nodes.
		if slice.Spec.SharedCounters != nil {
			if err := addSlice(pools, slice); err != nil {
				return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
			}
			continue
		}

		poolID := PoolID{Driver: draapi.MakeUniqueString(slice.Spec.Driver), Pool: draapi.MakeUniqueString(slice.Spec.Pool.Name)}
		if nodeName, allNodes := ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false); nodeName != "" || allNodes || slice.Spec.NodeSelector != nil {
			match, err := nodeMatches(node, nodeName, allNodes, slice.Spec.NodeSelector)
			if err != nil {
				return nil, fmt.Errorf("failed to perform node selection for slice %s: %w", slice.Name, err)
			}

			// We need all allocated devices to compute the available counters in a resource pool.
			for i := range slice.Spec.Devices {
				device := slice.Spec.Devices[i]
				if allocatedDeviceIDs.Has(MakeDeviceID(slice.Spec.Driver, slice.Spec.Pool.Name, device.Name)) {
					allocatedDevices[poolID] = append(allocatedDevices[poolID], &device)
				}
			}

			if match {
				if hasBindingConditions(slice) {
					// If there is a Device in the ResourceSlice that contains BindingConditions,
					// the ResourceSlice should be sorted to be after the ResourceSlice without BindingConditions
					// because then the allocation is going to prefer the simpler devices without
					// binding conditions.
					slicesWithBindingConditions = append(slicesWithBindingConditions, slice)
					continue
				}
				if err := addSlice(pools, slice); err != nil {
					return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
				}
			}
		}
		if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
			for i := range slice.Spec.Devices {
				device := slice.Spec.Devices[i]

				// We need all allocated devices to compute the available counters in a resource pool.
				if allocatedDeviceIDs.Has(MakeDeviceID(slice.Spec.Driver, slice.Spec.Pool.Name, device.Name)) {
					allocatedDevices[poolID] = append(allocatedDevices[poolID], &device)
				}

				match, err := nodeMatches(node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
				if err != nil {
					return nil, fmt.Errorf("failed to perform node selection for device %s in slice %s: %w",
						device.String(), slice.Name, err)
				}
				if match {
					if hasBindingConditions(slice) {
						// If there is a Device in the ResourceSlice that contains BindingConditions,
						// the ResourceSlice should be sorted to be after the ResourceSlice without BindingConditions.
						slicesWithBindingConditions = append(slicesWithBindingConditions, slice)
						break
					}
					if err := addSlice(pools, slice); err != nil {
						return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
					}
					break
				}
			}
		} else {
			// Nothing known was set. This must be some future, unknown extension,
			// so we don't know how to handle it. We may still be able to allocated from
			// other pools, so we continue.
			//
			// TODO (eventually): let caller decide how to report this to the user. Warning
			// about it for every slice on each scheduling attempt would be too noisy, but
			// perhaps once per run would be useful?
			continue
		}

	}

	for _, slice := range slicesWithBindingConditions {
		if err := addSlice(pools, slice); err != nil {
			return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
		}
	}

	// Find incomplete pools and flatten into a single slice.
	result := make([]*Pool, 0, len(pools))
	var resultWithBindingConditions []*Pool
	for id, slicesInPool := range pools {
		// We don't allocate devices from incomplete pools.
		poolIsIncomplete := poolIsIncomplete(slicesInPool)
		if poolIsIncomplete {
			logger.V(5).Info("reesource pool is incomplete", "pool", id)
			// We don't want to validate incomplete pools, as any issues might
			// just be because we are seeing an inconsistent view.
			result = append(result, buildIncompletePool(id))
			continue
		}

		pool, err := buildPool(id, slicesInPool, allocatedDevices)
		if err != nil {
			return nil, fmt.Errorf("pool %s is invalid: %w", id.Pool, err)
		}
		// if pool has binding conditions, add the pool to the end of the result
		if poolHasBindingConditions(*pool) {
			resultWithBindingConditions = append(resultWithBindingConditions, pool)
			continue
		}
		result = append(result, pool)
	}
	if len(resultWithBindingConditions) != 0 {
		result = append(result, resultWithBindingConditions...)
	}

	return result, nil
}

func addSlice(pools map[PoolID][]*draapi.ResourceSlice, s *resourceapi.ResourceSlice) error {
	var slice draapi.ResourceSlice
	if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(s, &slice, nil); err != nil {
		return fmt.Errorf("convert ResourceSlice: %w", err)
	}

	id := PoolID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name}
	slicesInPool, ok := pools[id]
	if !ok || len(slicesInPool) == 0 {
		pools[id] = []*draapi.ResourceSlice{&slice}
		return nil
	}

	if slice.Spec.Pool.Generation < slicesInPool[0].Spec.Pool.Generation {
		// Out-dated.
		return nil
	}

	if slice.Spec.Pool.Generation > slicesInPool[0].Spec.Pool.Generation {
		// Newer, replaces all old slices.
		pools[id] = []*draapi.ResourceSlice{&slice}
		return nil
	}

	// Add to pool.
	slicesInPool = append(slicesInPool, &slice)
	pools[id] = slicesInPool
	return nil
}

func poolIsIncomplete(slicesInPool []*draapi.ResourceSlice) bool {
	return int64(len(slicesInPool)) < slicesInPool[0].Spec.Pool.ResourceSliceCount
}

func hasBindingConditions(slice *resourceapi.ResourceSlice) bool {
	for _, device := range slice.Spec.Devices {
		if device.BindingConditions != nil {
			return true
		}
	}
	return false
}

func poolHasBindingConditions(pool Pool) bool {
	for _, slice := range pool.SlicesWithDevices {
		for _, device := range slice.Spec.Devices {
			if device.BindingConditions != nil {
				return true
			}
		}
	}
	return false
}

func buildIncompletePool(id PoolID) *Pool {
	return &Pool{
		PoolID:     id,
		Incomplete: true,
	}
}

func buildPool(id PoolID, slices []*draapi.ResourceSlice, allocatedDevices map[PoolID][]*resourceapi.Device) (*Pool, error) {
	// First collect all counter sets in the pool and put them into a map
	// so we can do quick lookups.
	counterSets := make(map[draapi.UniqueString]*draapi.CounterSet)
	counterSetSlicesCount := 0
	for _, slice := range slices {
		if slice.Spec.SharedCounters != nil {
			counterSetSlicesCount++
		}
		for i := range slice.Spec.SharedCounters {
			counterSet := slice.Spec.SharedCounters[i]
			if _, found := counterSets[counterSet.Name]; found {
				return nil, fmt.Errorf("duplicate counter set name %s", counterSet.Name)
			}
			counterSets[counterSet.Name] = &counterSet
		}
	}

	slicesWithDevices := make([]*draapi.ResourceSlice, 0, len(slices)-counterSetSlicesCount)
	deviceNames := sets.New[draapi.UniqueString]()
	for _, slice := range slices {
		for i := range slice.Spec.Devices {
			device := slice.Spec.Devices[i]

			// Make sure we don't have duplicate device names
			if deviceNames.Has(device.Name) {
				return nil, fmt.Errorf("duplicate device name %s", device.Name)
			}
			deviceNames.Insert(device.Name)

			// Make sure all consumed counters for the device references counter sets that exists and
			// that they consume counters that exists within those counter sets.
			for _, deviceCounterConsumption := range device.ConsumesCounters {
				counterSet, found := counterSets[deviceCounterConsumption.CounterSet]
				if !found {
					return nil, fmt.Errorf("counter set %s not found", deviceCounterConsumption.CounterSet)
				}

				for counterName := range deviceCounterConsumption.Counters {
					if _, found := counterSet.Counters[counterName]; !found {
						return nil, fmt.Errorf("counter %s not found in counter set %s", counterName, counterSet.Name)
					}
				}
			}
		}
		slicesWithDevices = append(slicesWithDevices, slice)
	}
	return &Pool{
		PoolID:            id,
		SlicesWithDevices: slicesWithDevices,
		CounterSets:       counterSets,
		AllocatedDevices:  allocatedDevices[id],
	}, nil
}

type Pool struct {
	PoolID
	Incomplete        bool
	SlicesWithDevices []*draapi.ResourceSlice
	CounterSets       map[draapi.UniqueString]*draapi.CounterSet
	AllocatedDevices  []*resourceapi.Device
}

type PoolID struct {
	Driver, Pool draapi.UniqueString
}

func (p PoolID) String() string {
	return p.Driver.String() + "/" + p.Pool.String()
}
