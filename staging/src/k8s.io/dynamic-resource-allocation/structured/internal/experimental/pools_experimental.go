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

type poolIdentifier struct {
	driver, pool string
}

// GatherPools collects information about all resource pools which provide
// devices that are accessible from the given node.
//
// Out-dated slices are silently ignored. Pools may be incomplete (not all
// required slices available) or invalid (for example, device names not unique).
// Both is recorded in the result.
func GatherPools(ctx context.Context, logger klog.Logger, slices []*resourceapi.ResourceSlice, node *v1.Node, allocatedDeviceIDs sets.Set[DeviceID], features Features) ([]*Pool, error) {

	// Collect all slices by pool so we can easily do checks across each pool.
	slicesByPool := make(map[poolIdentifier][]*resourceapi.ResourceSlice)
	for _, slice := range slices {
		poolId := poolIdentifier{
			driver: slice.Spec.Driver,
			pool:   slice.Spec.Pool.Name,
		}
		slicesByPool[poolId] = append(slicesByPool[poolId], slice)
	}

	incompletePools := sets.New[poolIdentifier]()
	// For each pool, check if it is valid and has devices that can be allocated
	// to this node. The goal is to quickly get rid of pools that aren't useful
	// for this node.
	for poolId, slices := range slicesByPool {
		complete := true
		sliceCount := len(slices)
		generation := slices[0].Spec.Pool.Generation
		targetsNode := false
		for _, slice := range slices {
			// If we find any slice that has at least one device that can be allocted to this
			// node, we need to include the pool.
			if nodeName, allNodes := ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false); nodeName != "" || allNodes || slice.Spec.NodeSelector != nil {
				match, err := nodeMatches(node, nodeName, allNodes, slice.Spec.NodeSelector)
				if err != nil {
					return nil, fmt.Errorf("failed to perform node selection for slice %s: %w", slice.Name, err)
				}
				if match {
					targetsNode = true
				}
			} else if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
				for _, device := range slice.Spec.Devices {
					match, err := nodeMatches(node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
					if err != nil {
						return nil, fmt.Errorf("failed to perform node selection for device %s in slice %s: %w",
							device.String(), slice.Name, err)
					}
					if match {
						targetsNode = true
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
				complete = false
			}
			// If the number of slices in the pool specified in any of the slices
			// doesn't match what we found, the pool is most likely being updated
			// by the controller. We can't allocate devices from it.
			if slice.Spec.Pool.ResourceSliceCount != int64(sliceCount) {
				complete = false
			}
			// If the generation of the pool isn't the same across all slices,
			// the pool is most likely being updated by the controller. We can't
			// allocate devices from it.
			if slice.Spec.Pool.Generation != generation {
				complete = false
			}
			// If we know enough, just break out of the loop
			if !targetsNode || !complete {
				break
			}
		}

		if !targetsNode {
			delete(slicesByPool, poolId)
			continue
		}
		if !complete {
			incompletePools.Insert(poolId)
		}
	}

	result := make([]*Pool, 0, len(slicesByPool))
	var resultWithBindingConditions []*Pool
	for poolId, slices := range slicesByPool {
		var slicesForPool []*draapi.ResourceSlice
		var slicesWithBindingConditions []*draapi.ResourceSlice
		// Keep track of whether we find any ResourceSlices with binding conditions so
		// we don't have to check again later.
		var poolHasBindingConditions bool
		for _, slice := range slices {
			s, err := convertSlice(slice)
			if err != nil {
				return nil, fmt.Errorf("failed to convert ResourceSlice: %w", err)
			}
			if hasBindingConditions(slice) {
				// If there is a Device in the ResourceSlice that contains BindingConditions,
				// the ResourceSlice should be sorted to be after the ResourceSlice without BindingConditions
				// because then the allocation is going to prefer the simpler devices without
				// binding conditions.chrome
				poolHasBindingConditions = true
				slicesWithBindingConditions = append(slicesWithBindingConditions, s)
			} else {
				slicesForPool = append(slicesForPool, s)
			}
		}
		if len(slicesWithBindingConditions) != 0 {
			slicesForPool = append(slicesForPool, slicesWithBindingConditions...)
		}

		id := PoolID{Driver: slicesForPool[0].Spec.Driver, Pool: slicesForPool[0].Spec.Pool.Name}
		pool := buildPool(id, slicesForPool, incompletePools.Has(poolId), features)
		// if pool has binding conditions, add the pool to the end of the result
		if poolHasBindingConditions {
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

func convertSlice(s *resourceapi.ResourceSlice) (*draapi.ResourceSlice, error) {
	var slice draapi.ResourceSlice
	if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(s, &slice, nil); err != nil {
		return nil, fmt.Errorf("convert ResourceSlice: %w", err)
	}
	return &slice, nil
}

func hasBindingConditions(slice *resourceapi.ResourceSlice) bool {
	for _, device := range slice.Spec.Devices {
		if device.BindingConditions != nil {
			return true
		}
	}
	return false
}

func buildPool(id PoolID, slices []*draapi.ResourceSlice, isIncomplete bool, features Features) *Pool {
	var deviceSlices []*draapi.ResourceSlice
	var counterSetSlices []*draapi.ResourceSlice
	// Find all ResourceSlices with devices since we need to include those in the
	// Pool even if we encounter validation errors.
	for _, slice := range slices {
		// We drop slices that doesn't contain either devices or counter sets.
		if len(slice.Spec.Devices) > 0 {
			if !features.PartitionableDevices && slice.Spec.PerDeviceNodeSelection != nil {
				continue
			}
			deviceSlices = append(deviceSlices, slice)
		}
		if len(slice.Spec.SharedCounters) > 0 {
			counterSetSlices = append(counterSetSlices, slice)
		}
	}

	// Validate the device names first, since the list of devices is important
	// even for incomplete or invalid ResourceSlices, since trying to allocate
	// all devices should not work if any of those devices are part of an invalid
	// or incomplete pool.
	if err := validateDeviceNames(deviceSlices); err != nil {
		return &Pool{
			PoolID:            id,
			SlicesWithDevices: deviceSlices,
			IsIncomplete:      isIncomplete,
			IsInvalid:         true,
			InvalidReason:     err.Error(),
		}
	}

	counterSets, err := getAndValidateCounterSets(counterSetSlices)
	if err != nil {
		return &Pool{
			PoolID:            id,
			SlicesWithDevices: deviceSlices,
			IsIncomplete:      isIncomplete,
			IsInvalid:         true,
			InvalidReason:     err.Error(),
		}
	}

	if err := validateDeviceCounterConsumption(deviceSlices, counterSets); err != nil {
		return &Pool{
			PoolID:            id,
			SlicesWithDevices: deviceSlices,
			CounterSets:       counterSets,
			IsIncomplete:      isIncomplete,
			IsInvalid:         true,
			InvalidReason:     err.Error(),
		}
	}
	return &Pool{
		PoolID:            id,
		SlicesWithDevices: deviceSlices,
		CounterSets:       counterSets,
		IsIncomplete:      isIncomplete,
	}
}

func getAndValidateCounterSets(slices []*draapi.ResourceSlice) (map[draapi.UniqueString]*draapi.CounterSet, error) {
	counterSets := make(map[draapi.UniqueString]*draapi.CounterSet)
	// We only capture the first error we encounter.
	for _, slice := range slices {
		for i := range slice.Spec.SharedCounters {
			counterSet := slice.Spec.SharedCounters[i]
			if _, found := counterSets[counterSet.Name]; found {
				return nil, fmt.Errorf("duplicate counter set name %s", counterSet.Name)
			}
			counterSets[counterSet.Name] = &counterSet
		}
	}
	return counterSets, nil
}

func validateDeviceNames(slices []*draapi.ResourceSlice) error {
	deviceNames := sets.New[draapi.UniqueString]()
	for _, slice := range slices {
		for _, device := range slice.Spec.Devices {
			// Make sure we don't have duplicate device names
			if deviceNames.Has(device.Name) {
				return fmt.Errorf("duplicate device name %s", device.Name)
			}
			deviceNames.Insert(device.Name)
		}
	}
	return nil
}

func validateDeviceCounterConsumption(slices []*draapi.ResourceSlice, counterSets map[draapi.UniqueString]*draapi.CounterSet) error {
	for _, slice := range slices {
		for _, device := range slice.Spec.Devices {

			// Make sure all consumed counters for the device references counter sets that exists and
			// that they consume counters that exists within those counter sets.
			for _, deviceCounterConsumption := range device.ConsumesCounters {
				counterSet, found := counterSets[deviceCounterConsumption.CounterSet]
				if !found {
					return fmt.Errorf("counter set %s not found", deviceCounterConsumption.CounterSet)
				}

				for counterName := range deviceCounterConsumption.Counters {
					if _, found := counterSet.Counters[counterName]; !found {
						return fmt.Errorf("counter %s not found in counter set %s", counterName, counterSet.Name)
					}
				}
			}
		}
	}
	return nil
}

type Pool struct {
	PoolID
	SlicesWithDevices []*draapi.ResourceSlice
	CounterSets       map[draapi.UniqueString]*draapi.CounterSet
	IsIncomplete      bool
	IsInvalid         bool
	InvalidReason     string
}

type PoolID struct {
	Driver, Pool draapi.UniqueString
}

func (p PoolID) String() string {
	return p.Driver.String() + "/" + p.Pool.String()
}
