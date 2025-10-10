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

package incubating

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	draapi "k8s.io/dynamic-resource-allocation/api"
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
func GatherPools(ctx context.Context, slices []*resourceapi.ResourceSlice, node *v1.Node, features Features) ([]*Pool, error) {

	// Collect all slices by pool so we can easily do checks across each pool.
	slicesByPool := make(map[poolIdentifier][]*resourceapi.ResourceSlice)
	for _, slice := range slices {
		poolID := poolIdentifier{
			driver: slice.Spec.Driver,
			pool:   slice.Spec.Pool.Name,
		}
		slicesByPool[poolID] = append(slicesByPool[poolID], slice)
	}

	incompletePools := sets.New[poolIdentifier]()
	// For each pool, check if it has any devices that can be allocated on this node. If it
	// doesn't we don't need to do any work on that pool. We don't do any other checks on each
	// ResourceSlice, since that work might be wasted if there are no devices that targets
	// the node.
	// Also check if pools are complete. We need to make that check while we have all
	// ResourceSlices that belong to a pool.
	for poolID, slices := range slicesByPool {
		complete := true
		sliceCount := len(slices)
		generation := slices[0].Spec.Pool.Generation
		targetsNode := false
		for _, slice := range slices {
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
			// If we find any slice that has at least one device that can be allocted to this
			// node, we need to include the pool. This is because counters are shared across the pool,
			// so even devices that can't be allocated to the current node might consume counters that
			// affects the devices available on the node.
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
						// If we find a device that matches the node, we don't need
						// to look at the rest of them.
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
				complete = false
			}
		}
		// No need to keep resource pools that doesn't have any devices that can
		// be allocated to the current node.
		if !targetsNode {
			delete(slicesByPool, poolID)
			continue
		}
		// We still need to keep incomplete pools, since we need to make sure
		// all devices available on a node is considered for allocationMode All.
		if !complete {
			incompletePools.Insert(poolID)
		}
	}

	result := make([]*Pool, 0, len(slicesByPool))
	for poolID, slices := range slicesByPool {
		var slicesForPool []*draapi.ResourceSlice
		for _, slice := range slices {
			s, err := convertSlice(slice)
			if err != nil {
				return nil, fmt.Errorf("failed to convert ResourceSlice: %w", err)
			}
			slicesForPool = append(slicesForPool, s)
		}

		id := PoolID{Driver: slicesForPool[0].Spec.Driver, Pool: slicesForPool[0].Spec.Pool.Name}
		pool, err := buildPool(id, slicesForPool, node, incompletePools.Has(poolID), features)
		if err != nil {
			return nil, fmt.Errorf("failed to build pool %s: %w", id.String(), err)
		}
		result = append(result, pool)
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

func buildPool(id PoolID, slices []*draapi.ResourceSlice, node *v1.Node, isIncomplete bool, features Features) (*Pool, error) {
	// These are resource slices that targets the node either on the resource slice level, or
	// sets PerDeviceNodeSelection, which means there might be devices targeting the node.
	var deviceSlicesForNode []*draapi.ResourceSlice
	// Resource slices that doesn't target the node can still contain devices that consume
	// counters, meaning we need to keep track of them. But we keep them separate since we
	// don't want to try allocating the devices defined in them.
	var deviceSlicesRemainingInPool []*draapi.ResourceSlice
	var counterSetSlices []*draapi.ResourceSlice
	// Find all ResourceSlices with devices since we need to include those in the
	// Pool even if we encounter validation errors.
	for _, slice := range slices {
		// We drop slices that doesn't contain either devices or counter sets.
		if len(slice.Spec.SharedCounters) > 0 {
			counterSetSlices = append(counterSetSlices, slice)
		}
		if len(slice.Spec.Devices) > 0 {
			// We have already checked for completeness, so we can throw away slices
			// that we know we won't need. If the partitionable devices featue is not enabled,
			// we know we will not be allocating devices from slices with perDeviceNodeSelection.
			// We also know that not having a complete set of devices will not impact computing
			// available counters, since that is also part of the partitionable devices feature.
			if !features.PartitionableDevices && slice.Spec.PerDeviceNodeSelection != nil {
				continue
			}

			var match bool
			if nodeName := ptr.Deref(slice.Spec.NodeName, ""); nodeName != "" || slice.Spec.AllNodes || slice.Spec.NodeSelector != nil {
				m, err := nodeMatches(node, nodeName, slice.Spec.AllNodes, slice.Spec.NodeSelector)
				if err != nil {
					return nil, fmt.Errorf("failed to perform node selection for slice %s: %w", slice.Name, err)
				}
				match = m
			} else if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
				for _, device := range slice.Spec.Devices {
					m, err := nodeMatches(node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
					if err != nil {
						return nil, fmt.Errorf("failed to perform node selection for device %s in slice %s: %w",
							device.Name, slice.Name, err)
					}
					if m {
						match = true
						// If we found a device that matches the node, no need to look at the
						// rest of the devices.
						break
					}
				}
			} else {
				continue
			}
			if match {
				deviceSlicesForNode = append(deviceSlicesForNode, slice)
			} else {
				deviceSlicesRemainingInPool = append(deviceSlicesRemainingInPool, slice)
			}
		}
	}

	// Validate the device names first, since the list of devices is important
	// even for incomplete or invalid ResourceSlices, since trying to allocate
	// all devices should not work if any of those devices are part of an invalid
	// or incomplete pool.
	if err := validateDeviceNames(deviceSlicesForNode, deviceSlicesRemainingInPool); err != nil {
		return &Pool{
			PoolID:                 id,
			DeviceSlicesForNode:    deviceSlicesForNode,
			DeviceSlicesNotForNode: deviceSlicesRemainingInPool,
			IsIncomplete:           isIncomplete,
			IsInvalid:              true,
			InvalidReason:          err.Error(),
		}, nil
	}

	counterSets, err := getAndValidateCounterSets(counterSetSlices)
	if err != nil {
		return &Pool{
			PoolID:                 id,
			DeviceSlicesForNode:    deviceSlicesForNode,
			DeviceSlicesNotForNode: deviceSlicesRemainingInPool,
			IsIncomplete:           isIncomplete,
			IsInvalid:              true,
			InvalidReason:          err.Error(),
		}, nil
	}

	if err := validateDeviceCounterConsumption(counterSets, deviceSlicesForNode, deviceSlicesRemainingInPool); err != nil {
		return &Pool{
			PoolID:                 id,
			DeviceSlicesForNode:    deviceSlicesForNode,
			DeviceSlicesNotForNode: deviceSlicesRemainingInPool,
			CounterSets:            counterSets,
			IsIncomplete:           isIncomplete,
			IsInvalid:              true,
			InvalidReason:          err.Error(),
		}, nil
	}
	return &Pool{
		PoolID:                 id,
		DeviceSlicesForNode:    deviceSlicesForNode,
		DeviceSlicesNotForNode: deviceSlicesRemainingInPool,
		CounterSets:            counterSets,
		IsIncomplete:           isIncomplete,
	}, nil
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

func validateDeviceNames(slicesOfResourceSlices ...[]*draapi.ResourceSlice) error {
	deviceNames := sets.New[draapi.UniqueString]()
	for _, resourceSlices := range slicesOfResourceSlices {
		for _, slice := range resourceSlices {
			for _, device := range slice.Spec.Devices {
				// Make sure we don't have duplicate device names
				if deviceNames.Has(device.Name) {
					return fmt.Errorf("duplicate device name %s", device.Name)
				}
				deviceNames.Insert(device.Name)
			}
		}
	}
	return nil
}

func validateDeviceCounterConsumption(counterSets map[draapi.UniqueString]*draapi.CounterSet, slicesOfResourceSlices ...[]*draapi.ResourceSlice) error {
	for _, resourceSlices := range slicesOfResourceSlices {
		for _, slice := range resourceSlices {
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
	}
	return nil
}

type Pool struct {
	PoolID
	DeviceSlicesForNode    []*draapi.ResourceSlice
	DeviceSlicesNotForNode []*draapi.ResourceSlice
	CounterSets            map[draapi.UniqueString]*draapi.CounterSet
	IsIncomplete           bool
	IsInvalid              bool
	InvalidReason          string
}

type PoolID struct {
	Driver, Pool draapi.UniqueString
}

func (p PoolID) String() string {
	return p.Driver.String() + "/" + p.Pool.String()
}
