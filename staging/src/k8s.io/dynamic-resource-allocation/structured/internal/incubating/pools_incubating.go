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
	"slices"

	"github.com/go-logr/logr"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

func NodeMatches(node *v1.Node, nodeNameToMatch string, allNodesMatch bool, nodeSelector *v1.NodeSelector) (bool, error) {
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
func GatherPools(ctx context.Context, slices []*resourceapi.ResourceSlice, node *v1.Node, features Features) ([]*Pool, error) {
	pools := make(map[PoolID][]*draapi.ResourceSlice)

	for _, slice := range slices {
		if !features.PartitionableDevices && (slice.Spec.PerDeviceNodeSelection != nil || len(slice.Spec.SharedCounters) > 0) {
			continue
		}

		// Always include slices with SharedCounters since they are needed to use a pool
		// regardless of their node selector.
		if len(slice.Spec.SharedCounters) > 0 {
			if err := addSlice(pools, slice); err != nil {
				return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
			}
		} else if nodeName, allNodes := ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false); nodeName != "" || allNodes || slice.Spec.NodeSelector != nil {
			match, err := NodeMatches(node, nodeName, allNodes, slice.Spec.NodeSelector)
			if err != nil {
				return nil, fmt.Errorf("failed to perform node selection for slice %s: %w", slice.Name, err)
			}
			if match {
				if err := addSlice(pools, slice); err != nil {
					return nil, fmt.Errorf("failed to add node slice %s: %w", slice.Name, err)
				}
			}
		} else if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
			for _, device := range slice.Spec.Devices {
				match, err := NodeMatches(node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
				if err != nil {
					return nil, fmt.Errorf("failed to perform node selection for device %s in slice %s: %w",
						device.String(), slice.Name, err)
				}
				if match {
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

	// Find incomplete pools and flatten into a single slice.
	//
	// When we get here, we only have slices relevant for the node.
	// There is at least one.
	// We may have skipped slices with a higher generation
	// if they are not relevant for the node, so we have to be
	// careful with the "is incomplete" check.
	result := make([]*Pool, 0, len(pools))
	for poolID, slicesForPool := range pools {
		// If we have all slices, we are done.
		isComplete := int64(len(slicesForPool)) == slicesForPool[0].Spec.Pool.ResourceSliceCount
		if isComplete {
			pool, err := buildPool(poolID, slicesForPool, features, nil)
			if err != nil {
				return nil, err
			}
			result = append(result, pool)
			continue
		}
		// If not, then we need to start looking for slices
		// which were filtered out above because their node selection made them look irrelevant
		// for the current node. This is necessary for "allocate all" mode (it rejects incomplete
		// pools).
		isObsolete, allSlicesForPool := checkSlicesInPool(slices, poolID, slicesForPool[0].Spec.Pool.Generation)
		if isObsolete {
			// A more thorough check determined that the DRA driver is in the process
			// of replacing the current generation. The newer one didn't have any slice
			// which devices for the node, or we would have noticed sooner.
			//
			// Let's ignore the old device information by ignoring the pool.
			continue
		}
		// Use the more complete number of slices to check for "incomplete pool".
		//
		// The slices that we return to the caller still don't represent the whole
		// pool, but that's okay: we *want* to limit the result to relevant devices
		// so the caller doesn't need to check node selectors unnecessarily.
		isComplete = int64(len(allSlicesForPool)) == slicesForPool[0].Spec.Pool.ResourceSliceCount
		// If a pool is incomplete, we don't allow allocation so we don't need
		// to do any validation. We need to keep track of the incomplete pools here
		// to make sure allocationMode: All doesn't succeed without considering all
		// devices.
		if !isComplete {
			result = append(result, &Pool{
				PoolID:       poolID,
				IsIncomplete: true,
			})
			continue
		}
		pool, err := buildPool(poolID, slicesForPool, features, allSlicesForPool)
		if err != nil {
			return nil, err
		}
		result = append(result, pool)
	}

	return result, nil
}

func addSlice(pools map[PoolID][]*draapi.ResourceSlice, s *resourceapi.ResourceSlice) error {
	var slice draapi.ResourceSlice
	if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(s, &slice, nil); err != nil {
		return fmt.Errorf("convert ResourceSlice: %w", err)
	}

	id := PoolID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name}
	slicesForPool := pools[id]
	if slicesForPool == nil {
		// New pool.
		pools[id] = []*draapi.ResourceSlice{&slice}
		return nil
	}

	if slice.Spec.Pool.Generation < slicesForPool[0].Spec.Pool.Generation {
		// Out-dated.
		return nil
	}

	if slice.Spec.Pool.Generation > slicesForPool[0].Spec.Pool.Generation {
		// Newer, replaces all old slices.
		pools[id] = []*draapi.ResourceSlice{&slice}
		return nil
	}

	// Add to pool.
	slicesForPool = append(slicesForPool, &slice)
	pools[id] = slicesForPool
	return nil
}

func buildPool(id PoolID, slices []*draapi.ResourceSlice, features Features, allSlicesForPool []*resourceapi.ResourceSlice) (*Pool, error) {
	var deviceSlices []*draapi.ResourceSlice
	var counterSetSlices []*draapi.ResourceSlice
	if features.PartitionableDevices {
		for _, slice := range slices {
			if len(slice.Spec.SharedCounters) > 0 {
				counterSetSlices = append(counterSetSlices, slice)
			} else {
				deviceSlices = append(deviceSlices, slice)
			}
		}
	} else {
		deviceSlices = slices
	}
	if err := validateDeviceNames(deviceSlices); err != nil {
		return &Pool{
			PoolID:        id,
			IsInvalid:     true,
			InvalidReason: err.Error(),
		}, nil
	}
	// If the partitionable devices feature is not enabled, we don't need to
	// validate counter sets and consumed counters, so we are done.
	if !features.PartitionableDevices {
		return &Pool{
			PoolID:                    id,
			DeviceSlicesTargetingNode: deviceSlices,
		}, nil
	}

	counterSets, err := getAndValidateCounterSets(counterSetSlices)
	if err != nil {
		return &Pool{
			PoolID:        id,
			IsInvalid:     true,
			InvalidReason: err.Error(),
		}, nil
	}

	if err := validateDeviceCounterConsumption(counterSets, slices); err != nil {
		return &Pool{
			PoolID:        id,
			IsInvalid:     true,
			InvalidReason: err.Error(),
		}, nil
	}
	// If we have already seen all slices (both with counter sets and devices),
	// we don't need to do any more validation.
	if allSlicesForPool == nil || len(slices) == len(allSlicesForPool) {
		return &Pool{
			PoolID:                    id,
			DeviceSlicesTargetingNode: deviceSlices,
			CounterSets:               counterSets,
		}, nil
	}

	// If we have slices that were discarded earlier because they didn't target the current node
	// we need to check them now. They might include devices that consume counters in the pool and
	// the allocator needs to know about them to correctly determine available counters.
	//
	// We only want to convert the slices we haven't already converted, so make it easy to
	// look up the names of converted slices.
	slicesTargetingNodeNames := sets.New[string]()
	for _, slice := range slices {
		slicesTargetingNodeNames.Insert(slice.Name)
	}
	var slicesNotTargetingNode []*draapi.ResourceSlice
	for _, slice := range allSlicesForPool {
		if slicesTargetingNodeNames.Has(slice.Name) {
			continue
		}
		var convertedSlice draapi.ResourceSlice
		if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(slice, &convertedSlice, nil); err != nil {
			return nil, fmt.Errorf("convert ResourceSlice: %w", err)
		}
		slicesNotTargetingNode = append(slicesNotTargetingNode, &convertedSlice)
	}
	// We need to make sure the devices here are correctly consuming counters and counter
	// sets. Otherwise the allocator might make incorrect decisions.
	// We don't validate the device names here. It might be that we should do that, but
	// this is consistent with existing behavior where we don't validate slices that
	// we don't allocate from.
	if err := validateDeviceCounterConsumption(counterSets, slicesNotTargetingNode); err != nil {
		return &Pool{
			PoolID:        id,
			IsInvalid:     true,
			InvalidReason: err.Error(),
		}, nil
	}
	return &Pool{
		PoolID:                       id,
		DeviceSlicesTargetingNode:    deviceSlices,
		DeviceSlicesNotTargetingNode: slicesNotTargetingNode,
		CounterSets:                  counterSets,
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

func validateDeviceNames(resourceSlices []*draapi.ResourceSlice) error {
	deviceNames := sets.New[draapi.UniqueString]()
	for _, slice := range resourceSlices {
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

func validateDeviceCounterConsumption(counterSets map[draapi.UniqueString]*draapi.CounterSet, resourceSlices []*draapi.ResourceSlice) error {
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
	return nil
}

// checkSlicesInPool is an expensive check of all slices in the pool.
// The generation is what the caller wants to move ahead with.
//
// It returns:
// - current generation is obsolete -> no further checking
// - all slices with the generation in the pool
//
// Future TODO: detect inconsistent ResourceSliceCount, also in poolIsInvalid.
func checkSlicesInPool(slices []*resourceapi.ResourceSlice, poolID PoolID, generation int64) (bool, []*resourceapi.ResourceSlice) {
	// A cached index by pool ID would make this more efficient.
	// It may be needed long-term to support features which always have to consider all slices.
	var allSlicesForPool []*resourceapi.ResourceSlice
	for i := range slices {
		slice := slices[i]
		if slice.Spec.Driver != poolID.Driver.String() ||
			slice.Spec.Pool.Name != poolID.Pool.String() {
			// Different pool.
			continue
		}
		switch {
		case slice.Spec.Pool.Generation == generation:
			allSlicesForPool = append(allSlicesForPool, slice)
		case slice.Spec.Pool.Generation > generation:
			// The caller must have missed some other slice in the pool.
			// Abort!
			return true, nil
		default:
			// Older generation, ignore.
		}
	}
	return false, allSlicesForPool
}

type Pool struct {
	PoolID
	IsIncomplete                 bool
	IsInvalid                    bool
	InvalidReason                string
	DeviceSlicesTargetingNode    []*draapi.ResourceSlice
	DeviceSlicesNotTargetingNode []*draapi.ResourceSlice
	CounterSets                  map[draapi.UniqueString]*draapi.CounterSet
}

type PoolID struct {
	Driver, Pool draapi.UniqueString
}

func (p PoolID) String() string {
	return p.Driver.String() + "/" + p.Pool.String()
}

// At V(6), log only a limited number of devices to avoid blowing up logs. For
// many E2E tests, 10 devices is enough for all devices without having to
// truncate, at least when running the tests sequentially.
const maxDevicesLevel6 = 10

// logPools returns a handle for the value in a structured log call which
// includes varying amounts of information about the pools, depending on
// the verbosity of the logger.
func logPools(logger klog.Logger, pools []*Pool) any {
	// We need to check verbosity here because our caller's source code
	// location may be relevant (-vmodule !).
	helper, logger := logger.WithCallStackHelper()
	helper()

	// We always produce the same output at V <= 5. 6 adds a summary and
	// 7 is a complete dump.
	verbosity := 5
	for i := 7; i > verbosity; i-- {
		if loggerV := logger.V(i); loggerV.Enabled() {
			verbosity = i
			break
		}
	}
	return &poolsLogger{verbosity, pools}
}

type poolsLogger struct {
	verbosity int
	pools     []*Pool
}

var _ logr.Marshaler = &poolsLogger{}

func (p *poolsLogger) MarshalLog() any {
	info := map[string]any{"count": len(p.pools)}
	if p.verbosity == 6 {
		meta := make([]map[string]any, len(p.pools))
		for i, pool := range p.pools {
			meta[i] = map[string]any{
				"id":            pool.PoolID.String(),
				"isIncomplete":  pool.IsIncomplete,
				"isInvalid":     pool.IsInvalid,
				"InvalidReason": pool.InvalidReason,
			}
		}
		info["meta"] = meta
		info["devices"] = p.listDevices(maxDevicesLevel6)
	}
	if p.verbosity >= 7 {
		info["devices"] = p.listDevices(-1)
		info["content"] = p.pools
	}
	return info
}

func (p *poolsLogger) listDevices(maxDevices int) []string {
	var devices []string
	for _, pool := range p.pools {
		devices = p.addDevicesInSlices(devices, pool.PoolID, pool.DeviceSlicesTargetingNode, maxDevices)
		devices = p.addDevicesInSlices(devices, pool.PoolID, pool.DeviceSlicesNotTargetingNode, maxDevices)
	}
	return devices
}

func (p *poolsLogger) addDevicesInSlices(devices []string, poolID PoolID, slices []*draapi.ResourceSlice, maxDevices int) []string {
	for _, slice := range slices {
		for _, device := range slice.Spec.Devices {
			if maxDevices != -1 && len(devices) >= maxDevices {
				devices = append(devices, "...")
				return devices
			}
			devices = append(devices, DeviceID{Driver: poolID.Driver, Pool: poolID.Pool, Device: device.Name}.String())
		}
	}
	return devices
}

// logPools returns a handle for the value in a structured log call which
// includes varying amounts of information about the allocated devices, depending on
// the verbosity of the logger.
func logAllocatedDevices(logger klog.Logger, allocatedDevices sets.Set[DeviceID]) any {
	// We need to check verbosity here because our caller's source code
	// location may be relevant (-vmodule !).
	helper, logger := logger.WithCallStackHelper()
	helper()

	// We always produce the same output at V <= 5. 6 adds all IDs.
	verbosity := 5
	for i := 7; i > verbosity; i-- {
		if loggerV := logger.V(i); loggerV.Enabled() {
			verbosity = i
			break
		}
	}

	return &allocatedDevicesLogger{verbosity, allocatedDevices}
}

type allocatedDevicesLogger struct {
	verbosity int
	devices   sets.Set[DeviceID]
}

var _ logr.Marshaler = &allocatedDevicesLogger{}

func (a *allocatedDevicesLogger) MarshalLog() any {
	info := map[string]any{"count": len(a.devices)}
	if a.verbosity >= 6 {
		ids := make([]string, 0, len(a.devices))
		for id := range a.devices {
			if a.verbosity == 6 && len(ids) >= maxDevicesLevel6 {
				ids = append(ids, "...")
				break
			}
			ids = append(ids, id.String())
		}
		slices.Sort(ids)
		info["devices"] = ids

	}
	return info
}
