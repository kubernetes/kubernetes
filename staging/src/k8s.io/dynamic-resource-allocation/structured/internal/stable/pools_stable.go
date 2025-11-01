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

package stable

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
	slicesByPool := make(map[poolIdentifier][]*resourceapi.ResourceSlice)
	for _, slice := range slices {
		poolID := poolIdentifier{
			driver: slice.Spec.Driver,
			pool:   slice.Spec.Pool.Name,
		}
		slicesByPool[poolID] = append(slicesByPool[poolID], slice)
	}

	// We need to check whether a pool is complete while we have all
	// the slices. Once we discard slices that don't target the node, we
	// no longer have the information needed to find out.
	incompletePools := sets.New[poolIdentifier]()
	for poolID, slices := range slicesByPool {
		complete := true
		sliceCount := len(slices)
		generation := slices[0].Spec.Pool.Generation
		for _, slice := range slices {
			// If the number of slices in the pool specified in any of the slices
			// doesn't match what we found, the pool is most likely being updated
			// by the controller.
			if slice.Spec.Pool.ResourceSliceCount != int64(sliceCount) {
				complete = false
			}
			// If the generation of the pool isn't the same across all slices,
			// the pool is most likely being updated by the controller. We can't
			// allocate devices from it.
			if slice.Spec.Pool.Generation != generation {
				complete = false
			}
		}
		// We still need to keep incomplete pools, since we need to make sure
		// all devices available on a node is considered for allocationMode All.
		if !complete {
			incompletePools.Insert(poolID)
		}
	}

	pools := make(map[PoolID]*Pool)
	for _, slice := range slices {
		if !features.PartitionableDevices && slice.Spec.PerDeviceNodeSelection != nil {
			continue
		}

		if nodeName, allNodes := ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false); nodeName != "" || allNodes || slice.Spec.NodeSelector != nil {
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
	result := make([]*Pool, 0, len(pools))
	for _, pool := range pools {
		pool.IsIncomplete = incompletePools.Has(poolIdentifier{driver: pool.Driver.String(), pool: pool.Pool.String()})
		pool.IsInvalid, pool.InvalidReason = poolIsInvalid(pool)
		result = append(result, pool)
	}

	return result, nil
}

func addSlice(pools map[PoolID]*Pool, s *resourceapi.ResourceSlice) error {
	var slice draapi.ResourceSlice
	if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(s, &slice, nil); err != nil {
		return fmt.Errorf("convert ResourceSlice: %w", err)
	}

	id := PoolID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name}
	pool := pools[id]
	if pool == nil {
		// New pool.
		pool = &Pool{
			PoolID: id,
			Slices: []*draapi.ResourceSlice{&slice},
		}
		pools[id] = pool
		return nil
	}

	if slice.Spec.Pool.Generation < pool.Slices[0].Spec.Pool.Generation {
		// Out-dated.
		return nil
	}

	if slice.Spec.Pool.Generation > pool.Slices[0].Spec.Pool.Generation {
		// Newer, replaces all old slices.
		pool.Slices = nil
	}

	// Add to pool.
	pool.Slices = append(pool.Slices, &slice)
	return nil
}

func poolIsInvalid(pool *Pool) (bool, string) {
	devices := sets.New[draapi.UniqueString]()
	for _, slice := range pool.Slices {
		for _, device := range slice.Spec.Devices {
			if devices.Has(device.Name) {
				return true, fmt.Sprintf("duplicate device name %s", device.Name)
			}
			devices.Insert(device.Name)
		}
	}
	return false, ""
}

type Pool struct {
	PoolID
	IsIncomplete  bool
	IsInvalid     bool
	InvalidReason string
	Slices        []*draapi.ResourceSlice
}

type PoolID struct {
	Driver, Pool draapi.UniqueString
}

func (p PoolID) String() string {
	return p.Driver.String() + "/" + p.Pool.String()
}
