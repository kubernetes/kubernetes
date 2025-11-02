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
func GatherPools(ctx context.Context, slices []*resourceapi.ResourceSlice, node *v1.Node, features Features) ([]*Pool, error) {
	pools := make(map[PoolID]*Pool)
	for _, slice := range slices {
		if !features.PartitionableDevices && slice.Spec.PerDeviceNodeSelection != nil {
			continue
		}

		if nodeName, allNodes := ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false); nodeName != "" || allNodes || slice.Spec.NodeSelector != nil {
			match, err := nodeMatches(node, nodeName, allNodes, slice.Spec.NodeSelector)
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
				match, err := nodeMatches(node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
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
	for poolID, pool := range pools {
		isIncomplete := int64(len(pool.Slices)) != pool.Slices[0].Spec.Pool.ResourceSliceCount
		// If we have all slices, we are done. If not, then we need to start looking for slices
		// which were filtered out above because their node selection made them look irrelevant
		// for the current node. This is necessary for "allocate all" mode (it rejects incomplete
		// pools).
		if isIncomplete {
			// Find all the slices that belongs to the current pool. Doing this for each pool
			// separately could be a big inefficient if there are multiple incomplete pools. But
			// that is probably a rare situation.
			slicesForPool := findSlicesForPool(slices, poolID)
			// This allows to check ALL slices in the pool to make sure whether the pool is
			// complete or not.
			isIncomplete = poolIsIncomplete(slicesForPool)
			// If the pool is truly incomplete after this check, then we need to find out if
			// the slices we have in the pool object really is on the latest generation. If one of
			// the slices that was filtered out earlier are on a newer generation, it is not.
			if isIncomplete {
				latestGeneration := findLatestGenerationInPool(slices)
				// All slices in the pool are on the same generation, so if the first one is on an
				// older generation, then they are all stale and must be removed from the pool.
				if len(pool.Slices) > 0 && pool.Slices[0].Spec.Pool.Generation != latestGeneration {
					pool.Slices = nil
				}
			}
		}
		pool.IsIncomplete = isIncomplete
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

// findSlicesForPool returns all ResourceSlices that belong to the provided pool.
func findSlicesForPool(slices []*resourceapi.ResourceSlice, poolID PoolID) []*resourceapi.ResourceSlice {
	driver := poolID.Driver.String()
	pool := poolID.Pool.String()

	var slicesForPool []*resourceapi.ResourceSlice
	for i := range slices {
		slice := slices[i]
		if slice.Spec.Driver == driver && slice.Spec.Pool.Name == pool {
			slicesForPool = append(slicesForPool, slice)
		}
	}
	return slicesForPool
}

// poolIsIncomplete checks if the provided ResourceSlices belong to a
// complete pool. All provided ResourceSlices must belong to the same pool.
//
// For a pool to be complete, the following must be true:
//   - All slices have the same generation
//   - All slices agree on the number of ResourceSlices in the pool
//   - The number of actual slices found must equal the expected number of
//     slices in the pool.
func poolIsIncomplete(slices []*resourceapi.ResourceSlice) bool {
	if len(slices) == 0 {
		return false
	}
	// Just take the resourceSliceCount from the first slice as the
	// point of comparison. If this doesn't equal the count in all
	// other ResourceSlices and the total number of slices in the pool,
	// then the pool is not complete.
	sliceCount := slices[0].Spec.Pool.ResourceSliceCount
	if int64(len(slices)) != sliceCount {
		return true
	}
	// Just take the generation of the first slice as the point of
	// comparison. If this value is not the latest version, then
	// the pool can't be complete.
	gen := slices[0].Spec.Pool.Generation
	for _, slice := range slices {
		if slice.Spec.Pool.Generation != gen {
			return true
		}
		if slice.Spec.Pool.ResourceSliceCount != sliceCount {
			return true
		}
	}
	return false
}

func findLatestGenerationInPool(slices []*resourceapi.ResourceSlice) int64 {
	var latestGeneration int64
	for _, slice := range slices {
		if slice.Spec.Pool.Generation > latestGeneration {
			latestGeneration = slice.Spec.Pool.Generation
		}
	}
	return latestGeneration
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
