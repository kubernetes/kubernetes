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

package structured

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

// GatherPools collects information about all resource pools which provide
// devices that are accessible from the given node.
//
// Out-dated slices are silently ignored. Pools may be incomplete (not all
// required slices available) or invalid (for example, device names not unique).
// Both is recorded in the result.
func GatherPools(ctx context.Context, slices []*resourceapi.ResourceSlice, node *v1.Node, partitionableDevicesEnabled bool) ([]*Pool, error) {
	pools := make(map[PoolID]*Pool)
	nodeName := ""
	if node != nil {
		nodeName = node.Name
	}

	for _, slice := range slices {
		switch {
		case slice.Spec.NodeName != "":
			if slice.Spec.NodeName == nodeName {
				if err := addSlice(pools, slice, node, partitionableDevicesEnabled); err != nil {
					return nil, fmt.Errorf("add node slice %s: %w", slice.Name, err)
				}
			}
		case slice.Spec.AllNodes:
			if err := addSlice(pools, slice, node, partitionableDevicesEnabled); err != nil {
				return nil, fmt.Errorf("add cluster slice %s: %w", slice.Name, err)
			}
		case slice.Spec.NodeSelector != nil:
			// TODO: move conversion into api.
			selector, err := nodeaffinity.NewNodeSelector(slice.Spec.NodeSelector)
			if err != nil {
				return nil, fmt.Errorf("node selector in resource slice %s: %w", slice.Name, err)
			}
			if selector.Match(node) {
				if err := addSlice(pools, slice, node, partitionableDevicesEnabled); err != nil {
					return nil, fmt.Errorf("add matching slice %s: %w", slice.Name, err)
				}
			}
		case slice.Spec.PerDeviceNodeSelection:
			// We add the slice here regardless of whether the partitionable devices feature is
			// enabled. If we don't, the full slice will be considered incomplete. So we filter
			// out devices that have fields from the partitionable devices feature set later.
			if err := addSlice(pools, slice, node, partitionableDevicesEnabled); err != nil {
				return nil, fmt.Errorf("add cluster slice %s: %w", slice.Name, err)
			}
		default:
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
		pool.IsIncomplete = int64(len(pool.Slices)) != pool.Slices[0].Spec.Pool.ResourceSliceCount
		pool.IsInvalid, pool.InvalidReason = poolIsInvalid(pool)
		result = append(result, pool)
	}

	return result, nil
}

func addSlice(pools map[PoolID]*Pool, s *resourceapi.ResourceSlice, node *v1.Node, partitionableDevicesEnabled bool) error {
	var slice draapi.ResourceSlice
	sliceScope := draapi.SliceScope{
		SliceContext: draapi.SliceContext{
			Slice:                       s,
			Node:                        node,
			PartitionableDevicesEnabled: partitionableDevicesEnabled,
		},
	}
	if err := draapi.Convert_v1beta1_ResourceSlice_To_api_ResourceSlice(s, &slice, sliceScope); err != nil {
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

type DeviceID struct {
	Driver, Pool, Device draapi.UniqueString
}

func (d DeviceID) String() string {
	return d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String()
}

func MakeDeviceID(driver, pool, device string) DeviceID {
	return DeviceID{
		Driver: draapi.MakeUniqueString(driver),
		Pool:   draapi.MakeUniqueString(pool),
		Device: draapi.MakeUniqueString(device),
	}
}
