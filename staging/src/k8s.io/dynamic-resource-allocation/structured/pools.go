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
	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/labels"
	resourcelisters "k8s.io/client-go/listers/resource/v1alpha3"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
)

// GatherPools collects information about all resource pools which provide
// devices that are accessible from the given node.
//
// Out-dated slices are silently ignored. Pools may be incomplete, which is
// recorded in the result.
func GatherPools(ctx context.Context, sliceLister resourcelisters.ResourceSliceLister, node *v1.Node) ([]*Pool, error) {
	pools := make(map[PoolID]*Pool)

	// TODO (future): use a custom lister interface and implement it with
	// and indexer on the node name field. Then here we can ask for only
	// slices with the right node name and those with no node name.
	slices, err := sliceLister.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("list resource slices: %w", err)
	}
	for _, slice := range slices {
		switch {
		case slice.Spec.NodeName != "":
			if slice.Spec.NodeName == node.Name {
				addSlice(pools, slice)
			}
		case slice.Spec.AllNodes:
			addSlice(pools, slice)
		case slice.Spec.NodeSelector != nil:
			selector, err := nodeaffinity.NewNodeSelector(slice.Spec.NodeSelector)
			if err != nil {
				return nil, fmt.Errorf("node selector in resource slice %s: %w", slice.Name, err)
			}
			if selector.Match(node) {
				addSlice(pools, slice)
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
		result = append(result, pool)
	}

	return result, nil
}

func addSlice(pools map[PoolID]*Pool, slice *resourceapi.ResourceSlice) {
	id := PoolID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name}
	pool := pools[id]
	if pool == nil {
		// New pool.
		pool = &Pool{
			PoolID: id,
			Slices: []*resourceapi.ResourceSlice{slice},
		}
		pools[id] = pool
		return
	}

	if slice.Spec.Pool.Generation < pool.Slices[0].Spec.Pool.Generation {
		// Out-dated.
		return
	}

	if slice.Spec.Pool.Generation > pool.Slices[0].Spec.Pool.Generation {
		// Newer, replaces all old slices.
		pool.Slices = []*resourceapi.ResourceSlice{slice}
	}

	// Add to pool.
	pool.Slices = append(pool.Slices, slice)
}

type Pool struct {
	PoolID
	IsIncomplete bool
	Slices       []*resourceapi.ResourceSlice
}

type PoolID struct {
	Driver, Pool string
}

func (p PoolID) String() string {
	return p.Driver + "/" + p.Pool
}

type DeviceID struct {
	Driver, Pool, Device string
}

func (d DeviceID) String() string {
	return d.Driver + "/" + d.Pool + "/" + d.Device
}
