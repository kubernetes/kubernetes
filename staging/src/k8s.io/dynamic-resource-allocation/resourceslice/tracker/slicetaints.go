/*
Copyright 2025 The Kubernetes Authors.

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

package tracker

import (
	"slices"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

// sliceTaintTracker keeps track of ResourceSlices with the same driver/pool/generation.
//
// Access is thread-safe.
type sliceTaintTracker struct {
	mutex  sync.RWMutex
	slices map[poolID]sets.Set[*draapi.ResourceSlice]
}

type poolID struct {
	driverName draapi.UniqueString
	poolName   draapi.UniqueString
	generation int64
}

func newPoolID(slice *draapi.ResourceSlice) poolID {
	return poolID{
		driverName: slice.Spec.Driver,
		poolName:   slice.Spec.Pool.Name,
		generation: slice.Spec.Pool.Generation,
	}
}

// insertSlice inserts a new or updated slice. If the taints in the pool
// changed, then it returns the ResourceSlices with devices, otherwise nil.
func (st *sliceTaintTracker) insertSlice(slice, oldSlice *draapi.ResourceSlice) []*draapi.ResourceSlice {
	id := newPoolID(slice)

	st.mutex.Lock()
	defer st.mutex.Unlock()

	slicesInPool := st.slices[id]
	if slicesInPool.Has(oldSlice) {
		slicesInPool.Delete(oldSlice)
		slicesInPool.Insert(slice)
		taintsChanged := slices.EqualFunc(oldSlice.Spec.Taints, slice.Spec.Taints, sliceTaintsEqual)
		if taintsChanged {
			return slicesWithDevices(slicesInPool)
		}
		return nil
	}

	// Not found. Insert it.
	if slicesInPool != nil {
		// The map entry existed. We can do an in-place insert.
		slicesInPool.Insert(slice)
		taintsChanged := len(slice.Spec.Taints) > 0
		if taintsChanged {
			return slicesWithDevices(slicesInPool)
		}
		return nil
	}
	slicesInPool = sets.New(slice)
	if st.slices == nil {
		st.slices = make(map[poolID]sets.Set[*draapi.ResourceSlice], 1)
	}
	st.slices[id] = slicesInPool
	return slicesWithDevices(slicesInPool)
}

// removeSlice removes a slice. If the taints in the pool
// changed, then it returns the ResourceSlices with devices, otherwise nil.
func (st *sliceTaintTracker) removeSlice(slice *draapi.ResourceSlice) []*draapi.ResourceSlice {
	id := newPoolID(slice)

	st.mutex.Lock()
	defer st.mutex.Unlock()

	sliceTaints := st.slices[id]
	if !sliceTaints.Has(slice) {
		return nil
	}

	sliceTaints.Delete(slice)
	if len(sliceTaints) == 0 {
		delete(st.slices, id)
	}

	taintsChanged := len(slice.Spec.Taints) > 0
	if taintsChanged {
		return slicesWithDevices(sliceTaints)
	}
	return nil
}

func slicesWithDevices(slicesInPool sets.Set[*draapi.ResourceSlice]) []*draapi.ResourceSlice {
	slices := make([]*draapi.ResourceSlice, 0, len(slicesInPool))
	for slice := range slicesInPool {
		if len(slice.Spec.Devices) > 0 {
			slices = append(slices, slice)
		}
	}
	return slices
}

// allSliceTaints returns all ResourceSlices with taints in the same pool and
// with the same generation. The order is deterministic.
func (st *sliceTaintTracker) allSliceTaints(slice *draapi.ResourceSlice) []*draapi.ResourceSlice {
	id := newPoolID(slice)

	st.mutex.RLock()
	defer st.mutex.RUnlock()

	sliceTaints := make([]*draapi.ResourceSlice, 0, 5)
	for slice := range st.slices[id] {
		if len(slice.Spec.Taints) != 0 {
			sliceTaints = append(sliceTaints, slice)
		}
	}
	slices.SortFunc(sliceTaints, func(a, b *draapi.ResourceSlice) int {
		// Arbitrarily sort by age first (fast), then by name (tie breaker).
		// What matters is that the result must be deterministic.
		res := a.CreationTimestamp.Compare(b.CreationTimestamp.Time)
		if res != 0 {
			return res
		}
		return strings.Compare(a.Name, b.Name)
	})

	return sliceTaints
}
