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

package experimental

import (
	"context"
	"sync"
	"testing"

	g "github.com/onsi/gomega"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/ktesting"

	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/internal/allocatortesting"
)

func TestAllocator(t *testing.T) {
	newAllocator := func(
		ctx context.Context,
		features Features,
		allocatedState AllocatedState,
		classLister DeviceClassLister,
		slices []*draapi.ResourceSlice,
		celCache *cel.Cache,
	) (internal.Allocator, error) {
		return NewAllocator(ctx, features, allocatedState, classLister, slices, celCache)
	}
	allocatortesting.TestAllocator(t,
		SupportedFeatures,
		newAllocator,
	)
	allocatortesting.TestLexicographicalAllocator(t,
		SupportedFeatures,
		newAllocator,
	)
}

// TestCounterPoolDeviceIndex keeps cached pool membership independent of node-local removals.
func TestCounterPoolDeviceIndex(t *testing.T) { testCounterPoolDeviceIndex(ktesting.Init(t)) }
func testCounterPoolDeviceIndex(tCtx ktesting.TContext) {
	id := DeviceID{Driver: draapi.MakeUniqueString(driverA), Pool: draapi.MakeUniqueString(pool1), Device: draapi.MakeUniqueString(device1)}
	shared := id
	shared.Device = draapi.MakeUniqueString("shared")
	other := id
	other.Pool = draapi.MakeUniqueString("other")
	otherDriver := id
	otherDriver.Driver = draapi.MakeUniqueString("driver-b")
	capacity := id
	capacity.Device = draapi.MakeUniqueString("capacity")
	a := &Allocator{allocatedState: AllocatedState{
		AllocatedDevices:         sets.New(id, other, otherDriver),
		AllocatedSharedDeviceIDs: sets.New(shared),
		AggregatedCapacity:       NewConsumedCapacityCollection(),
	}}
	a.allocatedState.AggregatedCapacity.Insert(DeviceConsumedCapacity{DeviceID: capacity, ConsumedCapacity: NewConsumedCapacity()})
	a.allocatedState.AggregatedCapacity.Insert(DeviceConsumedCapacity{DeviceID: shared, ConsumedCapacity: NewConsumedCapacity()})
	tCtx.Expect(a.allocatedCounterDevices).To(g.BeNil())
	pool := PoolID{Driver: id.Driver, Pool: id.Pool}
	devices := a.devicesForCounterPool(pool)
	tCtx.Expect(devices).To(g.Equal(sets.New(id, shared, capacity)))
	otherPool := PoolID{Driver: other.Driver, Pool: other.Pool}
	driverPool := PoolID{Driver: otherDriver.Driver, Pool: otherDriver.Pool}
	tCtx.Expect(a.allocatedCounterDevices[otherPool]).To(g.Equal(sets.New(other)))
	tCtx.Expect(a.allocatedCounterDevices[driverPool]).To(g.Equal(sets.New(otherDriver)))
	devices.Delete(id)
	tCtx.Expect(a.devicesForCounterPool(pool)).To(g.Equal(sets.New(id, shared, capacity)))
	tCtx.Expect(a.devicesForCounterPool(otherPool)).To(g.Equal(sets.New(other)))
	tCtx.Expect(a.devicesForCounterPool(driverPool)).To(g.Equal(sets.New(otherDriver)))
	tCtx.Expect(a.devicesForCounterPool(PoolID{Driver: id.Driver, Pool: draapi.MakeUniqueString("missing")})).To(g.BeEmpty())
	tCtx.Expect(a.allocatedCounterDevices).To(g.HaveLen(3))
}

// TestCounterPoolDeviceIndexConcurrent covers initialization and clone isolation across node searches.
func TestCounterPoolDeviceIndexConcurrent(t *testing.T) {
	testCounterPoolDeviceIndexConcurrent(ktesting.Init(t))
}
func testCounterPoolDeviceIndexConcurrent(tCtx ktesting.TContext) {
	id := MakeDeviceID(driverA, pool1, device1)
	pool := PoolID{Driver: id.Driver, Pool: id.Pool}
	a := &Allocator{allocatedState: AllocatedState{AllocatedDevices: sets.New(id)}}
	start := make(chan struct{})
	var workers sync.WaitGroup
	for range 16 {
		workers.Go(func() {
			<-start
			devices := a.devicesForCounterPool(pool)
			if !devices.Has(id) || len(devices) != 1 {
				tCtx.Errorf("unexpected allocated devices: %v", devices)
			}
			devices.Delete(id)
		})
	}
	close(start)
	workers.Wait()
	if devices := a.devicesForCounterPool(pool); !devices.Has(id) || len(devices) != 1 {
		tCtx.Fatalf("node-local removals changed cached membership: %v", devices)
	}
}

// TestAvailableCountersForPool preserves legacy accounting when an allocated device disappears.
func TestAvailableCountersForPool(t *testing.T) { testAvailableCountersForPool(ktesting.Init(t)) }
func testAvailableCountersForPool(tCtx ktesting.TContext) {
	id := MakeDeviceID(driverA, pool1, device1)
	missing := MakeDeviceID(driverA, pool1, "missing")
	set := draapi.MakeUniqueString("set")
	pool := &Pool{
		PoolID: PoolID{Driver: id.Driver, Pool: id.Pool},
		CounterSets: map[draapi.UniqueString]*draapi.CounterSet{
			set: {Name: set, Counters: map[string]resourceapi.Counter{"counter": {Value: three}}},
		},
		DeviceSlicesNotTargetingNode: []*draapi.ResourceSlice{{Spec: draapi.ResourceSliceSpec{
			Driver: id.Driver,
			Pool:   draapi.ResourcePool{Name: id.Pool},
			Devices: []draapi.Device{{Name: id.Device, ConsumesCounters: []draapi.DeviceCounterConsumption{{
				CounterSet: set,
				Counters:   map[string]resourceapi.CounterRequirement{"counter": {Value: &one}},
			}}}},
		}}},
	}
	a := &Allocator{
		allocatedState:    AllocatedState{AllocatedDevices: sets.New(id, missing)},
		availableCounters: make(map[PoolID]counterSets),
	}
	alloc := &allocator{Allocator: a}
	available := alloc.availableCountersForPool(pool)
	if got := available[set]["counter"]; got.Value.Cmp(two) != 0 {
		tCtx.Fatalf("available counter = %s, want 2", got.Value.String())
	}
	if a.allocatedCounterDevices != nil {
		tCtx.Fatal("legacy accounting initialized the shared-consumption device index")
	}
}
