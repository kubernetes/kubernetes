/*
Copyright The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"fmt"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
)

// This preserves the B access algorithm for an in-process comparison. PROriginal
// uses its same generation/CAS path without B's informer-store validation.
func benchmarkBMatch(c *DRACaches, slice *draapi.ResourceSlice, validate bool) *sliceMatchResults {
	var fresh *sliceMatchResults
	var match *sliceMatchResults
	for {
		value, ok := c.deviceMatchCache.Load(slice.UID)
		if ok && value.(*sliceMatchResults).generation == slice.Generation {
			match = value.(*sliceMatchResults)
			break
		}
		if fresh == nil {
			fresh = &sliceMatchResults{generation: slice.Generation}
		}
		if !ok {
			if _, loaded := c.deviceMatchCache.LoadOrStore(slice.UID, fresh); !loaded {
				match = fresh
				break
			}
		} else {
			if value.(*sliceMatchResults).generation > slice.Generation {
				match = fresh
				break
			}
			if c.deviceMatchCache.CompareAndSwap(slice.UID, value, fresh) {
				match = fresh
				break
			}
		}
	}
	if validate {
		obj, exists, err := c.resourceSliceStore.GetByKey(slice.Name)
		if err != nil || !exists || obj.(*resourceapi.ResourceSlice).UID != slice.UID || obj.(*resourceapi.ResourceSlice).Generation != slice.Generation {
			c.deviceMatchCache.CompareAndDelete(slice.UID, match)
		}
	}
	return match
}

func benchmarkDeviceMatchCache(b *testing.B, n int) (*resourceAllocationScorer, []*draapi.ResourceSlice, []*resourceapi.ResourceSlice) {
	b.Helper()
	c := &resourceAllocationScorer{DRACaches: DRACaches{celCache: cel.NewCache(10, cel.Features{})}}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	snapshots := make([]*draapi.ResourceSlice, n)
	external := make([]*resourceapi.ResourceSlice, n)
	class := &resourceapi.DeviceClass{Spec: resourceapi.DeviceClassSpec{Selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["bench-driver"].enabled`}}}}}
	for i := range n {
		external[i] = &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("slice-%d", i), UID: types.UID(fmt.Sprintf("uid-%d", i)), Generation: 10},
			Spec:       resourceapi.ResourceSliceSpec{Driver: "bench-driver", Devices: []resourceapi.Device{{Name: "gpu", Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"enabled": {BoolValue: new(i%2 == 0)}}}}},
		}
		snapshots[i] = &draapi.ResourceSlice{}
		if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(external[i], snapshots[i], nil); err != nil {
			b.Fatal(err)
		}
		if got, err := c.deviceMatchesClass(context.Background(), snapshots[i], snapshots[i].Spec.Devices[0], class); err != nil || got != (i%2 == 0) {
			b.Fatalf("populate: match=%v err=%v", got, err)
		}
		if err := store.Add(external[i]); err != nil {
			b.Fatal(err)
		}
	}
	c.resourceSliceStore = store
	return c, snapshots, external
}

func BenchmarkDeviceMatchLifecycleHit(b *testing.B) {
	for _, n := range []int{100, 1000, 5000, 10000, 20000} {
		b.Run(fmt.Sprintf("slices=%d", n), func(b *testing.B) {
			c, snapshots, _ := benchmarkDeviceMatchCache(b, n)
			for _, variant := range []string{"PROriginal", "B", "BC", "BCE"} {
				b.Run(variant, func(b *testing.B) {
					snapshot := snapshots[n/2]
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						switch variant {
						case "PROriginal":
							benchmarkBMatch(&c.DRACaches, snapshot, false)
						case "B":
							benchmarkBMatch(&c.DRACaches, snapshot, true)
						default:
							c.matchResultsForSlice(snapshot)
						}
					}
				})
			}
		})
	}
}

func BenchmarkDeviceMatchLifecycleChurn(b *testing.B) {
	for _, n := range []int{100, 1000, 5000, 10000, 20000} {
		b.Run(fmt.Sprintf("slices=%d", n), func(b *testing.B) {
			for _, variant := range []string{"PROriginal", "B", "BC", "BCE"} {
				b.Run(variant, func(b *testing.B) {
					c, snapshots, external := benchmarkDeviceMatchCache(b, n)
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						b.StopTimer()
						for i, snapshot := range snapshots {
							entry := &sliceMatchResults{generation: snapshot.Generation}
							entry.results.Store(deviceMatchKey{expression: `device.attributes["bench-driver"].enabled`, device: snapshot.Spec.Devices[0].Name}, i%2 == 0)
							c.deviceMatchCache.Store(snapshot.UID, entry)
						}
						b.StartTimer()
						switch variant {
						case "PROriginal":
							// The original PreScore prune with an empty live-slice list.
							current := make(map[types.UID]int64)
							c.deviceMatchCache.Range(func(key, value any) bool {
								if generation, ok := current[key.(types.UID)]; !ok || value.(*sliceMatchResults).generation < generation {
									c.deviceMatchCache.CompareAndDelete(key, value)
								}
								return true
							})
						case "B":
							for _, slice := range external {
								if value, ok := c.deviceMatchCache.Load(slice.UID); ok && value.(*sliceMatchResults).generation <= slice.Generation {
									c.deviceMatchCache.CompareAndDelete(slice.UID, value)
								}
							}
						case "BC":
							for _, slice := range external {
								c.invalidateDeviceMatchCache(slice)
							}
						case "BCE":
							for _, slice := range external {
								c.invalidateDeviceMatchCache(slice)
							}
							c.gcDeviceMatchCache()
						}
					}
					b.StopTimer()
					b.ReportMetric(float64(n), "entries/op")
				})
			}
		})
	}
}

func BenchmarkDeviceMatchGCBatch(b *testing.B) {
	c, snapshots, _ := benchmarkDeviceMatchCache(b, gcBatchSize)
	c.resourceSliceStore = cache.NewStore(cache.MetaNamespaceKeyFunc)
	b.ReportAllocs()
	inspected := 0
	for range b.N {
		b.StopTimer()
		for _, snapshot := range snapshots {
			c.deviceMatchCache.Store(snapshot.UID, &sliceMatchResults{generation: snapshot.Generation, name: snapshot.Name})
		}
		b.StartTimer()
		inspected += c.gcDeviceMatchCache().inspected
	}
	b.ReportMetric(float64(inspected)/float64(b.N), "entries/op")
}
