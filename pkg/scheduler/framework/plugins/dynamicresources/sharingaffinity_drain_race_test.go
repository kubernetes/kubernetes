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

package dynamicresources

import (
	"fmt"
	"sync"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// Observed removal, in-flight release and fencing publications race on one
// device. Afterwards the in-flight mirror must hold and the device is fenced
// iff an owner remains.
func TestAffinityDrainOwnershipSynchronized(t *testing.T) {
	for _, tc := range []struct{ keepOwner, tokenized bool }{{false, false}, {true, false}, {false, true}, {true, true}} {
		keepOwner, tokenized := tc.keepOwner, tc.tokenized
		t.Run(fmt.Sprintf("keepOwner=%t/tokenized=%t", keepOwner, tokenized), func(t *testing.T) {
			setSharingAffinityAccountingGates(t, true, true)
			c := newSharingAffinityAccountingTracker(t, true)
			base := c.affinity.slices.(accountingSliceLister).slices
			lister := &mutableSliceLister{slices: base}
			c.affinity.slices = lister
			c.metadataSlices = lister
			c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
			c.allocatedDevices.metadata = c.metadata
			_, ctx := ktesting.NewTestContext(t)
			c.initReconciler(ctx)
			c.cache.AddEventHandler(c.allocatedDevices.handlers())
			// This synthetic cache has no informer or initial objects.
			c.allocatedDevices.claimsSynced = true
			c.publishMetadata(true, nil)

			observed := sharingAffinityAccountingClaim("observed", "blue")
			device := sharingAffinityAccountingDevice(observed)
			assumecache.AddTestObject(c.cache, observed)
			pending := sharingAffinityAccountingClaim("pending", "blue")
			unallocated := pending.DeepCopy()
			unallocated.Status.Allocation = nil
			assumecache.AddTestObject(c.cache, unallocated)
			if err := c.SignalClaimPendingAllocation(pending.UID, pending); err != nil {
				t.Fatalf("signal in-flight: %v", err)
			}
			// Unreserve releases per-pod tokens; two joined reservations make
			// the race cover both non-final and final token release.
			tokens := []reservationToken{"token-a", "token-b"}
			if tokenized {
				c.inFlightMutex.Lock()
				entry := c.inFlightAllocations[pending.UID]
				entry.sharers = 0
				entry.tokens = sets.New(tokens...)
				c.inFlightAllocations[pending.UID] = entry
				c.inFlightMutex.Unlock()
			}
			var released [2]bool
			if keepOwner {
				anchor := sharingAffinityAccountingClaim("anchor", "blue")
				assumecache.AddTestObject(c.cache, anchor)
			}
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

			var wg sync.WaitGroup
			start := make(chan struct{})
			wg.Go(func() {
				<-start
				assumecache.DeleteTestObject(c.cache, observed)
			})
			wg.Go(func() {
				<-start
				if !tokenized {
					c.MaybeRemoveClaimPendingAllocation(pending.UID, true)
					return
				}
				released[0] = c.ReleaseClaimPendingAllocation(pending.UID, tokens[0])
			})
			wg.Go(func() {
				<-start
				if tokenized {
					released[1] = c.ReleaseClaimPendingAllocation(pending.UID, tokens[1])
				}
			})
			wg.Go(func() {
				<-start
				for i := range 20 {
					next := make([]*resourceapi.ResourceSlice, len(base))
					for j, s := range base {
						next[j] = s.DeepCopy()
						next[j].Spec.Pool.Generation = int64(i + 2)
					}
					next[0].Spec.SharingAffinity[0].Expression = fmt.Sprintf(`has(parameters.subnet) ? parameters.subnet + "-%d" : ""`, i)
					lister.set(next...)
					c.publishMetadata(false, affectedPools(next[0]))
				}
			})
			close(start)
			wg.Wait()
			// Digest changes only make a pool pending. The current extracted
			// map, not publication itself, establishes a semantic fence.
			want := structured.AffinityStatusClean
			if keepOwner {
				want = structured.AffinityStatusUnreconstructable
			}
			reconcileWaitState(t, ctx, c, device, want, nil)

			if tokenized && released[0] == released[1] {
				t.Errorf("final-token removals = %v, want exactly one", released)
			}
			c.inFlightMutex.RLock()
			_, stillPending := c.inFlightAllocations[pending.UID]
			mirror := c.checkInFlightMirrorLocked()
			c.inFlightMutex.RUnlock()
			if stillPending {
				t.Error("pending allocation survived release")
			}
			if !mirror {
				t.Error("in-flight owner mirror diverged from in-flight allocations")
			}
			a := c.allocatedDevices
			a.mutex.RLock()
			owner, fenced := a.hasOwnerLocked(device), a.fenced.Has(device)
			a.mutex.RUnlock()
			if owner != keepOwner {
				t.Fatalf("owner remains = %t, want %t", owner, keepOwner)
			}
			if fenced != owner {
				t.Errorf("device fenced = %t with owner = %t", fenced, owner)
			}
		})
	}
}
