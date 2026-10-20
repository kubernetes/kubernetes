/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func pendingRemovalWaiterPlugin(t *testing.T, c *claimTracker, pod *v1.Pod) (*DynamicResources, *affinityActivationHandle, func()) {
	t.Helper()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	if err := indexer.Add(pod); err != nil {
		t.Fatalf("cache waiting Pod: %v", err)
	}
	handle := &affinityActivationHandle{}
	pl := &DynamicResources{
		fh:              handle,
		affinityTracker: c,
		affinityWaiters: &affinityWaiters{podLister: corelisters.NewPodLister(indexer)},
	}
	unsubscribe := c.subscribeExclusionChanges(func(logger klog.Logger, changed sets.Set[poolName], _ bool) {
		pl.activateAffinityWaiters(logger, changed)
	})
	return pl, handle, unsubscribe
}

// Removing the owner that is keeping a pool pending must finalize
// already-computed conflicting owners before waiters are activated.
func TestSharingAffinityPendingOwnerRemovalFinalizesConflictAndActivatesWaiters(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	device := structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)

	for _, mode := range []string{"delete", "force-remove", "unreserve", "observed-delete"} {
		t.Run(mode, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			base := poolEventsSlices()
			c, lister := newInterleaveTracker(t, base)
			a := c.allocatedDevices

			blue := sharingAffinityAccountingClaim("blue-owner-"+types.UID(mode), "blue")
			red := sharingAffinityAccountingClaim("red-owner-"+types.UID(mode), "red")
			assumecache.AddTestObject(c.cache, blue)
			a.onAdd(blue)
			interleaveDrain(t, ctx, c)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

			blocker := sharingAffinityAccountingClaim("pending-blocker-"+types.UID(mode), "blue")
			switch mode {
			case "unreserve":
				unallocated := blocker.DeepCopy()
				unallocated.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unallocated)
				r, views := affinityReservationForCurrentPool(c, blocker, "blue")
				if err := c.commitAffinityReservation(r, views); err != nil {
					t.Fatalf("commit tokenized blocker: %v", err)
				}
				a.mutex.Lock()
				rec := a.owners[ownerPoolKey{uid: blocker.UID, inFlight: true, pool: key}]
				if rec == nil {
					t.Fatal("tokenized blocker did not create an in-flight owner record")
				}
				rec.pending, rec.reason = true, "held-for-unreserve"
				a.mutex.Unlock()
			case "observed-delete":
				// Observed (allocated) owner computed against an incomplete
				// pool stays pending until the worker revisits it; deleting
				// it via the informer must finalize without worker progress.
				incomplete := reconcileSlices(base, 2)
				if len(incomplete) == 0 {
					t.Fatal("pool fixture has no slices")
				}
				lister.set(incomplete[0])
				assumecache.AddTestObject(c.cache, blocker)
				a.onAdd(blocker)
				lister.set(base...)
			default:
				incomplete := reconcileSlices(base, 2)
				if len(incomplete) == 0 {
					t.Fatal("pool fixture has no slices")
				}
				lister.set(incomplete[0])
				unallocated := blocker.DeepCopy()
				unallocated.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unallocated)
				if err := c.SignalClaimPendingAllocation(blocker.UID, blocker); err != nil {
					t.Fatalf("SignalClaimPendingAllocation: %v", err)
				}
				lister.set(base...)
			}

			if interleaveReady(c, key) {
				t.Fatal("pool ready before pending blocker was removed")
			}

			assumecache.AddTestObject(c.cache, red)
			a.onAdd(red)
			if interleaveReady(c, key) {
				t.Fatal("pool ready after conflicting owner committed behind pending blocker")
			}
			state := gatherSharingAffinityAccounting(t, c)
			if len(state.DeviceSharingAffinity.StalePools) == 0 {
				t.Fatal("pending blocker did not make the pool stale")
			}
			if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusClean {
				t.Fatalf("pending blocker exposed a clean device while conflicting contributions were deferred: %#v", got)
			}

			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "waiter-" + mode, UID: types.UID("waiter-" + mode)}}
			pl, handle, unsubscribe := pendingRemovalWaiterPlugin(t, c, pod)
			defer unsubscribe()
			pl.registerAffinityWaiters(logger, pod, sets.New(key))
			if handle.count() != 0 {
				t.Fatal("waiter activated while the conflicting pool was still pending")
			}

			var removed bool
			switch mode {
			case "delete":
				removed = c.MaybeRemoveClaimPendingAllocation(blocker.UID, false)
			case "force-remove":
				removed = c.MaybeRemoveClaimPendingAllocation(blocker.UID, true)
			case "observed-delete":
				a.onDelete(blocker)
				removed = true
			case "unreserve":
				removed = c.ReleaseClaimPendingAllocation(blocker.UID, reservationToken(blocker.UID+"-token"))
			default:
				t.Fatalf("unknown mode %q", mode)
			}
			if !removed {
				t.Fatal("pending blocker removal returned false")
			}

			state = gatherSharingAffinityAccounting(t, c)
			if len(state.DeviceSharingAffinity.StalePools) != 0 {
				t.Fatalf("pool stale after pending blocker removal: %v", state.DeviceSharingAffinity.StalePools)
			}
			if !interleaveReady(c, key) {
				t.Fatal("pool not ready after pending blocker removal")
			}
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, state, device, "10")
			affinityWaitForActivation(t, ctx, handle, pod)
			assertAffinityWaiterIndexes(t, pl.affinityWaiters, nil, nil)

			a.mutex.RLock()
			for ownerKey, rec := range a.owners {
				if ownerKey.uid == blocker.UID {
					t.Errorf("removed blocker record still present: %+v", ownerKey)
				}
				if rec.pending {
					t.Errorf("remaining owner %v still pending after removal: reason=%q", ownerKey, rec.reason)
				}
			}
			a.mutex.RUnlock()
			if pending := c.GetPendingAllocation(blocker.UID); pending != nil {
				t.Fatalf("removed blocker still has pending allocation: %+v", pending)
			}
			if got := handle.count(); got != 1 {
				t.Fatalf("waiter activations=%d, want exactly one", got)
			}
		})
	}
}
