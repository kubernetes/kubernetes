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
	"context"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

const interleavePool2 = "pool2"

// newInterleaveTracker drives the real reconciler explicitly (no goroutine) so
// tests control exactly when a pool pass runs relative to cache events.
func newInterleaveTracker(t *testing.T, slices []*resourceapi.ResourceSlice) (*claimTracker, *mutableSliceLister) {
	t.Helper()
	setSharingAffinityAccountingGates(t, true, true)
	c := newSharingAffinityAccountingTracker(t, true)
	a := c.allocatedDevices
	lister := &mutableSliceLister{slices: slices}
	c.affinity.slices, c.metadataSlices = lister, lister
	c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
	a.metadata, a.tracker, a.claimsSynced = c.metadata, c, true
	c.publishMetadata(true, nil)
	a.reconcile = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[poolName]())
	t.Cleanup(a.reconcile.ShutDown)
	return c, lister
}

func interleaveDrain(t *testing.T, ctx context.Context, c *claimTracker) {
	t.Helper()
	for i := 0; c.allocatedDevices.reconcile.Len() > 0; i++ {
		if i > 20 {
			t.Fatal("reconcile queue did not drain")
		}
		c.processNext(ctx)
	}
}

func interleaveReady(c *claimTracker, pool poolName) bool {
	a := c.allocatedDevices
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	return a.poolReadyLocked(pool)
}

func interleaveRecord(c *claimTracker, key ownerPoolKey) (ownerPoolRecord, bool) {
	a := c.allocatedDevices
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	rec := a.owners[key]
	if rec == nil {
		return ownerPoolRecord{}, false
	}
	return *rec, true
}

func interleaveAllCoherent(t *testing.T, c *claimTracker) {
	t.Helper()
	a := c.allocatedDevices
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	for key, rec := range a.owners {
		if !a.recordCoherentLocked(key, rec) {
			t.Errorf("owner %+v not settled: pending=%t reason=%q", key, rec.pending, rec.reason)
		}
	}
}

// A cache event that lands while the worker is between compute
// and commit must supersede the worker's token; the worker must neither commit
// stale contributions nor requeue the pool.
func TestSharingAffinityReconciliation_CacheInterleavings(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	blue, red := map[string]string{"subnet": "blue"}, map[string]string{"subnet": "red"}
	// fenced marks an expected Unreconstructable result.
	fenced := map[string]string{"<fenced>": ""}
	withRV := func(claim *resourceapi.ResourceClaim, uid types.UID, rv, subnet string) *resourceapi.ResourceClaim {
		out := sharingAffinityAccountingClaim(claim.UID, subnet)
		out.UID, out.ResourceVersion = uid, rv
		return out
	}
	cases := []struct {
		name string
		// mutate runs inside the paused worker and returns the expected owner UID
		// (empty: no owner) and expected affinity at commit time.
		mutate     func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string)
		afterUnblk func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string)
	}{
		{
			name: "same-uid-update",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				assumecache.UpdateTestObject(c.cache, withRV(owner, owner.UID, "2", "blue"))
				return owner.UID, blue
			},
		},
		{
			name: "delete",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				assumecache.DeleteTestObject(c.cache, owner)
				return "", nil
			},
		},
		{
			name: "new-uid",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				assumecache.DeleteTestObject(c.cache, owner)
				assumecache.AddTestObject(c.cache, withRV(owner, "owner-new", "2", "red"))
				return "owner-new", red
			},
		},
		{
			name: "same-uid-readd",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				assumecache.DeleteTestObject(c.cache, owner)
				assumecache.AddTestObject(c.cache, withRV(owner, owner.UID, "2", "blue"))
				return owner.UID, blue
			},
		},
		{
			// Equal-RV Assume with a changed config source, then Restore.
			// The established blue lock must be fenced, never silently
			// replaced by red nor kept as a stale blue lock.
			name: "assume-changed-then-restore",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				if err := c.cache.Assume(withRV(owner, owner.UID, owner.ResourceVersion, "red")); err != nil {
					t.Fatal(err)
				}
				return owner.UID, fenced
			},
			afterUnblk: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				c.cache.Restore(owner.Namespace + "/" + owner.Name)
				return owner.UID, fenced
			},
		},
		{
			// Equal-RV Assume with unchanged allocation and config.
			name: "assume-unchanged-then-restore",
			mutate: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				if err := c.cache.Assume(owner.DeepCopy()); err != nil {
					t.Fatal(err)
				}
				return owner.UID, blue
			},
			afterUnblk: func(t *testing.T, c *claimTracker, owner *resourceapi.ResourceClaim) (types.UID, map[string]string) {
				c.cache.Restore(owner.Namespace + "/" + owner.Name)
				return owner.UID, blue
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			base := poolEventsSlices()
			c, lister := newInterleaveTracker(t, base)
			a := c.allocatedDevices
			c.cache.AddEventHandler(a.handlers())
			owner := sharingAffinityAccountingClaim("owner", "blue")
			device := sharingAffinityAccountingDevice(owner)
			assumecache.AddTestObject(c.cache, owner)
			interleaveDrain(t, ctx, c)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, blue)

			// A same-map republication invalidates the owner by epoch so the
			// worker recomputes it; the expression reads either field.
			updated := reconcileSlices(base, 2)
			updated[0].Spec.SharingAffinity[0].Expression = `has(parameters.network) ? parameters.network : parameters.subnet`
			lister.set(updated...)
			c.publishMetadata(false, affectedPools(updated[0]))
			a.reconcile.Add(key)

			var workerToken uint64
			var wantUID types.UID
			var want map[string]string
			hooked := false
			c.reconcileAfterComputeHook = func(_ poolName, uid types.UID) {
				if hooked {
					return
				}
				hooked = true
				rec, ok := interleaveRecord(c, ownerPoolKey{uid: uid, pool: key})
				if !ok || !rec.pending {
					t.Fatalf("worker computing without a pending record: %+v", rec)
				}
				workerToken = rec.token
				wantUID, want = tc.mutate(t, c, owner)
			}
			if !c.processNext(ctx) {
				t.Fatal("worker stopped")
			}
			c.reconcileAfterComputeHook = nil
			if !hooked {
				t.Fatal("worker did not compute the owner")
			}
			// Supersession is not a failure: the worker must not rate-limit
			// the pool (the cache event itself may enqueue it).
			if n := a.reconcile.NumRequeues(key); n != 0 {
				t.Errorf("superseded worker requeued pool with backoff: requeues=%d", n)
			}
			if rec, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: key}); ok && rec.token == workerToken {
				t.Errorf("worker token %d survived a concurrent cache event", workerToken)
			}
			check := func(wantUID types.UID, want map[string]string) {
				t.Helper()
				interleaveDrain(t, ctx, c)
				interleaveAllCoherent(t, c)
				if !interleaveReady(c, key) {
					t.Error("pool not ready after interleaving settled")
				}
				state := gatherSharingAffinityAccounting(t, c)
				if len(state.DeviceSharingAffinity.StalePools) != 0 {
					t.Errorf("StalePools=%v", state.DeviceSharingAffinity.StalePools)
				}
				a.mutex.RLock()
				owners := len(a.owners)
				_, has := a.owners[ownerPoolKey{uid: wantUID, pool: key}]
				a.mutex.RUnlock()
				if wantUID == "" {
					if owners != 0 {
						t.Errorf("owners=%d after delete, want 0", owners)
					}
					assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusClean, nil)
					return
				}
				if owners != 1 || !has {
					t.Errorf("owners=%d hasExpected(%s)=%t", owners, wantUID, has)
				}
				if _, ok := want["<fenced>"]; ok {
					assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
					return
				}
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, want)
			}
			check(wantUID, want)
			if tc.afterUnblk != nil {
				check(tc.afterUnblk(t, c, owner))
			}
		})
	}
}

func interleaveMultiPoolSlices() []*resourceapi.ResourceSlice {
	slices := poolEventsSlices()
	for _, s := range poolEventsSlices() {
		s.Name += "-2"
		s.UID = types.UID(s.Name)
		s.Spec.Pool.Name = interleavePool2
		slices = append(slices, s)
	}
	return slices
}

func interleaveMultiPoolClaim(uid types.UID, name, rv string) *resourceapi.ResourceClaim {
	claim := sharingAffinityAccountingClaim(uid, "blue")
	claim.Name, claim.ResourceVersion = name, rv
	results := claim.Status.Allocation.Devices.Results
	second := *results[0].DeepCopy()
	second.Pool, second.ShareID = interleavePool2, ptr.To(types.UID(string(uid)+"-share2"))
	claim.Status.Allocation.Devices.Results = append(results, second)
	return claim
}

// A delete for an owner whose name is already taken by a
// replacement in the cache holds the old lock until the replacement's OnAdd.
// If the replacement disappears before OnAdd, processing one pool must release
// every held pool rather than orphaning them as permanently pending.
func TestSharingAffinityReconciliation_ReplacementHolds(t *testing.T) {
	p1, p2 := poolName{accountingDriver, accountingPool}, poolName{accountingDriver, interleavePool2}
	blue := map[string]string{"subnet": "blue"}
	for _, resolve := range []string{"replacement-onadd", "replacement-deleted-before-onadd"} {
		t.Run(resolve, func(t *testing.T) {
			ctx := t.Context()
			c, _ := newInterleaveTracker(t, interleaveMultiPoolSlices())
			a := c.allocatedDevices
			// Handlers are invoked directly to model a lagging event FIFO:
			// the cache store is already ahead of the delivered events.
			old := interleaveMultiPoolClaim("old", "shared", "1")
			d1 := sharingAffinityAccountingDevice(old)
			d2 := structured.MakeDeviceID(accountingDriver, interleavePool2, accountingDevice)
			assumecache.AddTestObject(c.cache, old)
			a.onAdd(old)
			interleaveDrain(t, ctx, c)
			state := gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingState(t, state, d1, structured.AffinityStatusLocked, blue)
			assertSharingAffinityAccountingState(t, state, d2, structured.AffinityStatusLocked, blue)

			replacement := interleaveMultiPoolClaim("new", "shared", "2")
			replacement.Status.Allocation = nil
			assumecache.DeleteTestObject(c.cache, old)
			assumecache.AddTestObject(c.cache, replacement)
			a.onDelete(old)

			checkHeld := func() {
				t.Helper()
				for _, pool := range []poolName{p1, p2} {
					rec, ok := interleaveRecord(c, ownerPoolKey{uid: old.UID, pool: pool})
					if !ok || !rec.pending || rec.reason != "replacementQueued" || rec.heldFor.Name != "shared" {
						t.Errorf("pool %v: held record=%+v present=%t", pool, rec, ok)
					}
					if interleaveReady(c, pool) {
						t.Errorf("pool %v ready while held", pool)
					}
				}
				state := gatherSharingAffinityAccounting(t, c)
				for _, pool := range []string{accountingPool, interleavePool2} {
					if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: pool}) {
						t.Errorf("held pool %s not stale: %v", pool, state.DeviceSharingAffinity.StalePools)
					}
				}
				assertSharingAffinityAccountingState(t, state, d1, structured.AffinityStatusLocked, blue)
				assertSharingAffinityAccountingState(t, state, d2, structured.AffinityStatusLocked, blue)
			}
			checkHeld()
			// A worker pass while the replacement is visible keeps the hold
			// and must not even compute the held owner.
			heldComputes := 0
			c.reconcileAfterComputeHook = func(_ poolName, uid types.UID) {
				if uid == old.UID {
					heldComputes++
				}
			}
			a.reconcile.Add(p1)
			a.reconcile.Add(p2)
			interleaveDrain(t, ctx, c)
			c.reconcileAfterComputeHook = nil
			if heldComputes != 0 {
				t.Errorf("held owner computed %d times, want 0", heldComputes)
			}
			checkHeld()

			switch resolve {
			case "replacement-onadd":
				a.onAdd(replacement)
			case "replacement-deleted-before-onadd":
				assumecache.DeleteTestObject(c.cache, replacement)
				a.reconcile.Add(p1)
				interleaveDrain(t, ctx, c)
				// Processing p1 alone must already release p2's hold, before
				// any late callback could mask an orphaned pool.
				a.mutex.RLock()
				early := len(a.owners) + len(a.holds)
				a.mutex.RUnlock()
				if early != 0 {
					t.Errorf("hold not cleaned before late callbacks: owners+holds=%d", early)
				}
				for _, pool := range []poolName{p1, p2} {
					if !interleaveReady(c, pool) {
						t.Errorf("pool %v not ready before late callbacks", pool)
					}
				}
				// The late OnAdd/OnDelete for the vanished replacement must be harmless.
				a.onAdd(replacement)
				a.onDelete(replacement)
			}
			interleaveDrain(t, ctx, c)
			a.mutex.RLock()
			owners, holds := len(a.owners), len(a.holds)
			a.mutex.RUnlock()
			if owners != 0 || holds != 0 {
				t.Errorf("orphaned held state: owners=%d holds=%d", owners, holds)
			}
			for _, pool := range []poolName{p1, p2} {
				if !interleaveReady(c, pool) {
					t.Errorf("pool %v not ready after hold released", pool)
				}
			}
			state = gatherSharingAffinityAccounting(t, c)
			if len(state.DeviceSharingAffinity.StalePools) != 0 {
				t.Errorf("StalePools=%v", state.DeviceSharingAffinity.StalePools)
			}
			assertSharingAffinityAccountingState(t, state, d1, structured.AffinityStatusClean, nil)
			assertSharingAffinityAccountingState(t, state, d2, structured.AffinityStatusClean, nil)
		})
	}
}
