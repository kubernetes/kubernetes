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
	"errors"
	"fmt"
	"maps"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

func affinityReservationForCurrentPool(c *claimTracker, claim *resourceapi.ResourceClaim, subnet string) (affinityReservation, map[poolName]poolMeta) {
	key := poolName{accountingDriver, accountingPool}
	view := c.metadata.pools[key]
	return affinityReservation{
		claimUID: claim.UID, claimRV: claim.ResourceVersion, claim: claim, token: reservationToken(claim.UID + "-token"),
		provenance: []structured.DeviceAffinity{{
			ResultIndex: 0, Request: "req", Device: sharingAffinityAccountingDevice(claim),
			Pool:   sharingaffinity.PoolKey{Driver: key.driver, Name: key.pool, Generation: view.generation},
			Digest: view.digest, Constrained: true, Contributes: true,
			Values: map[string]string{"subnet": subnet},
		}},
		fenceEpochs: map[poolName]uint64{key: view.epoch},
	}, map[poolName]poolMeta{key: view}
}

// Startup finalization is the first moment at which two reconstructed
// owners can become ready, and it must install the final Locked/Fenced state atomically.
func TestSharingAffinityStartupAtomicFinalizationTwoOwners(t *testing.T) {
	for _, conflict := range []bool{false, true} {
		t.Run(fmt.Sprintf("conflict=%t", conflict), func(t *testing.T) {
			setSharingAffinityAccountingGates(t, true, true)
			logger, baseCtx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithTimeout(baseCtx, 10*time.Second)
			defer cancel()
			key := poolName{accountingDriver, accountingPool}
			poolID := schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}
			objects := []runtime.Object{}
			for _, slice := range newSharingAffinityAccounting().slices.(accountingSliceLister).slices {
				objects = append(objects, slice.DeepCopy())
			}
			client := fake.NewClientset(objects...)
			factory := informers.NewSharedInformerFactory(client, 0)
			tracker, err := resourceslicetracker.StartTracker(ctx, resourceslicetracker.Options{SliceInformer: factory.Resource().V1().ResourceSlices(), KubeClient: client})
			if err != nil {
				t.Fatalf("StartTracker: %v", err)
			}
			claimInformer := &reconstructionClaimInformer{synced: make(chan struct{}), observed: make(chan struct{})}
			claims := assumecache.NewAssumeCache(logger, claimInformer, "startup atomic claims", "", nil)
			manager := NewDRAManager(ctx, claims, tracker, factory)
			c := manager.resourceClaimTracker
			metadataEntered, metadataRelease := make(chan struct{}), make(chan struct{})
			c.metadataSyncedHook = func() {
				close(metadataEntered)
				select {
				case <-metadataRelease:
				case <-ctx.Done():
				}
			}
			// Hold every worker pass so startup finalization must be complete
			// without reconciliation. Installed before Start, so before any enqueue.
			workerEntered, workerRelease := make(chan struct{}, 64), make(chan struct{})
			c.reconcileProcessHook = func(poolName) {
				select {
				case workerEntered <- struct{}{}:
				default:
				}
				select {
				case <-workerRelease:
				case <-ctx.Done():
				}
			}
			var releaseOnce sync.Once
			releaseWorker := func() { releaseOnce.Do(func() { close(workerRelease) }) }
			defer releaseWorker()
			factory.Start(ctx.Done())
			defer factory.Shutdown()
			select {
			case <-metadataEntered:
			case <-ctx.Done():
				t.Fatal("metadata registration did not reach held startup barrier")
			}

			a := sharingAffinityAccountingClaim("startup-a", "blue")
			bSubnet := "blue"
			if conflict {
				bSubnet = "red"
			}
			b := sharingAffinityAccountingClaim("startup-b", bSubnet)
			claimInformer.handler.OnAdd(a, true)
			claimInformer.handler.OnAdd(b, true)
			device := sharingAffinityAccountingDevice(a)
			stopGather := make(chan struct{})
			wantStatus, wantValues := structured.AffinityStatusLocked, map[string]string{"subnet": "blue"}
			if conflict {
				wantStatus, wantValues = structured.AffinityStatusUnreconstructable, nil
			}
			var readyBeforeFinal, intermediateReady atomic.Bool
			var gathers atomic.Int32
			var gatherWG sync.WaitGroup
			gatherWG.Add(1)
			go func() {
				defer gatherWG.Done()
				for {
					select {
					case <-stopGather:
						return
					default:
					}
					state, _, err := c.gatherAllocatedStateWithMetadata()
					if err == nil && state != nil {
						gathers.Add(1)
						if !state.DeviceSharingAffinity.StalePools.Has(poolID) {
							c.allocatedDevices.mutex.RLock()
							claimsSynced := c.allocatedDevices.claimsSynced
							c.allocatedDevices.mutex.RUnlock()
							if !claimsSynced {
								readyBeforeFinal.Store(true)
							}
							got := state.DeviceSharingAffinity.DeviceStates[device]
							if got.Status != wantStatus || !maps.Equal(got.LockedAffinity, wantValues) {
								intermediateReady.Store(true)
							}
						}
					}
				}
			}()
			state := gatherSharingAffinityAccounting(t, c)
			if !state.DeviceSharingAffinity.StalePools.Has(poolID) {
				t.Fatal("startup pool ready before any registration synced")
			}
			assertSharingAffinityAccountingCapacity(t, state, device, "10")
			close(metadataRelease)
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
				c.allocatedDevices.mutex.RLock()
				defer c.allocatedDevices.mutex.RUnlock()
				return c.metadata.synced, nil
			}); err != nil {
				t.Fatalf("metadata did not sync: %v", err)
			}
			state = gatherSharingAffinityAccounting(t, c)
			if !state.DeviceSharingAffinity.StalePools.Has(poolID) {
				t.Fatal("startup pool ready before claim registration synced")
			}
			// The gather keeps running across the claim-sync transition so that a
			// ready-but-not-final publication would be observed.
			close(claimInformer.synced)
			select {
			case <-claimInformer.observed:
			case <-time.After(time.Second):
				t.Fatal("claim registration sync was not observed")
			}
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
				state, err := c.GatherAllocatedState()
				return err == nil && !state.DeviceSharingAffinity.StalePools.Has(poolID), nil
			}); err != nil {
				t.Fatalf("startup did not become ready with the worker held: %v", err)
			}
			close(stopGather)
			gatherWG.Wait()
			if gathers.Load() == 0 {
				t.Fatal("concurrent gather did not observe the held startup interval")
			}
			if readyBeforeFinal.Load() {
				t.Fatal("concurrent gather observed readiness before claim registration synced")
			}
			if intermediateReady.Load() {
				t.Fatal("concurrent gather observed a ready pool with non-final affinity state during startup")
			}
			func() {
				c.allocatedDevices.mutex.RLock()
				defer c.allocatedDevices.mutex.RUnlock()
				if !c.allocatedDevices.startupReady {
					t.Fatal("startupReady not set by finalization")
				}
				owners := 0
				for key, rec := range c.allocatedDevices.owners {
					if rec.pending {
						t.Fatalf("owner %v still pending after startup finalization with worker held (reason %q)", key, rec.reason)
					}
					owners++
				}
				if owners != 2 {
					t.Fatalf("got %d settled owner records, want 2", owners)
				}
			}()
			state = gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingState(t, state, device, wantStatus, wantValues)
			assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "10")
			assumecache.DeleteTestObject(c.cache, a)
			state = gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingCapacity(t, state, device, "5")
			if conflict {
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
				candidate := sharingAffinityAccountingClaim("startup-candidate", "green")
				unallocated := candidate.DeepCopy()
				unallocated.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unallocated)
				r, views := affinityReservationForCurrentPool(c, candidate, "green")
				err := c.commitAffinityReservation(r, views)
				if !errors.Is(err, errAffinityConflict) {
					t.Fatalf("Reserve against partially drained startup conflict = %v, want errAffinityConflict", err)
				}
				if c.GetPendingAllocation(candidate.UID) != nil {
					t.Fatal("rejected Reserve left a pending allocation")
				}
			} else {
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			}
			select {
			case <-workerEntered:
			default:
				t.Fatal("startup enqueued no reconciliation pass; the held-worker barrier was not exercised")
			}
			releaseWorker()
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
				return c.allocatedDevices.reconcile.Len() == 0, nil
			}); err != nil {
				t.Fatalf("worker did not drain: %v", err)
			}
			state = gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingCapacity(t, state, device, "5")
			assertSharingAffinityAccountingState(t, state, device, wantStatus, wantValues)
		})
	}
}

// Removing a pending owner must finalize the remaining owners in one
// critical section; unrelated pools must not become stale or change epoch.
func TestSharingAffinityPendingRemovalAndPoolIsolation(t *testing.T) {
	t.Run("forced-pending-removal-finalizes", func(t *testing.T) {
		ctx := t.Context()
		c, _ := newInterleaveTracker(t, poolEventsSlices())
		owner := sharingAffinityAccountingClaim("settled-owner", "blue")
		pending := sharingAffinityAccountingClaim("pending-owner", "blue")
		unallocatedPending := pending.DeepCopy()
		unallocatedPending.Status.Allocation = nil
		assumecache.AddTestObject(c.cache, unallocatedPending)
		assumecache.AddTestObject(c.cache, owner)
		c.allocatedDevices.onAdd(owner)
		interleaveDrain(t, ctx, c)
		if err := c.SignalClaimPendingAllocation(pending.UID, pending); err != nil {
			t.Fatalf("SignalClaimPendingAllocation: %v", err)
		}
		device := sharingAffinityAccountingDevice(owner)
		assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "10")
		if deleted := c.MaybeRemoveClaimPendingAllocation(pending.UID, true); !deleted {
			t.Fatal("forced pending removal did not delete in-flight owner")
		}
		state := gatherSharingAffinityAccounting(t, c)
		if len(state.DeviceSharingAffinity.StalePools) != 0 {
			t.Fatalf("pool stale after forced pending removal: %v", state.DeviceSharingAffinity.StalePools)
		}
		assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		assertSharingAffinityAccountingCapacity(t, state, device, "5")
		c.inFlightMutex.RLock()
		_, stillPending := c.inFlightAllocations[pending.UID]
		mirrorOK := c.checkInFlightMirrorLocked()
		c.inFlightMutex.RUnlock()
		if stillPending || !mirrorOK {
			t.Fatalf("pending owner cleanup incomplete: stillPending=%t mirrorOK=%t", stillPending, mirrorOK)
		}
	})

	t.Run("p1-ready-while-p2-incomplete", func(t *testing.T) {
		ctx := t.Context()
		p1, p2 := poolName{accountingDriver, accountingPool}, poolName{accountingDriver, interleavePool2}
		base := interleaveMultiPoolSlices()
		c, lister := newInterleaveTracker(t, base)
		owner := interleaveMultiPoolClaim("two-pool-owner", "two-pool-owner", "1")
		assumecache.AddTestObject(c.cache, owner)
		c.allocatedDevices.onAdd(owner)
		interleaveDrain(t, ctx, c)
		recBefore, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: p1})
		if !ok || recBefore.pending {
			t.Fatalf("p1 record not settled before p2 change: %+v", recBefore)
		}
		changed := make([]*resourceapi.ResourceSlice, 0, len(base)-1)
		for _, slice := range base {
			copy := slice.DeepCopy()
			if copy.Spec.Pool.Name == interleavePool2 {
				copy.Spec.Pool.Generation++
				copy.Spec.Pool.ResourceSliceCount = 3
				if len(copy.Spec.SharingAffinity) != 0 {
					copy.Spec.SharingAffinity[0].Expression = `"red"`
				}
			}
			if !(copy.Spec.Pool.Name == interleavePool2 && len(copy.Spec.Devices) != 0) {
				changed = append(changed, copy)
			}
		}
		lister.set(changed...)
		c.publishMetadata(false, sets.New(p2))
		state := gatherSharingAffinityAccounting(t, c)
		if state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: p1.driver, Pool: p1.pool}) {
			t.Fatalf("p1 became stale after p2-only publication: %v", state.DeviceSharingAffinity.StalePools)
		}
		if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: p2.driver, Pool: p2.pool}) {
			t.Fatalf("p2 not stale after incomplete publication: %v", state.DeviceSharingAffinity.StalePools)
		}
		recAfter, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: p1})
		if !ok || recAfter.pending || recAfter.epoch != recBefore.epoch || !reflect.DeepEqual(recAfter.contributions, recBefore.contributions) {
			t.Fatalf("p1 record changed during p2 reconciliation: before=%+v after=%+v present=%t", recBefore, recAfter, ok)
		}
		assertSharingAffinityAccountingState(t, state, sharingAffinityAccountingDevice(owner), structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

		// Actual worker commits for P2 while it is incomplete must neither
		// touch P1 nor let P1's state make P2 ready.
		c.allocatedDevices.reconcile.Add(p1)
		c.allocatedDevices.reconcile.Add(p2)
		interleaveDrain(t, ctx, c)
		recAfterWorker, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: p1})
		if !ok || recAfterWorker.pending || recAfterWorker.epoch != recBefore.epoch || !reflect.DeepEqual(recAfterWorker.contributions, recBefore.contributions) {
			t.Fatalf("p1 record changed by worker commits while p2 incomplete: before=%+v after=%+v present=%t", recBefore, recAfterWorker, ok)
		}
		if !interleaveReady(c, p1) {
			t.Fatal("p1 not ready after worker commits while p2 incomplete")
		}
		if interleaveReady(c, p2) {
			t.Fatal("p2 became ready while its publication is incomplete")
		}
		if rec2, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: p2}); !ok || !rec2.pending {
			t.Fatalf("p2 owner record not pending while incomplete: %+v present=%t", rec2, ok)
		}

		// Completing P2 settles P2 without changing P1.
		completed := make([]*resourceapi.ResourceSlice, 0, len(base))
		for _, slice := range base {
			copy := slice.DeepCopy()
			if copy.Spec.Pool.Name == interleavePool2 {
				copy.Spec.Pool.Generation++
			}
			completed = append(completed, copy)
		}
		lister.set(completed...)
		c.publishMetadata(false, sets.New(p2))
		interleaveDrain(t, ctx, c)
		if !interleaveReady(c, p2) {
			t.Fatal("p2 not ready after complete publication and worker commit")
		}
		recFinal, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: p1})
		if !ok || recFinal.pending || recFinal.epoch != recBefore.epoch || !reflect.DeepEqual(recFinal.contributions, recBefore.contributions) {
			t.Fatalf("p1 record changed by p2 settlement: before=%+v after=%+v present=%t", recBefore, recFinal, ok)
		}
	})
}

// An allocated same-name replacement is a hold, not a transient drain.
func TestSharingAffinityAllocatedReplacementHoldVariants(t *testing.T) {
	ctx := t.Context()
	key := poolName{accountingDriver, accountingPool}
	c, _ := newInterleaveTracker(t, poolEventsSlices())
	a := c.allocatedDevices
	a.reconcile = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[poolName]())
	defer a.reconcile.ShutDown()
	old := sharingAffinityAccountingClaim("old-replacement", "blue")
	old.Name = "shared-name"
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	interleaveDrain(t, ctx, c)
	device := sharingAffinityAccountingDevice(old)
	assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
	replacement := sharingAffinityAccountingClaim("new-replacement", "red")
	replacement.Name, replacement.ResourceVersion = old.Name, "2"
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	state := gatherSharingAffinityAccounting(t, c)
	if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
		t.Fatalf("held replacement pool not stale: %v", state.DeviceSharingAffinity.StalePools)
	}
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	// The departed owner's capacity is released during the hold.
	if q := state.AggregatedCapacity[device][accountingCapacity]; q != nil && !q.IsZero() {
		t.Fatalf("held replacement retained old capacity: %v", q)
	}
	candidate := sharingAffinityAccountingClaim("replacement-candidate", "blue")
	unallocated := candidate.DeepCopy()
	unallocated.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, unallocated)
	r, views := affinityReservationForCurrentPool(c, candidate, "blue")
	if err := c.commitAffinityReservation(r, views); !errors.Is(err, errAffinityNotReady) {
		t.Fatalf("Reserve during replacement hold = %v, want errAffinityNotReady", err)
	}
	if n := a.reconcile.NumRequeues(key); n != 0 {
		t.Fatalf("replacement hold requeued before worker compute: %d", n)
	}
	a.reconcile.Add(key)
	interleaveDrain(t, ctx, c)
	if n := a.reconcile.NumRequeues(key); n != 0 {
		t.Fatalf("held replacement worker requeued: %d", n)
	}
	var during *structured.AllocatedState
	a.observedAfterComputeHook = func() { during = gatherSharingAffinityAccounting(t, c) }
	a.onAdd(replacement)
	a.observedAfterComputeHook = nil
	if during == nil {
		t.Fatal("replacement OnAdd did not compute allocated replacement")
	}
	if got := during.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusClean {
		t.Fatalf("replacement exposed a transient Clean snapshot: %#v", during)
	}
	interleaveDrain(t, ctx, c)
	state = gatherSharingAffinityAccounting(t, c)
	if len(state.DeviceSharingAffinity.StalePools) != 0 {
		t.Fatalf("replacement did not settle stale pools: %v", state.DeviceSharingAffinity.StalePools)
	}
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
	assertSharingAffinityAccountingCapacity(t, state, device, "5")
	a.mutex.RLock()
	owners, holds := len(a.owners), len(a.holds)
	a.mutex.RUnlock()
	if owners != 1 || holds != 0 {
		t.Fatalf("replacement state owners=%d holds=%d, want one replacement owner and no holds", owners, holds)
	}
}

// An incompatible allocated replacement while another owner
// (observed or in-flight) survives must not drain; the device stays ambiguous
// until the survivor leaves, then locks to the replacement.
func TestSharingAffinityAllocatedReplacementWithSurvivor(t *testing.T) {
	for _, inFlight := range []bool{false, true} {
		t.Run(fmt.Sprintf("inFlightSurvivor=%t", inFlight), func(t *testing.T) {
			ctx := t.Context()
			key := poolName{accountingDriver, accountingPool}
			poolID := schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}
			c, _ := newInterleaveTracker(t, poolEventsSlices())
			a := c.allocatedDevices
			old := sharingAffinityAccountingClaim("old-replacement", "blue")
			old.Name = "shared-name"
			assumecache.AddTestObject(c.cache, old)
			a.onAdd(old)
			survivor := sharingAffinityAccountingClaim("survivor", "blue")
			if inFlight {
				unalloc := survivor.DeepCopy()
				unalloc.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unalloc)
				if err := c.SignalClaimPendingAllocation(survivor.UID, survivor); err != nil {
					t.Fatalf("SignalClaimPendingAllocation: %v", err)
				}
			} else {
				assumecache.AddTestObject(c.cache, survivor)
				a.onAdd(survivor)
			}
			interleaveDrain(t, ctx, c)
			device := sharingAffinityAccountingDevice(old)
			assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "10")

			replacement := sharingAffinityAccountingClaim("new-replacement", "red")
			replacement.Name, replacement.ResourceVersion = old.Name, "2"
			assumecache.DeleteTestObject(c.cache, old)
			assumecache.AddTestObject(c.cache, replacement)
			a.onDelete(old)
			state := gatherSharingAffinityAccounting(t, c)
			// Survivor keeps the lock; no drain to Clean during the hold.
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			assertSharingAffinityAccountingCapacity(t, state, device, "5")

			a.onAdd(replacement)
			interleaveDrain(t, ctx, c)
			state = gatherSharingAffinityAccounting(t, c)
			if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusClean {
				t.Fatalf("replacement with survivor drained to Clean: %#v", got)
			}
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, state, device, "10")

			// Partial drain: ambiguity is sticky until the device fully drains.
			if inFlight {
				if !c.MaybeRemoveClaimPendingAllocation(survivor.UID, true) {
					t.Fatal("survivor in-flight removal failed")
				}
			} else {
				assumecache.DeleteTestObject(c.cache, survivor)
				a.onDelete(survivor)
			}
			interleaveDrain(t, ctx, c)
			state = gatherSharingAffinityAccounting(t, c)
			if state.DeviceSharingAffinity.StalePools.Has(poolID) {
				t.Fatalf("pool stale after survivor removal: %v", state.DeviceSharingAffinity.StalePools)
			}
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, state, device, "5")

			// Full drain.
			assumecache.DeleteTestObject(c.cache, replacement)
			a.onDelete(replacement)
			interleaveDrain(t, ctx, c)
			state = gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusClean, nil)
			if q := state.AggregatedCapacity[device][accountingCapacity]; q != nil && !q.IsZero() {
				t.Fatalf("capacity after full drain = %v", q)
			}
		})
	}
}

// Replacement-hold handoff: admin-only replacements release the hold, devices
// outside the replacement are finalized, and a worker-settled replacement
// drops the departed baseline.
func TestSharingAffinityReplacementHoldAdminOnly(t *testing.T) {
	c, _ := newInterleaveTracker(t, poolEventsSlices())
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	replacement := sharingAffinityAccountingClaim("new", "red")
	replacement.Name = old.Name
	replacement.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	a.onAdd(replacement)
	a.reconcile.Add(poolName{accountingDriver, accountingPool})
	interleaveDrain(t, t.Context(), c)
	if len(a.holds) != 0 || len(a.owners) != 0 {
		t.Fatalf("admin-only replacement retains departed owner: holds=%v owners=%v stale=%v", a.holds, a.owners, gatherSharingAffinityAccounting(t, c).DeviceSharingAffinity.StalePools)
	}
}

func TestSharingAffinityReplacementHoldDifferentDeviceFinalization(t *testing.T) {
	base := poolEventsSlices()
	second := *base[1].Spec.Devices[0].DeepCopy()
	second.Name = "different-device"
	base[1].Spec.Devices = append(base[1].Spec.Devices, second)
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	survivor := sharingAffinityAccountingClaim("survivor", "blue")
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	assumecache.AddTestObject(c.cache, survivor)
	a.onAdd(survivor)
	device := sharingAffinityAccountingDevice(old)
	replacement := sharingAffinityAccountingClaim("new", "red")
	replacement.Name = old.Name
	replacement.Status.Allocation.Devices.Results[0].Device = "different-device"
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	changed := reconcileSlices(base, 2)
	changed[0].Spec.SharingAffinity[0].Expression = `"red"`
	lister.set(changed...)
	c.publishMetadata(false, nil)
	interleaveDrain(t, t.Context(), c)
	a.onAdd(replacement)
	interleaveDrain(t, t.Context(), c)
	state := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingCapacity(t, state, device, "5")
	if len(state.DeviceSharingAffinity.StalePools) != 0 {
		t.Fatalf("replacement did not settle: %v", state.DeviceSharingAffinity.StalePools)
	}
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
	candidate := sharingAffinityAccountingClaim("candidate", "red")
	unallocated := candidate.DeepCopy()
	unallocated.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, unallocated)
	reservation, views := affinityReservationForCurrentPool(c, candidate, "red")
	if err := c.commitAffinityReservation(reservation, views); err == nil {
		t.Fatal("Reserve admitted a red sharer while the surviving blue-baseline owner had not drained")
	}
}

func TestSharingAffinityReplacementHoldWorkerTakeover(t *testing.T) {
	c, lister := newInterleaveTracker(t, poolEventsSlices())
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	replacement := sharingAffinityAccountingClaim("new", "red")
	replacement.Name = old.Name
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	a.observedAfterComputeHook = func() {
		a.reconcile.Add(poolName{accountingDriver, accountingPool})
		interleaveDrain(t, t.Context(), c)
	}
	a.onAdd(replacement)
	a.observedAfterComputeHook = nil
	// A later same-input metadata generation must preserve the replacement's lock.
	lister.set(reconcileSlices(poolEventsSlices(), 2)...)
	c.publishMetadata(false, nil)
	interleaveDrain(t, t.Context(), c)
	assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), sharingAffinityAccountingDevice(old), structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
}

// Worker-settled and mixed-admin replacements keep an established baseline,
// so an immediately changed map fences the occupied device.
func TestSharingAffinityReplacementWorkerBaselineMutation(t *testing.T) {
	c, lister := newInterleaveTracker(t, poolEventsSlices())
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	replacement := sharingAffinityAccountingClaim("new", "red")
	replacement.Name = old.Name
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	a.observedAfterComputeHook = func() {
		a.reconcile.Add(poolName{accountingDriver, accountingPool})
		interleaveDrain(t, t.Context(), c)
	}
	a.onAdd(replacement)
	a.observedAfterComputeHook = nil
	device := sharingAffinityAccountingDevice(old)
	changed := reconcileSlices(poolEventsSlices(), 2)
	changed[0].Spec.SharingAffinity[0].Expression = `"green"`
	lister.set(changed...)
	c.publishMetadata(false, nil)
	interleaveDrain(t, t.Context(), c)
	assertReplacementFencedAgainstGreen(t, c, device)
}

func TestSharingAffinityReplacementMixedAdminFinalization(t *testing.T) {
	base := poolEventsSlices()
	second := *base[1].Spec.Devices[0].DeepCopy()
	second.Name = "different-device"
	base[1].Spec.Devices = append(base[1].Spec.Devices, second)
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	old := sharingAffinityAccountingClaim("old", "blue")
	survivor := sharingAffinityAccountingClaim("survivor", "blue")
	assumecache.AddTestObject(c.cache, old)
	a.onAdd(old)
	assumecache.AddTestObject(c.cache, survivor)
	a.onAdd(survivor)
	device := sharingAffinityAccountingDevice(old)
	replacement := sharingAffinityAccountingClaim("new", "green")
	replacement.Name = old.Name
	normalRequest := *replacement.Spec.Devices.Requests[0].DeepCopy()
	normalRequest.Name = "req-normal"
	replacement.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
	replacement.Spec.Devices.Requests = append(replacement.Spec.Devices.Requests, normalRequest)
	replacement.Spec.Devices.Config[0].Requests = []string{"req", "req-normal"}
	normal := replacement.Status.Allocation.Devices.Results[0]
	normal.Device, normal.Request = "different-device", "req-normal"
	replacement.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
	replacement.Status.Allocation.Devices.Results[0].ShareID = nil
	replacement.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
	replacement.Status.Allocation.Devices.Results = append(replacement.Status.Allocation.Devices.Results, normal)
	assumecache.DeleteTestObject(c.cache, old)
	assumecache.AddTestObject(c.cache, replacement)
	a.onDelete(old)
	changed := reconcileSlices(base, 2)
	changed[0].Spec.SharingAffinity[0].Expression = `"green"`
	lister.set(changed...)
	c.publishMetadata(false, nil)
	interleaveDrain(t, t.Context(), c)
	a.onAdd(replacement)
	interleaveDrain(t, t.Context(), c)
	assertReplacementFencedAgainstGreen(t, c, device)
}

func assertReplacementFencedAgainstGreen(t *testing.T, c *claimTracker, device structured.DeviceID) {
	t.Helper()
	state := gatherSharingAffinityAccounting(t, c)
	if len(state.DeviceSharingAffinity.StalePools) != 0 {
		t.Fatalf("unexpected stale pools: %v", state.DeviceSharingAffinity.StalePools)
	}
	if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status != structured.AffinityStatusUnreconstructable {
		t.Errorf("changed allocation lock became ready: %+v; want Unreconstructable", got)
	}
	candidate := sharingAffinityAccountingClaim("candidate", "green")
	unallocated := candidate.DeepCopy()
	unallocated.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, unallocated)
	r, views := affinityReservationForCurrentPool(c, candidate, "green")
	if err := c.commitAffinityReservation(r, views); err == nil {
		t.Error("Reserve admitted a new green sharer without draining the established owner")
	}
}
