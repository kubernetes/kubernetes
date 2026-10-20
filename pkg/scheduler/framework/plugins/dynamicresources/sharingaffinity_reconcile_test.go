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
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

// Lister visibility alone does not imply publication or owner reconciliation.
// Queue length alone also misses an item already being processed.
func reconcileWaitForIdle(t *testing.T, ctx context.Context, c *claimTracker) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		all, err := c.metadataSlices.ListPatchedResourceSlices()
		if err != nil {
			return false, err
		}
		listed := make(map[poolName]poolMeta)
		for key, slices := range groupSlicesByPool(all) {
			listed[key] = groupedMetadataView(slices, key)
		}
		a := c.allocatedDevices
		a.mutex.RLock()
		defer a.mutex.RUnlock()
		if !a.startupReady || !c.metadata.synced {
			return false, nil
		}
		for key, view := range listed {
			if !sameMetadata(view, c.metadata.pools[key]) {
				return false, nil
			}
		}
		for key, rec := range a.owners {
			if !a.recordCoherentLocked(key, rec) {
				return false, nil
			}
		}
		return a.reconcile == nil || a.reconcile.Len() == 0, nil
	}); err != nil {
		t.Fatalf("waiting for metadata publication and owner reconciliation: %v", err)
	}
}

// Publication and reconciliation are independent. A matching affinity map alone
// is not a barrier: it could still be the pre-publication map.
func reconcileWaitState(t *testing.T, ctx context.Context, c *claimTracker, device structured.DeviceID, status structured.AffinityStatus, values map[string]string) *structured.AllocatedState {
	t.Helper()
	var state *structured.AllocatedState
	err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
		var err error
		state, err = c.GatherAllocatedState()
		if errors.Is(err, errClaimTrackerConcurrentModification) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if _, stale := state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}]; stale {
			return false, nil
		}
		got, present := state.DeviceSharingAffinity.DeviceStates[device]
		if status == structured.AffinityStatusClean {
			return !present, nil
		}
		return present && got.Status == status && maps.Equal(got.LockedAffinity, values), nil
	})
	if err != nil {
		t.Errorf("reconciliation did not settle: device=%s status=%v values=%v state=%+v: %v", device, status, values, state, err)
	}
	if state == nil {
		t.Fatal("GatherAllocatedState returned no snapshot")
	}
	return state
}

func reconcilePublish(t *testing.T, ctx context.Context, tc *testContext, slices []*resourceapi.ResourceSlice) {
	t.Helper()
	for _, slice := range slices {
		poolEventsChange(t, ctx, tc, slice, false)
	}
	key := poolName{accountingDriver, accountingPool}
	want := metadataView(slices, key)
	c := tc.draManager.resourceClaimTracker
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
		return sameMetadata(publishRaceCommitted(c).pools[key], want), nil
	}); err != nil {
		t.Fatalf("metadata publication did not reach %+v: %v", want, err)
	}
}

func reconcileSlices(slices []*resourceapi.ResourceSlice, generation int64) []*resourceapi.ResourceSlice {
	out := make([]*resourceapi.ResourceSlice, len(slices))
	for i, slice := range slices {
		out[i] = slice.DeepCopy()
		out[i].Spec.Pool.Generation = generation
	}
	return out
}

func reconcileAwait(t *testing.T, ch <-chan struct{}, operation string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", operation)
	}
}

func reconcileSetup(t *testing.T, slices []*resourceapi.ResourceSlice, owners ...*resourceapi.ResourceClaim) (context.Context, *testContext) {
	t.Helper()
	// These cases isolate live reconciliation from the separately tested
	// startup ordering: establish owners only after metadata is coherent.
	ctx, tc := poolEventsSetup(t, slices)
	for _, owner := range owners {
		assumecache.AddTestObject(tc.draManager.resourceClaimTracker.cache, owner)
	}
	return ctx, tc
}

// A result computed against an obsolete view must wait for replay. In
// particular, a first owner has no historical lock against which to conflict.
func reconcileLateObservedMetadata(t *testing.T, initiallyConstrained bool) {
	t.Helper()
	slices := poolEventsSlices()
	initial := slices
	if !initiallyConstrained {
		devices := slices[1].DeepCopy()
		devices.Spec.Pool.ResourceSliceCount = 1
		initial = []*resourceapi.ResourceSlice{devices}
	}
	ctx, tc := reconcileSetup(t, initial)
	c := tc.draManager.resourceClaimTracker
	claim := sharingAffinityAccountingClaim("late-owner", "blue")
	if !initiallyConstrained {
		claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
	}
	device := sharingAffinityAccountingDevice(claim)
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var computations atomic.Int32
	c.allocatedDevices.observedAfterComputeHook = func() {
		if computations.Add(1) == 1 {
			close(entered)
			<-release
		}
	}
	go func() {
		defer close(done)
		assumecache.AddTestObject(c.cache, claim)
	}()
	reconcileAwait(t, entered, "first owner extraction")
	updated := reconcileSlices(slices, 2)
	updated[0].Spec.SharingAffinity[0].Expression = `"changed"`
	reconcilePublish(t, ctx, tc, updated)
	state := gatherSharingAffinityAccounting(t, c)
	if _, stale := state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}]; !stale {
		// Publication can let the reconciler supersede the paused handler.
		// A ready snapshot must then contain the fully reconciled new map.
		assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "changed"})
	}
	if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusUnreconstructable {
		t.Errorf("metadata advanced before first owner settled, but device was poisoned: %+v", got)
	}
	unblock()
	reconcileAwait(t, done, "first owner callback")
	state = reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "changed"})
	if initiallyConstrained {
		assertSharingAffinityAccountingCapacity(t, state, device, "5")
	} else if !state.AllocatedSharedDeviceIDs.Has(device) || len(state.AggregatedCapacity[device]) != 0 {
		t.Errorf("share-ID-only owner accounting changed during replay: %+v", state)
	}
	assumecache.DeleteTestObject(c.cache, claim)
	reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
}

func reconcileObservedAheadOfPublication(t *testing.T) {
	t.Helper()
	slices := poolEventsSlices()
	ctx, tc := reconcileSetup(t, slices)
	c := tc.draManager.resourceClaimTracker
	entered, release, published := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var publications atomic.Int32
	c.publishBeforeCommitHook = func(driver, pool string) {
		if driver == accountingDriver && pool == accountingPool && publications.Add(1) == 1 {
			close(entered)
			<-release
		}
	}
	changed := slices[0].DeepCopy()
	changed.Spec.SharingAffinity[0].Expression = `"changed"`
	go func() {
		defer close(published)
		poolEventsChange(t, ctx, tc, changed, false)
	}()
	reconcileAwait(t, entered, "metadata publication pause")
	claim := sharingAffinityAccountingClaim("ahead-owner", "blue")
	device := sharingAffinityAccountingDevice(claim)
	assumecache.AddTestObject(c.cache, claim)
	state := gatherSharingAffinityAccounting(t, c)
	if _, stale := state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}]; !stale {
		t.Error("lister-ahead owner committed without marking its pool stale")
	}
	if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusUnreconstructable {
		t.Errorf("lister-ahead provenance was treated as semantic ambiguity: %+v", got)
	}
	unblock()
	reconcileAwait(t, published, "metadata publication")
	reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "changed"})
	assumecache.DeleteTestObject(c.cache, claim)
	reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
}

func TestSharingAffinityReconciliation_AdminOnlyPools(t *testing.T) {
	for _, constrained := range []bool{true, false} {
		for _, tokenized := range []bool{true, false} {
			t.Run(fmt.Sprintf("constrained=%t/tokenized=%t", constrained, tokenized), func(t *testing.T) {
				setSharingAffinityAccountingGates(t, true, true)
				ctx := t.Context()
				c := newSharingAffinityAccountingTracker(t, true)
				a := c.allocatedDevices
				base := reconcileSlices(c.affinity.slices.(accountingSliceLister).slices, 1)
				if !constrained {
					base[0].Spec.SharingAffinity = nil
				}
				lister := &mutableSliceLister{slices: base}
				c.affinity.slices, c.metadataSlices = lister, lister
				c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
				a.metadata, a.tracker, a.claimsSynced = c.metadata, c, true
				c.cache.AddEventHandler(a.handlers())
				c.publishMetadata(true, nil)

				key := poolName{accountingDriver, accountingPool}
				poolID := schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}
				pending := sharingAffinityAccountingClaim("pending-admin", "blue")
				pending.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
				unallocated := pending.DeepCopy()
				unallocated.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unallocated)
				device := sharingAffinityAccountingDevice(pending)
				if tokenized {
					view := c.metadata.pools[key]
					err := c.ReserveClaimWithAffinity(ctx, affinityReservation{
						claimUID: pending.UID, claimRV: pending.ResourceVersion, claim: pending,
						token: "admin-reservation",
						provenance: []structured.DeviceAffinity{{
							ResultIndex: 0, Request: "req", Device: device,
							Pool:   sharingaffinity.PoolKey{Driver: key.driver, Name: key.pool, Generation: view.generation},
							Digest: view.digest, Constrained: constrained, Contributes: false,
							Values: map[string]string{"subnet": "blue"},
						}},
						fenceEpochs: map[poolName]uint64{key: view.epoch},
					})
					if err != nil {
						t.Fatal(err)
					}
				} else if err := c.SignalClaimPendingAllocation(pending.UID, pending); err != nil {
					t.Fatal(err)
				}
				observed := sharingAffinityAccountingClaim("observed-admin", "blue")
				observed.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
				assumecache.AddTestObject(c.cache, observed)
				owner := sharingAffinityAccountingClaim("shared-owner", "blue")
				assumecache.AddTestObject(c.cache, owner)

				// Reconcile a real owner across publication while both admin
				// allocations remain present. Neither may keep the pool stale.
				a.reconcile = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[poolName]())
				t.Cleanup(a.reconcile.ShutDown)
				updated := reconcileSlices(base, 2)
				lister.set(updated...)
				c.publishMetadata(false, affectedPools(updated[0], updated[1]))
				if constrained && !gatherSharingAffinityAccounting(t, c).DeviceSharingAffinity.StalePools.Has(poolID) {
					t.Fatal("publication did not make the shared owner's pool stale")
				}
				if a.reconcile.Len() != 1 || !c.processNext(ctx) {
					t.Fatal("metadata publication did not enqueue the pool for reconciliation")
				}
				state := gatherSharingAffinityAccounting(t, c)
				if state.DeviceSharingAffinity.StalePools.Has(poolID) {
					t.Fatalf("admin-only allocations kept the reconciled pool stale: %v", state.DeviceSharingAffinity.StalePools)
				}
				resolved, err := c.exclusionResolved(key)
				if err != nil || !resolved {
					t.Fatalf("reconciled pool remains excluded: resolved=%t, err=%v", resolved, err)
				}
				for _, id := range []ownerID{{uid: observed.UID}, {uid: pending.UID, inFlight: true}} {
					if len(a.ownerPools[id]) != 0 {
						t.Errorf("admin-only allocation has affinity owner pools: %v", a.ownerPools[id])
					}
				}
				if c.GetPendingAllocation(pending.UID) == nil {
					t.Error("reconciliation removed the in-flight admin allocation")
				}
				assertSharingAffinityAccountingCapacity(t, state, device, "5")
				status, values := structured.AffinityStatusClean, map[string]string(nil)
				if constrained {
					status, values = structured.AffinityStatusLocked, map[string]string{"subnet": "blue"}
				}
				assertSharingAffinityAccountingState(t, state, device, status, values)
			})
		}
	}
}

func TestSharingAffinityReconciliation_PublishedEpochFencesReaders(t *testing.T) {
	for _, inFlight := range []bool{false, true} {
		for _, mutation := range []string{"same-map", "changed-map", "removed"} {
			t.Run(fmt.Sprintf("in-flight=%t/%s", inFlight, mutation), func(t *testing.T) {
				setSharingAffinityAccountingGates(t, true, true)
				ctx := t.Context()
				c := newSharingAffinityAccountingTracker(t, true)
				a := c.allocatedDevices
				base := c.affinity.slices.(accountingSliceLister).slices
				lister := &mutableSliceLister{slices: base}
				c.affinity.slices, c.metadataSlices = lister, lister
				c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
				a.metadata, a.tracker, a.claimsSynced = c.metadata, c, true
				c.cache.AddEventHandler(a.handlers())
				c.publishMetadata(true, nil)

				owner := sharingAffinityAccountingClaim("owner", "blue")
				device := sharingAffinityAccountingDevice(owner)
				if inFlight {
					unallocated := owner.DeepCopy()
					unallocated.Status.Allocation = nil
					assumecache.AddTestObject(c.cache, unallocated)
					if err := c.SignalClaimPendingAllocation(owner.UID, owner); err != nil {
						t.Fatal(err)
					}
				} else {
					assumecache.AddTestObject(c.cache, owner)
				}
				before := gatherSharingAffinityAccounting(t, c)
				assertSharingAffinityAccountingState(t, before, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				if len(before.DeviceSharingAffinity.StalePools) != 0 {
					t.Fatalf("initial owner is not ready: %v", before.DeviceSharingAffinity.StalePools)
				}
				candidate := sharingAffinityAccountingClaim("candidate", "blue")
				unallocated := candidate.DeepCopy()
				unallocated.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, unallocated)

				// Drive the real worker explicitly so publication is observed
				// before beginPool can even mark the owner pending.
				a.reconcile = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[poolName]())
				t.Cleanup(a.reconcile.ShutDown)
				updated := reconcileSlices(base, 2)
				switch mutation {
				case "same-map":
					updated[0].Spec.SharingAffinity[0].Expression = `has(parameters.network) ? parameters.network : parameters.subnet`
				case "changed-map":
					updated[0].Spec.SharingAffinity[0].Expression = `"red"`
				case "removed":
					updated[0].Spec.SharingAffinity = nil
				}
				lister.set(updated...)
				key := poolName{accountingDriver, accountingPool}
				oldEpoch := c.metadata.pools[key].epoch
				c.publishMetadata(false, affectedPools(updated[0], updated[1]))
				view := c.metadata.pools[key]
				rec := a.owners[ownerPoolKey{uid: owner.UID, inFlight: inFlight, pool: key}]
				if view.epoch <= oldEpoch || rec == nil || rec.pending || rec.epoch == view.epoch {
					t.Fatalf("publication must invalidate a settled owner by epoch alone: view=%+v owner=%+v", view, rec)
				}

				checkReaders := func(pending bool) {
					t.Helper()
					a.mutex.RLock()
					ready := a.poolReadyLocked(key)
					a.mutex.RUnlock()
					if ready == pending {
						t.Errorf("pool ready=%t, want %t", ready, !pending)
					}
					resolved, err := c.exclusionResolved(key)
					if err != nil || resolved == pending {
						t.Errorf("exclusion resolved=%t, err=%v, want %t", resolved, err, !pending)
					}
					checkState := func(state *structured.AllocatedState) {
						t.Helper()
						if state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) != pending {
							t.Errorf("StalePools=%v, want pending=%t", state.DeviceSharingAffinity.StalePools, pending)
						}
						wantStatus := structured.AffinityStatusLocked
						values := map[string]string{"subnet": "blue"}
						if !pending && mutation != "same-map" {
							wantStatus, values = structured.AffinityStatusUnreconstructable, nil
						}
						assertSharingAffinityAccountingState(t, state, device, wantStatus, values)
						if !reflect.DeepEqual(state.AggregatedCapacity, before.AggregatedCapacity) ||
							!state.AllocatedDevices.Equal(before.AllocatedDevices) ||
							!state.AllocatedSharedDeviceIDs.Equal(before.AllocatedSharedDeviceIDs) {
							t.Error("metadata reconciliation changed allocation accounting")
						}
					}
					// The public gather must use the affinity-aware path,
					// not the legacy path that omits StalePools.
					checkState(gatherSharingAffinityAccounting(t, c))
					state, metadata, err := c.gatherAllocatedStateWithMetadata()
					if err != nil || !metadata.coherent(updated) {
						t.Fatalf("gather with published metadata: metadata=%+v err=%v", metadata, err)
					}
					checkState(state)
					snapshot, err := c.gatherAllocatedStateForSlices(updated)
					if err != nil {
						t.Fatal(err)
					}
					checkState(snapshot.state)
					if snapshot.excluded.Has(key) != pending {
						t.Errorf("cycle exclusions=%v, want pending=%t", snapshot.excluded, pending)
					}
					if _, captured := snapshot.captured[key]; captured == pending {
						t.Errorf("captured Reserve provenance=%t, want %t", captured, !pending)
					}
				}
				checkReaders(true)

				entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
				unblock := sync.OnceFunc(func() { close(release) })
				defer func() {
					unblock()
					reconcileAwait(t, done, "worker shutdown")
				}()
				c.reconcileAfterComputeHook = func(poolName, types.UID) {
					close(entered)
					<-release
				}
				go func() {
					defer close(done)
					c.processNext(ctx)
				}()
				reconcileAwait(t, entered, "owner computation before commit")
				checkReaders(true)

				// Even a fresh metadata epoch must not let Reserve consume
				// the old lock while its bound owner is being re-extracted.
				constrained := mutation != "removed"
				err := c.commitAffinityReservation(affinityReservation{
					claimUID: candidate.UID, claimRV: candidate.ResourceVersion, claim: candidate,
					token: "pending-reconciliation",
					provenance: []structured.DeviceAffinity{{
						ResultIndex: 0, Request: "req", Device: device,
						Pool:   sharingaffinity.PoolKey{Driver: key.driver, Name: key.pool, Generation: view.generation},
						Digest: view.digest, Constrained: constrained, Contributes: constrained,
						Values: map[string]string{"subnet": "blue"},
					}},
					fenceEpochs: map[poolName]uint64{key: view.epoch},
				}, map[poolName]poolMeta{key: view})
				if !errors.Is(err, errAffinityNotReady) {
					t.Fatalf("Reserve during reconciliation=%v, want not-ready", err)
				}
				if c.GetPendingAllocation(candidate.UID) != nil {
					t.Fatal("not-ready Reserve published an allocation")
				}
				unblock()
				reconcileAwait(t, done, "owner reconciliation commit")
				checkReaders(false)
			})
		}
	}
}

// Use the real manager's publication callbacks, not a
// test-side replay of claims that would conceal a missing reconciler.
func TestSharingAffinityReconciliation(t *testing.T) {
	blue := map[string]string{"subnet": "blue"}
	for _, sameGeneration := range []bool{false, true} {
		t.Run(fmt.Sprintf("same-map-dual-path-expression/same-generation=%t", sameGeneration), func(t *testing.T) {
			slices := poolEventsSlices()
			a := sharingAffinityAccountingClaim("owner-a", "blue")
			b := sharingAffinityAccountingClaim("owner-b", "blue")
			ctx, tc := reconcileSetup(t, slices, a, b)
			c := tc.draManager.resourceClaimTracker
			device := sharingAffinityAccountingDevice(a)
			before := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			generation := int64(2)
			if sameGeneration {
				generation = 1
			}
			updated := reconcileSlices(slices, generation)
			updated[0].Spec.SharingAffinity[0].Expression = `has(parameters.network) ? parameters.network : parameters.subnet`
			reconcilePublish(t, ctx, tc, updated)
			after := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			assertSharingAffinityAccountingCapacity(t, after, device, "10")
			assertSharingAffinityAccountingState(t, before, device, structured.AffinityStatusLocked, blue)
			// Replay adds no claim API mutation or eviction.
			for _, claim := range []*resourceapi.ResourceClaim{a, b} {
				got, err := c.Get(claim.Namespace, claim.Name)
				if err != nil || !reflect.DeepEqual(got, claim) {
					t.Errorf("bound owner changed during reconciliation: got=%+v err=%v want=%+v", got, err, claim)
				}
			}
		})
	}

	changes := []struct {
		name   string
		mutate func(*resourceapi.ResourceSlice)
	}{
		{"changed-value", func(s *resourceapi.ResourceSlice) { s.Spec.SharingAffinity[0].Expression = `"red"` }},
		{"renamed-key", func(s *resourceapi.ResourceSlice) { s.Spec.SharingAffinity[0].Name = "network" }},
		{"added-extractor", func(s *resourceapi.ResourceSlice) {
			second := s.Spec.SharingAffinity[0]
			second.Name = "tenant"
			s.Spec.SharingAffinity = append(s.Spec.SharingAffinity, second)
		}},
		{"removed-affinity", func(s *resourceapi.ResourceSlice) { s.Spec.SharingAffinity = nil }},
	}
	for _, change := range changes {
		t.Run(change.name, func(t *testing.T) {
			slices := poolEventsSlices()
			a := sharingAffinityAccountingClaim("owner-a", "blue")
			b := sharingAffinityAccountingClaim("owner-b", "blue")
			ctx, tc := reconcileSetup(t, slices, a, b)
			c := tc.draManager.resourceClaimTracker
			device := sharingAffinityAccountingDevice(a)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			updated := reconcileSlices(slices, 2)
			change.mutate(updated[0])
			reconcilePublish(t, ctx, tc, updated)
			state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, state, device, "10")
			// A compatible republish cannot erase established ambiguity.
			reconcilePublish(t, ctx, tc, reconcileSlices(slices, 3))
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
			assumecache.DeleteTestObject(c.cache, a)
			state = reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, state, device, "5")
			assumecache.DeleteTestObject(c.cache, b)
			state = reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
			if state.AllocatedSharedDeviceIDs.Has(device) {
				t.Error("fully drained device retains a shared owner")
			}
			next := sharingAffinityAccountingClaim("new-owner", "green")
			assumecache.AddTestObject(c.cache, next)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "green"})
		})
	}

	t.Run("removed-one-of-two-extractors", func(t *testing.T) {
		slices := poolEventsSlices()
		second := slices[0].Spec.SharingAffinity[0]
		second.Name = "tenant"
		slices[0].Spec.SharingAffinity = append(slices[0].Spec.SharingAffinity, second)
		a := sharingAffinityAccountingClaim("owner", "blue")
		ctx, tc := reconcileSetup(t, slices, a)
		c := tc.draManager.resourceClaimTracker
		device := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue", "tenant": "blue"})
		updated := reconcileSlices(slices, 2)
		updated[0].Spec.SharingAffinity = updated[0].Spec.SharingAffinity[:1]
		reconcilePublish(t, ctx, tc, updated)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
	})

	for _, changed := range []bool{false, true} {
		t.Run(fmt.Sprintf("locked-incomplete-complete/changed=%t", changed), func(t *testing.T) {
			slices := poolEventsSlices()
			a := sharingAffinityAccountingClaim("owner", "blue")
			ctx, tc := reconcileSetup(t, slices, a)
			c := tc.draManager.resourceClaimTracker
			device := sharingAffinityAccountingDevice(a)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			poolEventsChange(t, ctx, tc, slices[0], true)
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
				return !publishRaceCommitted(c).pools[poolName{accountingDriver, accountingPool}].complete, nil
			}); err != nil {
				t.Fatalf("incomplete metadata not published: %v", err)
			}
			state := gatherSharingAffinityAccounting(t, c)
			if _, stale := state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}]; !stale {
				t.Error("incomplete active pool is not in StalePools")
			}
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, blue)
			c.allocatedDevices.mutex.RLock()
			fenced := c.allocatedDevices.fenced.Has(device)
			c.allocatedDevices.mutex.RUnlock()
			if fenced {
				t.Error("transient metadata loss established a sticky fence")
			}
			restored := reconcileSlices(slices, 1)
			want, values := structured.AffinityStatusLocked, blue
			if changed {
				restored[0].Spec.SharingAffinity[0].Expression = `"red"`
				want, values = structured.AffinityStatusUnreconstructable, nil
			}
			reconcilePublish(t, ctx, tc, restored)
			reconcileWaitState(t, ctx, c, device, want, values)
		})
	}

	t.Run("allocation-updated-same-uid", func(t *testing.T) {
		slices := poolEventsSlices()
		second := *slices[1].Spec.Devices[0].DeepCopy()
		second.Name = "second"
		slices[1].Spec.Devices = append(slices[1].Spec.Devices, second)
		a := sharingAffinityAccountingClaim("owner", "blue")
		ctx, tc := reconcileSetup(t, slices, a)
		c := tc.draManager.resourceClaimTracker
		oldDevice := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, oldDevice, structured.AffinityStatusLocked, blue)
		updated := a.DeepCopy()
		updated.ResourceVersion = "2"
		updated.Status.Allocation.Devices.Results[0].Device = "second"
		updated.Status.Allocation.Devices.Results[0].ConsumedCapacity["units"] = resource.MustParse("7")
		assumecache.UpdateTestObject(c.cache, updated)
		state := reconcileWaitState(t, ctx, c, sharingAffinityAccountingDevice(updated), structured.AffinityStatusLocked, blue)
		assertSharingAffinityAccountingState(t, state, oldDevice, structured.AffinityStatusClean, nil)
		assertSharingAffinityAccountingCapacity(t, state, sharingAffinityAccountingDevice(updated), "7")
		if state.AllocatedSharedDeviceIDs.Has(oldDevice) || len(state.AggregatedCapacity[oldDevice]) != 0 {
			t.Errorf("old allocation still accounted after same-UID update: %+v", state)
		}
	})

	t.Run("update-fastpath-spec-config-change", func(t *testing.T) {
		a := sharingAffinityAccountingClaim("owner", "blue")
		ctx, tc := reconcileSetup(t, poolEventsSlices(), a)
		c := tc.draManager.resourceClaimTracker
		device := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
		updated := a.DeepCopy()
		updated.ResourceVersion = "2"
		updated.Spec.Devices.Config = sharingAffinityAccountingClaim("unused", "red").Spec.Devices.Config
		assumecache.UpdateTestObject(c.cache, updated)
		state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		assertSharingAffinityAccountingCapacity(t, state, device, "5")
	})

	t.Run("same-uid-rv-only-update", func(t *testing.T) {
		a := sharingAffinityAccountingClaim("owner", "blue")
		ctx, tc := reconcileSetup(t, poolEventsSlices(), a)
		c := tc.draManager.resourceClaimTracker
		device := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
		before := c.allocatedDevices.Revision()
		updated := a.DeepCopy()
		updated.ResourceVersion = "2"
		assumecache.UpdateTestObject(c.cache, updated)
		if got := c.allocatedDevices.Revision(); got != before {
			t.Errorf("RV-only update changed accounting revision: got=%d want=%d", got, before)
		}
	})

	t.Run("unchanged-device-update-while-fenced", func(t *testing.T) {
		a := sharingAffinityAccountingClaim("blue-owner", "blue")
		b := sharingAffinityAccountingClaim("red-owner", "red")
		ctx, tc := reconcileSetup(t, poolEventsSlices(), a, b)
		c := tc.draManager.resourceClaimTracker
		device := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		updated := b.DeepCopy()
		updated.ResourceVersion = "2"
		updated.Spec.Devices.Config = a.Spec.Devices.Config
		updated.Status.Allocation.Devices.Results[0].ConsumedCapacity["units"] = resource.MustParse("7")
		assumecache.UpdateTestObject(c.cache, updated)
		state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		assertSharingAffinityAccountingCapacity(t, state, device, "12")
		assumecache.DeleteTestObject(c.cache, a)
		state = reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		assertSharingAffinityAccountingCapacity(t, state, device, "7")
		assumecache.DeleteTestObject(c.cache, updated)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
	})

	t.Run("nil-capacity-owner-affinity-removed", func(t *testing.T) {
		a := sharingAffinityAccountingClaim("capacity-owner", "blue")
		b := sharingAffinityAccountingClaim("share-only-owner", "blue")
		b.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
		slices := poolEventsSlices()
		ctx, tc := reconcileSetup(t, slices, a, b)
		c := tc.draManager.resourceClaimTracker
		device := sharingAffinityAccountingDevice(a)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
		updated := reconcileSlices(slices, 2)
		updated[0].Spec.SharingAffinity = nil
		reconcilePublish(t, ctx, tc, updated)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		reconcilePublish(t, ctx, tc, reconcileSlices(slices, 3))
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		assumecache.DeleteTestObject(c.cache, a)
		state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusUnreconstructable, nil)
		if !state.AllocatedSharedDeviceIDs.Has(device) || len(state.AggregatedCapacity[device]) != 0 {
			t.Errorf("share-ID-only owner missing or capacity retained after capacity owner drained: %+v", state)
		}
		assumecache.DeleteTestObject(c.cache, b)
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
	})

	for _, operation := range []string{"delete", "recreate-new-uid", "equal-rv-assume"} {
		t.Run("stale-observed-commit/"+operation, func(t *testing.T) {
			anchor := sharingAffinityAccountingClaim("anchor", "blue")
			ctx, tc := reconcileSetup(t, poolEventsSlices(), anchor)
			c := tc.draManager.resourceClaimTracker
			device := sharingAffinityAccountingDevice(anchor)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			stale := sharingAffinityAccountingClaim("stale", "red")
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			t.Cleanup(unblock)
			var calls atomic.Int32
			c.allocatedDevices.observedAfterComputeHook = func() {
				if calls.Add(1) == 1 {
					close(entered)
					<-release
				}
			}
			var intermediate *structured.AllocatedState
			// This callback runs after the stale OnAdd but before the queued
			// replacement/delete. It catches poison later hidden by drain.
			c.cache.AddEventHandler(cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					if obj.(*resourceapi.ResourceClaim).UID == stale.UID {
						intermediate, _ = c.GatherAllocatedState()
					}
				},
			})
			go func() {
				defer close(done)
				assumecache.AddTestObject(c.cache, stale)
			}()
			reconcileAwait(t, entered, "observed computation")
			replacement := stale.DeepCopy()
			replacement.Spec.Devices.Config = anchor.Spec.Devices.Config
			var mutations sync.WaitGroup
			waitCached := func(want *resourceapi.ResourceClaim) {
				t.Helper()
				if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
					got, err := c.Get(stale.Namespace, stale.Name)
					if want == nil {
						return errors.Is(err, assumecache.ErrNotFound), nil
					}
					return err == nil && reflect.DeepEqual(got, want), nil
				}); err != nil {
					t.Fatalf("cache mutation not visible while callback paused: %v", err)
				}
			}
			switch operation {
			case "delete":
				mutations.Go(func() { assumecache.DeleteTestObject(c.cache, stale) })
				waitCached(nil)
			case "recreate-new-uid":
				mutations.Go(func() { assumecache.DeleteTestObject(c.cache, stale) })
				waitCached(nil)
				replacement.UID = types.UID("replacement")
				mutations.Go(func() { assumecache.AddTestObject(c.cache, replacement) })
				waitCached(replacement)
			case "equal-rv-assume":
				mutations.Go(func() {
					if err := c.cache.Assume(replacement); err != nil {
						t.Errorf("Assume equal-RV changed spec: %v", err)
					}
				})
				waitCached(replacement)
			}
			unblock()
			reconcileAwait(t, done, "queued claim callbacks")
			mutations.Wait()
			if intermediate == nil {
				t.Fatal("post-compute snapshot callback did not run")
			}
			got := intermediate.DeviceSharingAffinity.DeviceStates[device]
			if got.Status == structured.AffinityStatusUnreconstructable || got.Status == structured.AffinityStatusLocked && !maps.Equal(got.LockedAffinity, blue) {
				t.Errorf("stale computation poisoned established owner before queued event: %+v", got)
			}
			state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			wantCapacity := "10"
			if operation == "delete" {
				wantCapacity = "5"
			}
			assertSharingAffinityAccountingCapacity(t, state, device, wantCapacity)
		})
	}

	for _, equal := range []bool{true, false} {
		t.Run(fmt.Sprintf("handoff-metadata-reconciliation/equal-map=%t", equal), func(t *testing.T) {
			ctx, tc := poolEventsSetup(t, poolEventsSlices())
			c := tc.draManager.resourceClaimTracker
			a := sharingAffinityAccountingClaim("handoff-owner", "blue")
			signalSharingAffinityAccounting(t, c, a)
			assumecache.UpdateTestObject(c.cache, a)
			device := sharingAffinityAccountingDevice(a)
			state := reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
			assertSharingAffinityAccountingCapacity(t, state, device, "10")
			updated := reconcileSlices(poolEventsSlices(), 2)
			want, values := structured.AffinityStatusLocked, blue
			if equal {
				updated[0].Spec.SharingAffinity[0].Expression = `has(parameters.network) ? parameters.network : parameters.subnet`
			} else {
				updated[0].Spec.SharingAffinity[0].Expression = `"red"`
				want, values = structured.AffinityStatusUnreconstructable, nil
			}
			reconcilePublish(t, ctx, tc, updated)
			state = reconcileWaitState(t, ctx, c, device, want, values)
			assertSharingAffinityAccountingCapacity(t, state, device, "10")
			if !c.MaybeRemoveClaimPendingAllocation(a.UID, true) {
				t.Error("in-flight handoff owner was not removed")
			}
			state = reconcileWaitState(t, ctx, c, device, want, values)
			assertSharingAffinityAccountingCapacity(t, state, device, "5")
			assumecache.DeleteTestObject(c.cache, a)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
		})
	}
}
