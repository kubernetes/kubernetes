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
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// gateHandler is registered before the tracker's handlers so that, while it
// blocks, the AssumeCache store is already ahead and the tracker's callbacks
// remain queued in the FIFO.
type gateHandler struct {
	armed   bool
	entered chan struct{}
	release chan struct{}
}

func (g *gateHandler) block() {
	if !g.armed {
		return
	}
	g.armed = false
	close(g.entered)
	<-g.release
}
func (g *gateHandler) OnAdd(any, bool)   { g.block() }
func (g *gateHandler) OnUpdate(any, any) { g.block() }
func (g *gateHandler) OnDelete(any)      { g.block() }

var _ cache.ResourceEventHandler = &gateHandler{}

// The worker computes, then the cache moves while tracker
// callbacks stay queued. Commit must see the cache via Get and stay pending
// (cacheAhead/cacheGone) without contributing the stale computation.
func TestSharingAffinityReconciliation_QueuedCacheCallbacks(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	blue := map[string]string{"subnet": "blue"}
	cases := []struct {
		name       string
		mutate     func(c *claimTracker, owner *resourceapi.ResourceClaim)
		wantReason string
		wantUID    types.UID
	}{
		{
			name: "allocation-changing-update",
			mutate: func(c *claimTracker, owner *resourceapi.ResourceClaim) {
				upd := sharingAffinityAccountingClaim(owner.UID, "blue")
				upd.ResourceVersion = "2"
				upd.Status.Allocation.Devices.Results[0].ConsumedCapacity["units"] = resource.MustParse("3")
				assumecache.UpdateTestObject(c.cache, upd)
			},
			wantReason: "cacheAhead", wantUID: "owner",
		},
		{
			name:       "delete",
			mutate:     func(c *claimTracker, owner *resourceapi.ResourceClaim) { assumecache.DeleteTestObject(c.cache, owner) },
			wantReason: "cacheGone",
		},
		{
			name: "recreation-new-uid",
			mutate: func(c *claimTracker, owner *resourceapi.ResourceClaim) {
				// Delete+re-add collapsed by a relist: the store jumps straight
				// to a same-name, new-UID object.
				n := sharingAffinityAccountingClaim("owner-new", "blue")
				n.Name, n.ResourceVersion = owner.Name, "2"
				assumecache.AddTestObject(c.cache, n)
			},
			wantReason: "cacheGone", wantUID: "owner-new",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			base := poolEventsSlices()
			c, lister := newInterleaveTracker(t, base)
			a := c.allocatedDevices
			gate := &gateHandler{entered: make(chan struct{}), release: make(chan struct{})}
			c.cache.AddEventHandler(gate)
			c.cache.AddEventHandler(a.handlers())
			owner := sharingAffinityAccountingClaim("owner", "blue")
			device := sharingAffinityAccountingDevice(owner)
			assumecache.AddTestObject(c.cache, owner)
			interleaveDrain(t, ctx, c)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, blue)

			updated := reconcileSlices(base, 2)
			lister.set(updated...)
			c.publishMetadata(false, affectedPools(updated[0]))
			a.reconcile.Add(key)

			done := make(chan struct{})
			c.reconcileAfterComputeHook = func(poolName, types.UID) {
				c.reconcileAfterComputeHook = nil
				gate.armed = true
				go func() { defer close(done); tc.mutate(c, owner) }()
				select {
				case <-gate.entered:
				case <-time.After(10 * time.Second):
					t.Fatal("cache event never reached gate")
				}
				// Store is ahead; the tracker callback is still queued.
				obj, err := c.cache.Get(owner.Namespace + "/" + owner.Name)
				if tc.wantUID == "" {
					if err == nil {
						t.Errorf("Get returned %v, want not found", obj)
					}
				} else if err != nil || obj.(*resourceapi.ResourceClaim).UID != tc.wantUID {
					t.Errorf("Get=%v err=%v, want UID %s", obj, err, tc.wantUID)
				}
			}
			c.processNext(ctx)
			rec, ok := interleaveRecord(c, ownerPoolKey{uid: owner.UID, pool: key})
			if !ok || !rec.pending || rec.reason != tc.wantReason {
				t.Errorf("record after commit = %+v present=%t, want pending %s", rec, ok, tc.wantReason)
			}
			if interleaveReady(c, key) {
				t.Error("pool ready before queued callbacks delivered")
			}
			if n := a.reconcile.NumRequeues(key); n != 0 {
				t.Errorf("requeues=%d, want 0", n)
			}
			state := gatherSharingAffinityAccounting(t, c)
			if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}) {
				t.Errorf("pool not stale while pending: %v", state.DeviceSharingAffinity.StalePools)
			}

			close(gate.release)
			<-done
			interleaveDrain(t, ctx, c)
			interleaveAllCoherent(t, c)
			if !interleaveReady(c, key) {
				t.Error("pool not ready after callbacks delivered")
			}
			a.mutex.RLock()
			_, oldPresent := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
			_, newPresent := a.owners[ownerPoolKey{uid: tc.wantUID, pool: key}]
			a.mutex.RUnlock()
			if tc.wantUID != owner.UID && oldPresent {
				t.Error("stale owner record survived")
			}
			state = gatherSharingAffinityAccounting(t, c)
			if tc.wantUID == "" {
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusClean, nil)
				return
			}
			if !newPresent {
				t.Errorf("owner %s missing", tc.wantUID)
			}
			assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, blue)
			if tc.name == "allocation-changing-update" {
				// Old 5-unit contribution must be replaced, not added to.
				assertSharingAffinityAccountingCapacity(t, state, device, "3")
			}
		})
	}
}

// Equal-RV Assume (changed config, same allocation) and its
// Restore both stay queued behind the gate while the worker commits. The
// stale assumed computation must never install contributions or establish
// sticky ambiguity against the compatible API owner and sibling.
func TestSharingAffinityReconciliation_QueuedAssumeRestoreNoPoison(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	blue := map[string]string{"subnet": "blue"}
	ctx := t.Context()
	base := poolEventsSlices()
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	_ = blue
	gate := &gateHandler{entered: make(chan struct{}), release: make(chan struct{})}
	c.cache.AddEventHandler(gate)
	c.cache.AddEventHandler(a.handlers())
	owner := sharingAffinityAccountingClaim("owner", "blue")
	sibling := sharingAffinityAccountingClaim("sibling", "blue")
	device := sharingAffinityAccountingDevice(owner)
	assumecache.AddTestObject(c.cache, owner)
	assumecache.AddTestObject(c.cache, sibling)
	interleaveDrain(t, ctx, c)
	state := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, blue)
	assertSharingAffinityAccountingCapacity(t, state, device, "10")

	updated := reconcileSlices(base, 2)
	lister.set(updated...)
	c.publishMetadata(false, affectedPools(updated[0]))
	a.reconcile.Add(key)

	done := make(chan struct{})
	c.reconcileAfterComputeHook = func(poolName, types.UID) {
		c.reconcileAfterComputeHook = nil
		gate.armed = true
		go func() {
			defer close(done)
			assumed := sharingAffinityAccountingClaim(owner.UID, "red")
			if err := c.cache.Assume(assumed); err != nil {
				t.Errorf("Assume: %v", err)
			}
			c.cache.Restore(owner.Namespace + "/" + owner.Name)
		}()
		select {
		case <-gate.entered:
		case <-time.After(10 * time.Second):
			t.Fatal("cache event never reached gate")
		}
	}
	c.processNext(ctx)
	// Intermediate: whatever the commit decided, no red contribution or
	// sticky fence may be visible while the callbacks are still queued.
	state = gatherSharingAffinityAccounting(t, c)
	if got, ok := state.DeviceSharingAffinity.DeviceStates[device]; ok && (got.Status == structured.AffinityStatusUnreconstructable || got.LockedAffinity["subnet"] == "red") {
		t.Errorf("queued assumed config poisoned state: %#v", got)
	}
	if n := a.reconcile.NumRequeues(key); n != 0 {
		t.Errorf("requeues=%d, want 0", n)
	}

	close(gate.release)
	<-done
	interleaveDrain(t, ctx, c)
	interleaveAllCoherent(t, c)
	if !interleaveReady(c, key) {
		t.Error("pool not ready after callbacks delivered")
	}
	state = gatherSharingAffinityAccounting(t, c)
	// Once delivered, the Assume itself is an observed source change and
	// fences per design; the stale worker result must never have locked red
	// or altered capacity/ownership.
	if got := state.DeviceSharingAffinity.DeviceStates[device]; got.LockedAffinity["subnet"] == "red" {
		t.Errorf("assumed red config locked: %#v", got)
	}
	assertSharingAffinityAccountingCapacity(t, state, device, "10")
	a.mutex.RLock()
	_, o1 := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
	_, o2 := a.owners[ownerPoolKey{uid: sibling.UID, pool: key}]
	a.mutex.RUnlock()
	if !o1 || !o2 {
		t.Errorf("owner records present owner=%t sibling=%t, want both", o1, o2)
	}
}

// Equal-RV Restore while the worker is paused on a computation
// against the assumed (changed) config must never lock the assumed value.
func TestSharingAffinityReconciliation_RestoreDuringPausedWorker(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	ctx := t.Context()
	base := poolEventsSlices()
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	c.cache.AddEventHandler(a.handlers())
	owner := sharingAffinityAccountingClaim("owner", "blue")
	device := sharingAffinityAccountingDevice(owner)
	assumecache.AddTestObject(c.cache, owner)
	interleaveDrain(t, ctx, c)

	assumed := sharingAffinityAccountingClaim(owner.UID, "red")
	if err := c.cache.Assume(assumed); err != nil {
		t.Fatal(err)
	}
	updated := reconcileSlices(base, 2)
	lister.set(updated...)
	c.publishMetadata(false, affectedPools(updated[0]))
	a.reconcile.Add(key)
	paused := false
	c.reconcileAfterComputeHook = func(poolName, types.UID) {
		c.reconcileAfterComputeHook = nil
		paused = true
		c.cache.Restore(owner.Namespace + "/" + owner.Name)
	}
	interleaveDrain(t, ctx, c)
	if !paused {
		t.Fatal("worker did not compute")
	}
	interleaveAllCoherent(t, c)
	state := gatherSharingAffinityAccounting(t, c)
	// The assumed red config must never become a lock; the changed source
	// fences the established blue lock (same as the post-settlement case).
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
	if !interleaveReady(c, key) {
		t.Error("pool not ready after settlement")
	}
}
