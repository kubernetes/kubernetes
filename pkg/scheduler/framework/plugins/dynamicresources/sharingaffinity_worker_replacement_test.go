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
	"maps"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// The queued worker computes an incompatible (red) assumed owner
// while a delete or compatible same-name new UID becomes cache-visible before
// commit. The stale red result must not install contribution, established
// lock or ambiguity; the compatible sibling/new owner end up locked blue.
func TestSharingAffinityReconciliation_WorkerDeleteOrNewUIDRejectsIncompatibleResult(t *testing.T) {
	for _, tc := range []struct {
		name   string
		newUID types.UID
	}{{name: "delete"}, {name: "recreation-new-uid", newUID: "owner-new"}} {
		t.Run(tc.name, func(t *testing.T) { testWorkerReplacementRejectsIncompatible(t, tc.newUID) })
	}
}

func testWorkerReplacementRejectsIncompatible(t *testing.T, newUID types.UID) {
	key := poolName{accountingDriver, accountingPool}
	blue := map[string]string{"subnet": "blue"}
	red := map[string]string{"subnet": "red"}
	ctx := t.Context()
	base := poolEventsSlices()
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	c.cache.AddEventHandler(a.handlers())
	owner := sharingAffinityAccountingClaim("owner", "blue")
	sibling := sharingAffinityAccountingClaim("sibling", "blue")
	device := sharingAffinityAccountingDevice(owner)
	assumecache.AddTestObject(c.cache, owner)
	assumecache.AddTestObject(c.cache, sibling)
	interleaveDrain(t, ctx, c)
	before := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingState(t, before, device, structured.AffinityStatusLocked, blue)
	assertSharingAffinityAccountingCapacity(t, before, device, "10")

	// Hold the observed red computation so red is never committed by it.
	entered, release, assumedDone := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var observed atomic.Int32
	a.observedAfterComputeHook = func() {
		if observed.Add(1) == 1 {
			close(entered)
			<-release
		}
	}
	t.Cleanup(func() { a.observedAfterComputeHook = nil })
	go func() {
		defer close(assumedDone)
		if err := c.cache.Assume(sharingAffinityAccountingClaim(owner.UID, "red")); err != nil {
			t.Errorf("Assume: %v", err)
		}
	}()
	reconcileAwait(t, entered, "observed assumed computation")

	updated := reconcileSlices(base, 2)
	lister.set(updated...)
	c.publishMetadata(false, affectedPools(updated[0]))
	a.reconcile.Add(key)

	restoredDone := make(chan struct{})
	var workerSawRed, restored bool
	c.reconcileAfterComputeHook = func(_ poolName, uid types.UID) {
		if uid != owner.UID || restored {
			return
		}
		a.mutex.RLock()
		rec := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
		a.mutex.RUnlock()
		workerSawRed = rec.claim != nil && maps.Equal(extractClaimSubnet(t, rec.claim), red)
		restored = true
		// Equal-RV Restore of the compatible API object becomes cache-visible
		// before commit; its callback queues behind the paused observed one.
		// Delete (or same-name compatible new UID) becomes cache-visible
		// before commit; tracker callbacks queue behind the paused observed one.
		go func() {
			defer close(restoredDone)
			if newUID == "" {
				assumecache.DeleteTestObject(c.cache, owner)
				return
			}
			n := sharingAffinityAccountingClaim(newUID, "blue")
			n.Name, n.ResourceVersion = owner.Name, "2"
			assumecache.AddTestObject(c.cache, n)
		}()
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			obj, err := c.cache.Get(owner.Namespace + "/" + owner.Name)
			if newUID == "" {
				return err != nil, nil
			}
			if err != nil {
				return false, nil
			}
			cl := obj.(*resourceapi.ResourceClaim)
			return cl.UID == newUID && maps.Equal(extractClaimSubnet(t, cl), blue), nil
		}); err != nil {
			t.Errorf("replacement not cache-visible before worker commit: %v", err)
		}
	}
	t.Cleanup(func() { c.reconcileAfterComputeHook = nil })
	c.processNext(ctx)
	if !restored {
		t.Fatal("worker never computed the owner")
	}
	if !workerSawRed {
		t.Fatal("worker did not compute the uncommitted assumed red owner")
	}
	mid := gatherSharingAffinityAccounting(t, c)
	if got, ok := mid.DeviceSharingAffinity.DeviceStates[device]; ok && (got.Status == structured.AffinityStatusUnreconstructable || got.LockedAffinity["subnet"] == "red") {
		t.Fatalf("rejected worker result poisoned state: %#v", got)
	}
	a.mutex.RLock()
	midRec := a.affinityDevices[device]
	midOwner := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
	a.mutex.RUnlock()
	if midRec != nil && (midRec.ambiguous || midRec.established["subnet"] == "red") {
		t.Fatalf("stale red contribution installed: %+v", midRec)
	}
	if midOwner.claim != nil && !midOwner.pending && maps.Equal(extractClaimSubnet(t, midOwner.claim), red) {
		t.Fatalf("stale red owner contribution settled: %+v", midOwner)
	}
	if n := a.reconcile.NumRequeues(key); n != 0 {
		t.Errorf("requeues=%d, want 0", n)
	}

	unblock()
	reconcileAwait(t, assumedDone, "stale observed assumed callback")
	reconcileAwait(t, restoredDone, "replacement callback")
	interleaveDrain(t, ctx, c)
	interleaveAllCoherent(t, c)
	if !interleaveReady(c, key) {
		t.Error("pool not ready after settlement")
	}
	final := gatherSharingAffinityAccounting(t, c)
	if len(final.DeviceSharingAffinity.StalePools) != 0 {
		t.Fatalf("final stale pools = %v", final.DeviceSharingAffinity.StalePools)
	}
	assertSharingAffinityAccountingState(t, final, device, structured.AffinityStatusLocked, blue)
	a.mutex.RLock()
	_, oldOK := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
	newRec, newOK := a.owners[ownerPoolKey{uid: newUID, pool: key}]
	siblingRec, siblingOK := a.owners[ownerPoolKey{uid: sibling.UID, pool: key}]
	rec := a.affinityDevices[device]
	a.mutex.RUnlock()
	if oldOK {
		t.Fatal("deleted/replaced owner UID still tracked")
	}
	if !siblingOK || siblingRec.pending {
		t.Fatalf("sibling record = %+v/%t, want settled", siblingRec, siblingOK)
	}
	if newUID == "" {
		assertSharingAffinityAccountingCapacity(t, final, device, "5")
	} else {
		assertSharingAffinityAccountingCapacity(t, final, device, "10")
		if !newOK || newRec.pending {
			t.Fatalf("new UID record = %+v/%t, want settled", newRec, newOK)
		}
	}
	if rec == nil || !maps.Equal(rec.established, blue) || rec.ambiguous {
		t.Fatalf("affinity record = %+v, want established blue without ambiguity", rec)
	}
}
