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

// The queued worker computes against an uncommitted equal-RV
// assumed (incompatible red) owner. A Restore to the compatible blue API object
// lands before the worker commits; the red worker result must be rejected and
// must not poison the compatible sibling's lock, capacity or ambiguity state.
func TestSharingAffinityReconciliation_WorkerEqualRVRestoreRejectsUncommittedResult(t *testing.T) {
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
		go func() {
			defer close(restoredDone)
			c.cache.Restore(owner.Namespace + "/" + owner.Name)
		}()
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			obj, err := c.cache.Get(owner.Namespace + "/" + owner.Name)
			if err != nil {
				return false, err
			}
			return maps.Equal(extractClaimSubnet(t, obj.(*resourceapi.ResourceClaim)), blue), nil
		}); err != nil {
			t.Errorf("Restore not cache-visible before worker commit: %v", err)
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
	if n := a.reconcile.NumRequeues(key); n != 0 {
		t.Errorf("requeues=%d, want 0", n)
	}

	unblock()
	reconcileAwait(t, assumedDone, "stale observed assumed callback")
	reconcileAwait(t, restoredDone, "Restore callback")
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
	assertSharingAffinityAccountingCapacity(t, final, device, "10")
	a.mutex.RLock()
	ownerRec, ownerOK := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
	siblingRec, siblingOK := a.owners[ownerPoolKey{uid: sibling.UID, pool: key}]
	rec := a.affinityDevices[device]
	a.mutex.RUnlock()
	if !ownerOK || !siblingOK || ownerRec.pending || siblingRec.pending {
		t.Fatalf("owner records = owner:%+v/%t sibling:%+v/%t, want both settled", ownerRec, ownerOK, siblingRec, siblingOK)
	}
	if rec == nil || !maps.Equal(rec.established, blue) || rec.ambiguous {
		t.Fatalf("affinity record = %+v, want established blue without ambiguity", rec)
	}
}
