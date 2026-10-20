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
	"encoding/json"
	"errors"
	"maps"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func affinityReservationForPoolView(claim *resourceapi.ResourceClaim, token reservationToken, view poolMeta, constrained bool, values map[string]string) affinityReservation {
	result := claim.Status.Allocation.Devices.Results[0]
	return affinityReservation{
		claimUID: claim.UID,
		claimRV:  claim.ResourceVersion,
		claim:    claim,
		token:    token,
		provenance: []structured.DeviceAffinity{{
			ResultIndex: 0,
			Request:     result.Request,
			Device:      structured.MakeDeviceID(result.Driver, result.Pool, result.Device),
			Pool:        sharingaffinity.PoolKey{Driver: result.Driver, Name: result.Pool, Generation: view.generation},
			Digest:      view.digest,
			Constrained: constrained,
			Contributes: constrained && result.ShareID != nil,
			Values:      maps.Clone(values),
		}},
		fenceEpochs: map[poolName]uint64{{result.Driver, result.Pool}: view.epoch},
	}
}

func extractClaimSubnet(t *testing.T, claim *resourceapi.ResourceClaim) map[string]string {
	t.Helper()
	var parameters map[string]any
	if err := json.Unmarshal(claim.Spec.Devices.Config[0].Opaque.Parameters.Raw, &parameters); err != nil {
		t.Fatalf("decode claim config: %v", err)
	}
	subnet, ok := parameters["subnet"].(string)
	if !ok {
		t.Fatalf("claim config has no string subnet: %v", parameters)
	}
	return map[string]string{"subnet": subnet}
}

func makeReserveFixtureSingleUnconstrainedPool(t *testing.T, f *reserveFixture) *resourceapi.ResourceSlice {
	t.Helper()
	if err := f.tc.client.ResourceV1().ResourceSlices().Delete(f.ctx, f.metadata.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		t.Fatalf("delete metadata ResourceSlice: %v", err)
	}
	current, err := f.tc.client.ResourceV1().ResourceSlices().Get(f.ctx, f.devices.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get device ResourceSlice: %v", err)
	}
	devices := current.DeepCopy()
	devices.Spec.Pool.ResourceSliceCount = 1
	devices.Spec.SharingAffinity = nil
	if _, err := f.tc.client.ResourceV1().ResourceSlices().Update(f.ctx, devices, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("update device ResourceSlice: %v", err)
	}
	if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		slices, err := f.tc.draManager.ResourceSlices().ListWithDeviceTaintRules()
		if err != nil {
			return false, err
		}
		return len(slices) == 1 && slices[0].Name == devices.Name && reflect.DeepEqual(slices[0].Spec, devices.Spec), nil
	}); err != nil {
		t.Fatalf("wait for single unconstrained pool: %v", err)
	}
	return devices
}

func updateReserveFixtureSlice(t *testing.T, f *reserveFixture, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
	t.Helper()
	current, err := f.tc.client.ResourceV1().ResourceSlices().Get(f.ctx, slice.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ResourceSlice %s: %v", slice.Name, err)
	}
	updated := slice.DeepCopy()
	updated.UID, updated.ResourceVersion = current.UID, current.ResourceVersion
	f.updateSlice(t, updated)
	return updated
}

// A same-RV Assume can compute against changed Spec config while
// another goroutine restores the API object in the cache. The stale assumed
// commit must be superseded by the cache-visible Restore, preserving the
// compatible sibling's established lock and accounting.
func TestSharingAffinityReconciliation_EqualRVAssumeRestoreSupersedesPausedObservedCommit(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	blue := map[string]string{"subnet": "blue"}
	ctx := t.Context()
	c, _ := newInterleaveTracker(t, poolEventsSlices())
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

	entered, release, assumedDone, restoredDone := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var computes atomic.Int32
	a.observedAfterComputeHook = func() {
		if computes.Add(1) == 1 {
			close(entered)
			<-release
		}
	}
	defer func() { a.observedAfterComputeHook = nil }()

	go func() {
		defer close(assumedDone)
		assumed := sharingAffinityAccountingClaim(owner.UID, "red")
		if err := c.cache.Assume(assumed); err != nil {
			t.Errorf("Assume: %v", err)
		}
	}()
	reconcileAwait(t, entered, "assumed owner computation")

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
		t.Fatalf("Restore did not become cache-visible while Assume callback was paused: %v", err)
	}
	restored := gatherSharingAffinityAccounting(t, c)
	if !restored.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
		t.Fatalf("paused stale Assume exposed a ready snapshot before Restore callback: %+v", restored.DeviceSharingAffinity.DeviceStates[device])
	}
	unblock()
	reconcileAwait(t, assumedDone, "stale assumed callback")
	reconcileAwait(t, restoredDone, "cache Restore callback")
	interleaveDrain(t, ctx, c)
	final := gatherSharingAffinityAccounting(t, c)
	if len(final.DeviceSharingAffinity.StalePools) != 0 {
		t.Fatalf("final stale pools = %v", final.DeviceSharingAffinity.StalePools)
	}
	assertSharingAffinityAccountingState(t, final, device, structured.AffinityStatusLocked, blue)
	assertSharingAffinityAccountingCapacity(t, final, device, "10")
	a.mutex.RLock()
	ownerRec, ownerOK := a.owners[ownerPoolKey{uid: owner.UID, pool: key}]
	siblingRec, siblingOK := a.owners[ownerPoolKey{uid: sibling.UID, pool: key}]
	affinityRecord := a.affinityDevices[device]
	a.mutex.RUnlock()
	if !ownerOK || !siblingOK || ownerRec.pending || siblingRec.pending {
		t.Fatalf("restored owner records = owner:%+v/%t sibling:%+v/%t, want both settled", ownerRec, ownerOK, siblingRec, siblingOK)
	}
	if affinityRecord == nil || !maps.Equal(affinityRecord.established, blue) || affinityRecord.ambiguous {
		t.Fatalf("restored affinity record = %+v, want established blue without ambiguity", affinityRecord)
	}
}

// While a two-owner reconciliation batch is still computing,
// current-epoch reservations are not ready and stale provenance is stale. The
// public state must remain stale rather than exposing a partial Locked result.
func TestSharingAffinityReconciliation_TwoOwnerBatchReserveBarrier(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	ctx := t.Context()
	base := poolEventsSlices()
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	c.cache.AddEventHandler(a.handlers())
	blueOwner := sharingAffinityAccountingClaim("blue-owner", "blue")
	redOwner := sharingAffinityAccountingClaim("red-owner", "blue")
	device := sharingAffinityAccountingDevice(blueOwner)
	assumecache.AddTestObject(c.cache, blueOwner)
	assumecache.AddTestObject(c.cache, redOwner)
	interleaveDrain(t, ctx, c)
	assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

	oldView := publishRaceCommitted(c).pools[key]
	updated := reconcileSlices(base, 2)
	updated[0].Spec.SharingAffinity[0].Expression = `parameters.subnet`
	lister.set(updated...)
	c.publishMetadata(false, affectedPools(updated[0]))
	currentView := publishRaceCommitted(c).pools[key]
	a.reconcile.Add(key)

	candidate := sharingAffinityAccountingClaim("candidate", "blue")
	unallocated := candidate.DeepCopy()
	unallocated.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, unallocated)
	currentReservation := affinityReservationForPoolView(candidate, "current-epoch", currentView, true, map[string]string{"subnet": "blue"})
	staleReservation := affinityReservationForPoolView(candidate, "stale-provenance", oldView, true, map[string]string{"subnet": "blue"})

	var checked atomic.Bool
	c.reconcileAfterComputeHook = func(pool poolName, uid types.UID) {
		if pool != key || !checked.CompareAndSwap(false, true) {
			return
		}
		state := gatherSharingAffinityAccounting(t, c)
		if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
			t.Fatalf("batch computation exposed ready state before commit: %+v", state.DeviceSharingAffinity.DeviceStates[device])
		}
		if err := c.commitAffinityReservation(currentReservation, map[poolName]poolMeta{key: currentView}); !errors.Is(err, errAffinityNotReady) {
			t.Fatalf("current-epoch Reserve during batch = %v, want not-ready", err)
		}
		if c.GetPendingAllocation(candidate.UID) != nil {
			t.Fatal("not-ready Reserve left a pending allocation")
		}
		if err := c.ReserveClaimWithAffinity(ctx, staleReservation); !errors.Is(err, errAffinityStale) {
			t.Fatalf("stale provenance Reserve = %v, want stale", err)
		}
		if c.GetPendingAllocation(candidate.UID) != nil {
			t.Fatal("stale Reserve left a pending allocation")
		}
	}
	c.processNext(ctx)
	c.reconcileAfterComputeHook = nil
	if !checked.Load() {
		t.Fatal("reconciliation batch did not hit the compute barrier")
	}
	state := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	assertSharingAffinityAccountingCapacity(t, state, device, "10")
}

// An occupied but unconstrained pool may accept a lister-ahead
// inactive reservation. Publication of the same unconstrained generation settles
// that inactive record without worker recomputation; a later lister generation
// invalidates old captures and leaves no reservation leftovers.
func TestSharingAffinityReconciliation_UnconstrainedListerAheadInactiveReserve(t *testing.T) {
	for _, occupied := range []bool{false, true} {
		name := "unoccupied"
		if occupied {
			name = "occupied"
		}
		t.Run(name, func(t *testing.T) { testUnconstrainedListerAheadInactiveReserve(t, occupied) })
	}
}

func testUnconstrainedListerAheadInactiveReserve(t *testing.T, occupied bool) {
	claim := reserveClaim("inactive", "blue")
	f := newReserveFixture(t, true, claim)
	tracker := f.tc.draManager.resourceClaimTracker
	key := poolName{accountingDriver, accountingPool}
	device := structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)
	var settlementComputes atomic.Int32
	tracker.reconcileAfterComputeHook = func(pool poolName, uid types.UID) {
		if pool == key && uid == claim.UID {
			settlementComputes.Add(1)
		}
	}

	unconstrained := makeReserveFixtureSingleUnconstrainedPool(t, f)
	reconcileWaitForIdle(t, f.ctx, tracker)
	if occupied {
		// Allocated capacity owners arrive only after extractors are gone, so
		// the pool is occupied without any baseline, fence or backer.
		for _, subnet := range []string{"blue", "red"} {
			o := sharingAffinityAccountingClaim(types.UID("occupant-"+subnet), subnet)
			o.Finalizers = []string{resourceapi.Finalizer}
			o.UID, o.ResourceVersion = "", ""
			created, err := f.tc.client.ResourceV1().ResourceClaims(o.Namespace).Create(f.ctx, o, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("create occupant: %v", err)
			}
			waitForResourceClaimInformer(t, f.ctx, tracker, created)
		}
		reconcileWaitForIdle(t, f.ctx, tracker)
		assertSharingAffinityAccountingCapacity(t, f.snapshot(t), device, "10")
	}

	block := make(chan struct{})
	entered := make(chan struct{})
	var publication atomic.Int32
	tracker.publishBeforeCommitHook = func(driver, pool string) {
		if driver == key.driver && pool == key.pool && publication.Add(1) == 1 {
			close(entered)
			<-block
		}
	}
	defer func() {
		tracker.publishBeforeCommitHook = nil
		select {
		case <-block:
		default:
			close(block)
		}
	}()

	ahead := unconstrained.DeepCopy()
	ahead.Spec.Pool.Generation++
	ahead = updateReserveFixtureSlice(t, f, ahead)
	reconcileAwait(t, entered, "held unconstrained publication")
	if occupied {
		assertSharingAffinityAccountingCapacity(t, f.snapshot(t), device, "10")
	}

	pod := reservePod("inactive-pod", claim)
	state := f.filter(t, pod, nil)
	f.reserve(t, state, pod)
	if pending := tracker.GetPendingAllocation(claim.UID); pending == nil {
		t.Fatal("inactive Reserve did not publish a pending allocation")
	}
	token := reserveStateToken(t, state, claim.UID)
	if tokens := f.pendingTokens(claim.UID); !tokens.Has(token) || tokens.Len() != 1 {
		t.Fatalf("pending tokens = %v, want exactly Reserve token %q", tokens, token)
	}
	if occupied {
		assertSharingAffinityAccountingCapacity(t, f.snapshot(t), device, "15")
	}
	a := tracker.allocatedDevices
	a.mutex.RLock()
	rec := a.owners[ownerPoolKey{uid: claim.UID, inFlight: true, pool: key}]
	affinityRecord := a.affinityDevices[device]
	var recSnapshot ownerPoolRecord
	if rec != nil {
		recSnapshot = *rec
	}
	a.mutex.RUnlock()
	if rec == nil || !recSnapshot.inactive || recSnapshot.view.generation != ahead.Spec.Pool.Generation {
		t.Fatalf("inactive record = %+v, want inactive generation %d", recSnapshot, ahead.Spec.Pool.Generation)
	}
	if affinityRecord != nil && (len(affinityRecord.backers) != 0 || len(affinityRecord.owners) != 0 || len(affinityRecord.established) != 0 || affinityRecord.ambiguous) {
		t.Fatalf("inactive unconstrained reserve created affinity backer: %+v", affinityRecord)
	}

	beforeReleaseComputes := settlementComputes.Load()
	if beforeReleaseComputes > 1 {
		t.Fatalf("held publication pass recomputed inactive claim %d times before release, want at most one", beforeReleaseComputes)
	}
	if recSnapshot.pending && beforeReleaseComputes != 1 {
		t.Fatalf("inactive record is pending from worker pass but compute count=%d, want one held-pass compute", beforeReleaseComputes)
	}
	// While the original publication is still paused, a second lister advance
	// between Filter and Reserve must invalidate a capture admitted through the
	// unpublished-unconstrained exception.
	pausedClaim := reserveClaim("inactive-paused-stale", "blue")
	pausedCreate := pausedClaim.DeepCopy()
	pausedCreate.UID, pausedCreate.ResourceVersion = "", ""
	pausedCreated, err := f.tc.client.ResourceV1().ResourceClaims(pausedClaim.Namespace).Create(f.ctx, pausedCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create paused stale claim: %v", err)
	}
	waitForResourceClaimInformer(t, f.ctx, tracker, pausedCreated)
	pausedPod := reservePod("inactive-paused-stale-pod", pausedCreated)
	pausedState := f.filter(t, pausedPod, nil)
	select {
	case <-block:
		t.Fatal("publication released before paused stale capture")
	default:
	}
	if publication.Load() != 1 {
		t.Fatalf("publication passes = %d before paused advance, want 1 still held", publication.Load())
	}
	// The held publication also blocks the tracker's informer delivery, so the
	// second advance is injected at the lister seam that Reserve re-reads.
	realLister := tracker.metadataSlices
	tracker.metadataSlices = generationAdvancingLister{inner: realLister, pool: key, delta: 1}
	pausedBefore := f.snapshot(t)
	pausedStatus := f.tc.p.Reserve(f.ctx, pausedState, pausedPod, workerNode.Name)
	if pausedStatus.Code() != fwk.Unschedulable || !errors.Is(pausedStatus.AsError(), errAffinityStale) {
		t.Fatalf("paused-publication stale Reserve = %v, want Unschedulable wrapping errAffinityStale", pausedStatus)
	}
	reserveAssertUnchanged(t, pausedBefore, f.snapshot(t))
	if tracker.GetPendingAllocation(pausedCreated.UID) != nil || f.pendingTokens(pausedCreated.UID).Len() != 0 {
		t.Fatal("paused-publication stale Reserve left a pending allocation or token")
	}
	a.mutex.RLock()
	_, pausedLeftover := a.owners[ownerPoolKey{uid: pausedCreated.UID, inFlight: true, pool: key}]
	a.mutex.RUnlock()
	if pausedLeftover {
		t.Fatal("paused-publication stale Reserve left an owner record")
	}
	tracker.metadataSlices = realLister
	if tokens := f.pendingTokens(claim.UID); !tokens.Has(token) || tokens.Len() != 1 {
		t.Fatalf("paused stale Reserve disturbed first reservation tokens = %v, want %q", tokens, token)
	}

	close(block)
	reconcileWaitForIdle(t, f.ctx, tracker)
	if got := settlementComputes.Load(); got != beforeReleaseComputes {
		t.Fatalf("inactive unconstrained publication settlement recomputed claim %s: before release=%d after settlement=%d, want unchanged", claim.UID, beforeReleaseComputes, got)
	}
	a.mutex.RLock()
	settled := a.owners[ownerPoolKey{uid: claim.UID, inFlight: true, pool: key}]
	settledAffinityRecord := a.affinityDevices[device]
	var settledSnapshot ownerPoolRecord
	if settled != nil {
		settledSnapshot = *settled
	}
	a.mutex.RUnlock()
	if settled == nil || settledSnapshot.pending || settledSnapshot.view.generation != ahead.Spec.Pool.Generation {
		t.Fatalf("inactive settlement failed to settle at lister generation %d: %+v", ahead.Spec.Pool.Generation, settledSnapshot)
	}
	if settledAffinityRecord != nil && (len(settledAffinityRecord.backers) != 0 || len(settledAffinityRecord.owners) != 0 || len(settledAffinityRecord.established) != 0 || settledAffinityRecord.ambiguous) {
		t.Fatalf("inactive settlement created affinity backer: %+v", settledAffinityRecord)
	}

	staleClaim := reserveClaim("inactive-stale", "blue")
	staleCreate := staleClaim.DeepCopy()
	staleCreate.UID, staleCreate.ResourceVersion = "", ""
	created, err := f.tc.client.ResourceV1().ResourceClaims(staleClaim.Namespace).Create(f.ctx, staleCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create stale claim: %v", err)
	}
	if created.UID == "" {
		created.UID = staleClaim.UID
		created, err = f.tc.client.ResourceV1().ResourceClaims(staleClaim.Namespace).Update(f.ctx, created, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("set stale claim UID: %v", err)
		}
	}
	staleClaim = created
	waitForResourceClaimInformer(t, f.ctx, tracker, staleClaim)
	stalePod := reservePod("inactive-stale-pod", staleClaim)
	staleState := f.filter(t, stalePod, nil)
	newer := ahead.DeepCopy()
	newer.Spec.Pool.Generation++
	newer = updateReserveFixtureSlice(t, f, newer)
	before := f.snapshot(t)
	status := f.tc.p.Reserve(f.ctx, staleState, stalePod, workerNode.Name)
	if status.Code() != fwk.Unschedulable {
		t.Fatalf("stale inactive Reserve = %v, want Unschedulable", status)
	}
	if !errors.Is(status.AsError(), errAffinityStale) {
		t.Fatalf("stale inactive Reserve error = %v, want errAffinityStale identity", status.AsError())
	}
	reserveAssertUnchanged(t, before, f.snapshot(t))
	if tracker.GetPendingAllocation(staleClaim.UID) != nil || f.pendingTokens(staleClaim.UID).Len() != 0 {
		t.Fatal("stale inactive Reserve left a pending allocation or token")
	}
	if tokens := f.pendingTokens(claim.UID); !tokens.Has(token) || tokens.Len() != 1 {
		t.Fatalf("stale Reserve disturbed first reservation tokens = %v, want %q", tokens, token)
	}
}

// Once an unconstrained lister-ahead pool is affinity-relevant
// because of an active fence or newly added extractors, old captures are stale
// and new cycles must exclude the pool and register waiters instead.
func TestSharingAffinityReconciliation_UnconstrainedListerAheadAffinityRelevantContrasts(t *testing.T) {
	for _, mode := range []string{"active-fence", "new-affinity", "occupied-established"} {
		t.Run(mode, func(t *testing.T) {
			claim := reserveClaim("contrast-"+mode, "blue")
			f := newReserveFixture(t, true, claim)
			tracker := f.tc.draManager.resourceClaimTracker
			key := poolName{accountingDriver, accountingPool}
			device := structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)

			unconstrained := makeReserveFixtureSingleUnconstrainedPool(t, f)
			reconcileWaitForIdle(t, f.ctx, tracker)
			if mode == "occupied-established" {
				// Establish a blue owner under extractors; the lister-ahead
				// generation then removes the extractors while the lock is held.
				withAffinity := unconstrained.DeepCopy()
				withAffinity.Spec.Pool.Generation++
				withAffinity.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `parameters.subnet`)}
				unconstrained = updateReserveFixtureSlice(t, f, withAffinity)
				reconcileWaitForIdle(t, f.ctx, tracker)
				o := sharingAffinityAccountingClaim("established-owner", "blue")
				o.Finalizers = []string{resourceapi.Finalizer}
				o.UID, o.ResourceVersion = "", ""
				created, err := f.tc.client.ResourceV1().ResourceClaims(o.Namespace).Create(f.ctx, o, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("create established owner: %v", err)
				}
				waitForResourceClaimInformer(t, f.ctx, tracker, created)
				reconcileWaitForIdle(t, f.ctx, tracker)
				st := gatherSharingAffinityAccounting(t, tracker)
				assertSharingAffinityAccountingState(t, st, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			}
			pod := reservePod("contrast-"+mode+"-pod", claim)
			oldState := f.filter(t, pod, nil)
			if mode == "active-fence" {
				tracker.allocatedDevices.mutex.Lock()
				tracker.allocatedDevices.fenced.Insert(device)
				tracker.allocatedDevices.mutex.Unlock()
			}
			block := make(chan struct{})
			entered := make(chan struct{})
			var publication atomic.Int32
			tracker.publishBeforeCommitHook = func(driver, pool string) {
				if driver == key.driver && pool == key.pool && publication.Add(1) == 1 {
					close(entered)
					<-block
				}
			}
			defer func() {
				tracker.publishBeforeCommitHook = nil
				select {
				case <-block:
				default:
					close(block)
				}
			}()
			ahead := unconstrained.DeepCopy()
			ahead.Spec.Pool.Generation++
			if mode == "new-affinity" {
				ahead.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `parameters.subnet`)}
			}
			ahead = updateReserveFixtureSlice(t, f, ahead)
			reconcileAwait(t, entered, "held affinity-relevant publication")

			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, oldState, pod, workerNode.Name)
			if status.Code() != fwk.Unschedulable {
				t.Fatalf("old capture Reserve = %v, want Unschedulable stale", status)
			}
			if !errors.Is(status.AsError(), errAffinityStale) {
				t.Fatalf("old capture Reserve error = %v, want errAffinityStale identity", status.AsError())
			}
			reserveAssertUnchanged(t, before, f.snapshot(t))
			if tracker.GetPendingAllocation(claim.UID) != nil || f.pendingTokens(claim.UID).Len() != 0 {
				t.Fatal("stale affinity-relevant Reserve left pending allocation or token")
			}

			cycle := framework.NewCycleState()
			_, preFilterStatus := f.tc.p.PreFilter(f.ctx, cycle, pod, f.tc.nodeInfos)
			if !preFilterStatus.IsSuccess() {
				t.Fatalf("PreFilter after lister advance: %v", preFilterStatus)
			}
			filterStatus := f.tc.p.Filter(f.ctx, cycle, pod, f.tc.nodeInfos[0])
			if filterStatus.Code() != fwk.Unschedulable {
				t.Fatalf("Filter after %s lister advance = %v, want Unschedulable exclusion", mode, filterStatus)
			}
			data, err := getStateData(cycle)
			if err != nil {
				t.Fatal(err)
			}
			if !data.affinityExcluded.Has(key) || data.affinityExcludedReasons[key] != "listerAhead" {
				t.Fatalf("cycle exclusions = %v reasons=%v, want listerAhead for %v", data.affinityExcluded, data.affinityExcludedReasons, key)
			}

			waitPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name + "-waiting", UID: types.UID(pod.Name + "-waiting")}}
			pl, handle, _ := affinityPluginWaiter(t, f.ctx, f.tc, waitPod)
			if _, postStatus := pl.PostFilter(f.ctx, cycle, waitPod, nil); postStatus.Code() != fwk.Unschedulable {
				t.Fatalf("PostFilter = %v, want Unschedulable waiter registration", postStatus)
			}
			assertAffinityWaiterIndexes(t, pl.affinityWaiters,
				map[poolName]map[types.UID]*v1.Pod{key: {waitPod.UID: waitPod}},
				map[types.UID]sets.Set[poolName]{waitPod.UID: sets.New(key)})
			pl.activateAffinityWaiters(klog.FromContext(f.ctx), sets.New(key))
			if handle.count() != 0 {
				t.Fatal("affinity-relevant lister-ahead pool activated waiter before publication")
			}
			close(block)
		})
	}
}

func waitForResourceClaimInformer(t *testing.T, ctx context.Context, tracker *claimTracker, claim *resourceapi.ResourceClaim) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		current, err := tracker.Get(claim.Namespace, claim.Name)
		return err == nil && current.UID == claim.UID, nil
	}); err != nil {
		t.Fatalf("wait for ResourceClaim informer: %v", err)
	}
}

// A two-owner batch whose recomputed values conflict must not
// publish a partial Locked result while any owner is still computing, and must
// finish Unreconstructable rather than keeping the first owner's lock.
func TestSharingAffinityReconciliation_TwoOwnerConflictingBatchFinalization(t *testing.T) {
	key := poolName{accountingDriver, accountingPool}
	ctx := t.Context()
	base := poolEventsSlices()
	c, lister := newInterleaveTracker(t, base)
	a := c.allocatedDevices
	c.cache.AddEventHandler(a.handlers())
	withZone := func(uid types.UID, zone string) *resourceapi.ResourceClaim {
		claim := sharingAffinityAccountingClaim(uid, "blue")
		claim.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(
			`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"blue","zone":"` + zone + `"}`)
		return claim
	}
	first, second := withZone("zone-a-owner", "a"), withZone("zone-b-owner", "b")
	device := sharingAffinityAccountingDevice(first)
	assumecache.AddTestObject(c.cache, first)
	assumecache.AddTestObject(c.cache, second)
	interleaveDrain(t, ctx, c)
	assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

	updated := reconcileSlices(base, 2)
	updated[0].Spec.SharingAffinity[0].Expression = `parameters.zone`
	lister.set(updated...)
	c.publishMetadata(false, affectedPools(updated[0]))
	a.reconcile.Add(key)

	var computes atomic.Int32
	c.reconcileAfterComputeHook = func(pool poolName, uid types.UID) {
		if pool != key {
			return
		}
		computes.Add(1)
		state := gatherSharingAffinityAccounting(t, c)
		if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
			t.Errorf("after computing %s the pool was not stale: %+v", uid, state.DeviceSharingAffinity.DeviceStates[device])
		}
		if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusLocked && !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
			t.Errorf("partial-ready Locked publication after computing %s: %+v", uid, got)
		}
	}
	interleaveDrain(t, ctx, c)
	c.reconcileAfterComputeHook = nil
	if computes.Load() < 2 {
		t.Fatalf("batch computed %d owners, want both", computes.Load())
	}
	state := gatherSharingAffinityAccounting(t, c)
	if state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}) {
		t.Fatal("conflicting batch did not finalize")
	}
	assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusUnreconstructable, nil)
	assertSharingAffinityAccountingCapacity(t, state, device, "10")
}

// generationAdvancingLister reports a later pool generation than the wrapped
// lister without going through event delivery.
type generationAdvancingLister struct {
	inner poolSliceLister
	pool  poolName
	delta int64
}

func (l generationAdvancingLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	slices, err := l.inner.ListPatchedResourceSlices()
	if err != nil {
		return nil, err
	}
	out := make([]*resourceapi.ResourceSlice, 0, len(slices))
	for _, slice := range slices {
		if slice.Spec.Driver == l.pool.driver && slice.Spec.Pool.Name == l.pool.pool {
			slice = slice.DeepCopy()
			slice.Spec.Pool.Generation += l.delta
		}
		out = append(out, slice)
	}
	return out, nil
}
