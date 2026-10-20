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
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// The pending owner must come from a real allocation-changing informer
// update held after extraction, not from a direct record mutation. Reserve
// between Filter and the commit must fail with Error (backoff) and leave
// nothing behind; the event-free retry after settlement must succeed and see
// the updated owner's capacity, with the old contribution gone.
func TestSharingAffinityReserve_RealOwnerUpdateBetweenFilterAndReserve(t *testing.T) {
	owner := sharingAffinityAccountingClaim("owner", "blue")
	claim := reserveClaim("candidate", "blue")
	f := newReserveFixture(t, true, owner, claim)
	c := f.tc.draManager.resourceClaimTracker
	a := c.allocatedDevices
	if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return f.snapshot(t).DeviceSharingAffinity.DeviceStates[structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)].Status == structured.AffinityStatusLocked, nil
	}); err != nil {
		t.Fatalf("initial owner did not settle: %v", err)
	}
	reserveAssertState(t, f.snapshot(t), "blue", "5")

	pod := reservePod("candidate-pod", claim)
	state := f.filter(t, pod, nil)

	entered, release := make(chan struct{}), make(chan struct{})
	var once, enterOnce sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	a.mutex.Lock()
	a.observedAfterComputeHook = func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	}
	a.mutex.Unlock()

	cached, err := c.cache.Get(owner.Namespace + "/" + owner.Name)
	if err != nil {
		t.Fatal(err)
	}
	current := cached.(*resourceapi.ResourceClaim)
	updated := current.DeepCopy()
	updated.ResourceVersion = "99"
	updated.Status.Allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("7")}
	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		// AssumeCache delivers the update to the tracker's real handler.
		assumecache.AddTestObject(c.cache, updated)
	}()
	reserveAwait(t, entered)

	status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
	if status.Code() != fwk.Error || !errors.Is(status.AsError(), errAffinityNotReady) {
		t.Fatalf("Reserve during held owner update = %v, want Error wrapping errAffinityNotReady", status)
	}
	if c.GetPendingAllocation(claim.UID) != nil {
		t.Error("failed Reserve published a pending allocation")
	}
	if tokens := f.pendingTokens(claim.UID); len(tokens) != 0 {
		t.Errorf("failed Reserve left tokens %v", tokens)
	}
	f.unreserve(state, pod)

	unblock()
	reserveAwait(t, handlerDone)
	if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		a.mutex.Lock()
		defer a.mutex.Unlock()
		for k, rec := range a.owners {
			if k.uid == owner.UID && rec.pending {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("owner update did not settle: %v", err)
	}
	// Old 5-unit contribution replaced by the new 7-unit one, lock retained.
	reserveAssertState(t, f.snapshot(t), "blue", "7")

	// Event-free retry: a fresh cycle with no intervening cluster event.
	retry := f.filter(t, pod, nil)
	f.reserve(t, retry, pod)
	if c.GetPendingAllocation(claim.UID) == nil {
		t.Fatal("retried Reserve did not publish the pending allocation")
	}
	if tokens := f.pendingTokens(claim.UID); len(tokens) != 1 {
		t.Errorf("retried Reserve tokens = %v, want exactly one", tokens)
	}
	f.unreserve(retry, pod)
	if c.GetPendingAllocation(claim.UID) != nil {
		t.Error("Unreserve after retry left a pending allocation")
	}
	after := f.snapshot(t)
	reserveAssertState(t, after, "blue", "7")
}

func reserveAwait(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for held owner extraction")
	}
}
