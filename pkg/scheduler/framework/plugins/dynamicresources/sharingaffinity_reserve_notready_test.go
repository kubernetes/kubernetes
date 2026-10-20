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
	"errors"
	"testing"

	fwk "k8s.io/kube-scheduler/framework"
)

// Covers plugin-level Reserve when a same-epoch owner becomes pending after
// Filter: Error (backoff, no event needed), no leftovers, and a later
// event-free retry succeeds. Stale provenance must win over not-ready.
func TestSharingAffinityReserve_OwnerPendingAfterFilter(t *testing.T) {
	for _, stale := range []bool{false, true} {
		name := "not-ready"
		if stale {
			name = "stale-precedes-not-ready"
		}
		t.Run(name, func(t *testing.T) {
			owner := sharingAffinityAccountingClaim("owner", "blue")
			claim := reserveClaim("candidate", "blue")
			f := newReserveFixture(t, true, owner, claim)
			c := f.tc.draManager.resourceClaimTracker
			a := c.allocatedDevices
			key := poolName{accountingDriver, accountingPool}

			setPending := func(pending bool) {
				t.Helper()
				a.mutex.Lock()
				defer a.mutex.Unlock()
				found := false
				for k, rec := range a.owners {
					if k.pool == key && k.uid == owner.UID {
						rec.pending, found = pending, true
					}
				}
				if !found {
					t.Fatalf("no owner record for %s in pool %v", owner.UID, key)
				}
			}

			pod := reservePod("candidate-pod", claim)
			state := f.filter(t, pod, nil)
			before := f.snapshot(t)
			setPending(true)
			if stale {
				changed := f.metadata.DeepCopy()
				changed.Spec.SharingAffinity[0].Expression = `"red"`
				f.updateSlice(t, changed)
			}
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if stale {
				if status.Code() != fwk.Unschedulable {
					t.Fatalf("Reserve = %v, want Unschedulable stale provenance despite not-ready pool", status)
				}
				if !errors.Is(status.AsError(), errAffinityStale) {
					t.Errorf("Reserve error %v does not wrap errAffinityStale", status.AsError())
				}
				if errors.Is(status.AsError(), errAffinityNotReady) {
					t.Errorf("Reserve error %v reports not-ready instead of stale provenance", status.AsError())
				}
			} else {
				if status.Code() != fwk.Error {
					t.Fatalf("Reserve = %v, want Error so backoff retries without an event", status)
				}
				if !errors.Is(status.AsError(), errAffinityNotReady) {
					t.Errorf("Reserve error %v does not wrap errAffinityNotReady", status.AsError())
				}
			}
			if c.GetPendingAllocation(claim.UID) != nil {
				t.Error("failed Reserve published a pending allocation")
			}
			if tokens := f.pendingTokens(claim.UID); len(tokens) != 0 {
				t.Errorf("failed Reserve left tokens %v", tokens)
			}
			s, err := getStateData(state)
			if err != nil {
				t.Fatal(err)
			}
			s.mutex.Lock()
			_, recorded := s.affinityReservations[claim.UID]
			s.mutex.Unlock()
			if recorded {
				t.Error("failed Reserve recorded a reservation token in cycle state")
			}
			f.unreserve(state, pod)
			if stale {
				return
			}
			// Reconciliation settles; the backoff retry is a fresh cycle
			// with no cluster event in between.
			setPending(false)
			reserveAssertUnchanged(t, before, f.snapshot(t))
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
		})
	}
}
