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
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// The first owner of a pool is computed by the observed handler
// outside the lock. A worker pass for the same pool can take over that pending
// record with a newer token before the observed commit. Every snapshot in that
// window must fence the pool (stale); the superseded observed commit must not
// publish a ready state; only the worker's commit may lock.
func TestSharingAffinityFirstOwnerTokenTakeover(t *testing.T) {
	for _, owner := range []string{"observed", "share-id-only"} {
		t.Run(owner, func(t *testing.T) {
			ctx, tc := poolEventsSetup(t, poolEventsSlices())
			c := tc.draManager.resourceClaimTracker
			a := c.allocatedDevices
			pool := poolName{accountingDriver, accountingPool}
			poolID := schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}
			claim := sharingAffinityAccountingClaim("first", "blue")
			if owner == "share-id-only" {
				claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
			}
			device := sharingAffinityAccountingDevice(claim)

			obsEntered, obsRelease := make(chan struct{}), make(chan struct{})
			wkEntered, wkRelease := make(chan struct{}), make(chan struct{})
			var o1, o2, r1, r2 sync.Once
			t.Cleanup(func() { r1.Do(func() { close(obsRelease) }); r2.Do(func() { close(wkRelease) }) })
			a.observedAfterComputeHook = func() { o1.Do(func() { close(obsEntered); <-obsRelease }) }
			c.reconcileAfterComputeHook = func(p poolName, uid types.UID) {
				if p == pool && uid == claim.UID {
					o2.Do(func() { close(wkEntered); <-wkRelease })
				}
			}

			tokenOf := func() (uint64, bool) {
				unlock := a.lockOwners()
				defer unlock()
				rec := a.owners[ownerPoolKey{uid: claim.UID, pool: pool}]
				if rec == nil {
					return 0, false
				}
				return rec.token, rec.pending
			}
			assertFenced := func(stage string) {
				t.Helper()
				state := gatherSharingAffinityAccounting(t, c)
				if _, stale := state.DeviceSharingAffinity.StalePools[poolID]; !stale {
					t.Errorf("%s: pool ready while first owner pending: %+v", stage, state.DeviceSharingAffinity.DeviceStates[device])
				}
			}

			added := make(chan struct{})
			go func() { defer close(added); assumecache.AddTestObject(c.cache, claim) }()
			reconcileAwait(t, obsEntered, "observed compute")
			observedToken, pending := tokenOf()
			if !pending || observedToken == 0 {
				t.Fatalf("observed owner not pending: token=%d pending=%t", observedToken, pending)
			}
			assertFenced("observed computing")

			func() {
				unlock := a.lockOwners()
				defer unlock()
				a.enqueuePoolLocked(pool)
			}()
			reconcileAwait(t, wkEntered, "worker takeover compute")
			workerToken, pending := tokenOf()
			if !pending || workerToken <= observedToken {
				t.Fatalf("worker did not take over: observed=%d worker=%d pending=%t", observedToken, workerToken, pending)
			}
			assertFenced("both computing")

			r1.Do(func() { close(obsRelease) })
			reconcileAwait(t, added, "observed commit")
			if tok, pending := tokenOf(); tok != workerToken || !pending {
				t.Errorf("superseded observed commit settled the record: token=%d pending=%t", tok, pending)
			}
			assertFenced("observed superseded")

			r2.Do(func() { close(wkRelease) })
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			a.observedAfterComputeHook, c.reconcileAfterComputeHook = nil, nil
			assumecache.DeleteTestObject(c.cache, claim)
			reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
		})
	}
}
