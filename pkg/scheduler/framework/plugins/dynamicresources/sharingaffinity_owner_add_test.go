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

// An owner added while the reconciler is mid-pass for the same
// pool must be either Locked or fail closed (stale pool); a ready snapshot
// with the device absent would permit a conflicting allocation.
func TestSharingAffinityOwnerAddDuringWorkerPass(t *testing.T) {
	for _, phase := range []string{"before-begin", "after-compute"} {
		for _, owner := range []string{"observed", "share-id-only"} {
			t.Run(phase+"/"+owner, func(t *testing.T) {
				slices := poolEventsSlices()
				ctx, tc := poolEventsSetup(t, slices)
				c := tc.draManager.resourceClaimTracker
				pool := poolName{accountingDriver, accountingPool}
				anchor := sharingAffinityAccountingClaim("anchor", "blue")
				assumecache.AddTestObject(c.cache, anchor)
				reconcileWaitState(t, ctx, c, sharingAffinityAccountingDevice(anchor), structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})

				entered, release := make(chan struct{}), make(chan struct{})
				var once, relOnce sync.Once
				unblock := func() { relOnce.Do(func() { close(release) }) }
				t.Cleanup(unblock)
				pause := func() { once.Do(func() { close(entered); <-release }) }
				if phase == "before-begin" {
					c.reconcileProcessHook = func(p poolName) {
						if p == pool {
							pause()
						}
					}
				} else {
					c.reconcileAfterComputeHook = func(p poolName, _ types.UID) {
						if p == pool {
							pause()
						}
					}
					// Force a recompute of the anchor so the worker has an item.
					func() {
						unlock := c.allocatedDevices.lockOwners()
						defer unlock()
						for key, rec := range c.allocatedDevices.owners {
							if key.pool == pool {
								rec.pending = true
							}
						}
					}()
				}
				func() {
					unlock := c.allocatedDevices.lockOwners()
					defer unlock()
					c.allocatedDevices.enqueuePoolLocked(pool)
				}()
				reconcileAwait(t, entered, "reconciler pause")

				claim := sharingAffinityAccountingClaim("owner", "blue")
				claim.Status.Allocation.Devices.Results[0].Device = anchor.Status.Allocation.Devices.Results[0].Device
				if owner == "share-id-only" {
					claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
				}
				assumecache.AddTestObject(c.cache, claim)
				device := sharingAffinityAccountingDevice(claim)
				state := gatherSharingAffinityAccounting(t, c)
				_, stale := state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}]
				got, present := state.DeviceSharingAffinity.DeviceStates[device]
				if !stale && (!present || got.Status != structured.AffinityStatusLocked || got.LockedAffinity["subnet"] != "blue") {
					t.Errorf("ready snapshot during worker pass lost the lock: present=%t state=%+v", present, got)
				}
				if got.Status == structured.AffinityStatusUnreconstructable {
					t.Errorf("compatible owner poisoned during worker pass: %+v", got)
				}
				unblock()
				reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				c.reconcileProcessHook, c.reconcileAfterComputeHook = nil, nil
				assumecache.DeleteTestObject(c.cache, claim)
				reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				assumecache.DeleteTestObject(c.cache, anchor)
				reconcileWaitState(t, ctx, c, device, structured.AffinityStatusClean, nil)
			})
		}
	}
}
