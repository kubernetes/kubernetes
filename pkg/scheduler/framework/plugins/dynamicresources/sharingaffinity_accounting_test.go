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
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

const (
	accountingDriver = "accounting.example.com"
	accountingPool   = "pool"
	accountingDevice = "device"
)

var accountingCapacity = draapi.MakeFullyQualifiedName("units", accountingDriver)

type accountingSliceLister struct {
	slices []*resourceapi.ResourceSlice
}

func (l accountingSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	return l.slices, nil
}

type accountingForbiddenSliceLister struct {
	t *testing.T
}

func (l accountingForbiddenSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	l.t.Error("removing an allocation must use recorded contributions, not read changed pool metadata")
	return nil, errors.New("metadata is no longer available")
}

func newSharingAffinityAccounting() *affinityAccounting {
	pool := resourceapi.ResourcePool{Name: accountingPool, Generation: 1, ResourceSliceCount: 2}
	return &affinityAccounting{
		extractor: sharingaffinity.NewExtractor(256),
		slices: accountingSliceLister{slices: []*resourceapi.ResourceSlice{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "accounting-metadata"},
				Spec: resourceapi.ResourceSliceSpec{
					Driver: accountingDriver, Pool: pool, AllNodes: ptr.To(true),
					SharingAffinity: []resourceapi.SharingAffinityExtractor{{
						Name: "subnet", APIGroup: "config.example.com", Kind: "Network",
						Expression: `has(parameters.subnet) ? parameters.subnet : ""`,
					}},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "accounting-devices"},
				Spec: resourceapi.ResourceSliceSpec{
					Driver: accountingDriver, Pool: pool, AllNodes: ptr.To(true),
					Devices: []resourceapi.Device{{Name: accountingDevice, AllowMultipleAllocations: ptr.To(true)}},
				},
			},
		}},
	}
}

func sharingAffinityAccountingClaim(uid types.UID, subnet string) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: string(uid), UID: uid, ResourceVersion: "1"},
		Spec: resourceapi.ResourceClaimSpec{Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name: "req", Exactly: &resourceapi.ExactDeviceRequest{DeviceClassName: "network"},
			}},
			Config: []resourceapi.DeviceClaimConfiguration{{
				DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver: accountingDriver,
					Parameters: runtime.RawExtension{Raw: []byte(fmt.Sprintf(
						`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":%q}`, subnet))},
				}},
			}},
		}},
		Status: resourceapi.ResourceClaimStatus{Allocation: &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: "req", Driver: accountingDriver, Pool: accountingPool, Device: accountingDevice,
				ShareID:          ptr.To(types.UID(string(uid) + "-share")),
				ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("5")},
			}}},
		}},
	}
}

func setSharingAffinityAccountingGates(t *testing.T, affinityEnabled, capacityEnabled bool) {
	t.Helper()
	// Feature gates and the revision hook are process-global; these tests stay serial.
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: capacityEnabled,
		features.DRASharingAffinity:    affinityEnabled,
	})
}

func newSharingAffinityAccountingTracker(t *testing.T, enabled bool) *claimTracker {
	t.Helper()
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(5)))
	var acct *affinityAccounting
	if enabled {
		acct = newSharingAffinityAccounting()
	}
	return &claimTracker{
		cache:               assumecache.NewAssumeCache(logger, nil, "sharing-affinity claim", "", nil),
		inFlightAllocations: make(map[types.UID]inFlightAllocation),
		allocatedDevices:    newAllocatedDevices(logger, acct),
		logger:              logger,
		affinity:            acct,
	}
}

func sharingAffinityAccountingDevice(claim *resourceapi.ResourceClaim) structured.DeviceID {
	result := claim.Status.Allocation.Devices.Results[0]
	return structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
}

func assertSharingAffinityAccountingOwners(t *testing.T, a *allocatedDevices, device structured.DeviceID, count int, status structured.AffinityStatus, values map[string]string) {
	t.Helper()
	snapshot, revision := a.affinitySnapshot()
	if revision != a.Revision() {
		t.Fatalf("affinity snapshot revision = %d, current revision = %d", revision, a.Revision())
	}
	record, found := snapshot[device]
	if count == 0 {
		if found {
			t.Fatalf("drained device %s retains ownership: %#v", device, record)
		}
		return
	}
	if !found || record == nil {
		t.Fatalf("device %s has no affinity ownership, want %d owners", device, count)
	}
	if len(record.owners) != count {
		t.Errorf("device %s owners = %d, want %d", device, len(record.owners), count)
	}
	state, present := foldDeviceAffinity(record)
	if !present || state.Status != status || !maps.Equal(state.LockedAffinity, values) {
		t.Errorf("device %s affinity = %#v, present = %t; want status %v, values %v", device, state, present, status, values)
	}
	if status != structured.AffinityStatusLocked && state.LockedAffinity != nil {
		t.Errorf("non-Locked state retains a lock map: %#v", state)
	}
}

func gatherSharingAffinityAccounting(t *testing.T, c *claimTracker) *structured.AllocatedState {
	t.Helper()
	state, err := c.GatherAllocatedState()
	if err != nil || state == nil {
		t.Fatalf("GatherAllocatedState() = %#v, %v", state, err)
	}
	return state
}

func assertSharingAffinityAccountingState(t *testing.T, state *structured.AllocatedState, device structured.DeviceID, status structured.AffinityStatus, values map[string]string) {
	t.Helper()
	got, found := state.DeviceSharingAffinity.DeviceStates[device]
	if status == structured.AffinityStatusClean {
		if found {
			t.Errorf("Clean device %s has an affinity entry: %#v", device, got)
		}
		return
	}
	if !found || got.Status != status || !maps.Equal(got.LockedAffinity, values) {
		t.Errorf("device %s affinity = %#v, present = %t; want status %v, values %v", device, got, found, status, values)
	}
	if status != structured.AffinityStatusLocked && got.LockedAffinity != nil {
		t.Errorf("non-Locked state retains a lock map: %#v", got)
	}
}

func assertSharingAffinityAccountingCapacity(t *testing.T, state *structured.AllocatedState, device structured.DeviceID, want string) {
	t.Helper()
	quantity := state.AggregatedCapacity[device][accountingCapacity]
	if quantity == nil || quantity.Cmp(resource.MustParse(want)) != 0 {
		t.Errorf("device %s capacity = %v, want %s", device, quantity, want)
	}
}

func signalSharingAffinityAccounting(t *testing.T, c *claimTracker, claim *resourceapi.ResourceClaim) {
	t.Helper()
	pending := claim.DeepCopy()
	pending.Status.Allocation = nil
	pending.ResourceVersion = "0"
	assumecache.AddTestObject(c.cache, pending)
	if err := c.SignalClaimPendingAllocation(claim.UID, claim); err != nil {
		t.Fatalf("SignalClaimPendingAllocation(%s): %v", claim.UID, err)
	}
}

// Ownership is allocation-identity based, not event-count based.
func TestSharingAffinityAccountingLifecycle(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	blue := map[string]string{"subnet": "blue"}
	t.Run("compatible-owners-and-duplicate-events", func(t *testing.T) {
		c := newSharingAffinityAccountingTracker(t, true)
		a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "blue")
		device := sharingAffinityAccountingDevice(a)
		c.allocatedDevices.onAdd(a)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onAdd(a.DeepCopy())
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onAdd(b)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 2, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onDelete(a)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onDelete(a.DeepCopy())
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onDelete(b)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
		if len(c.allocatedDevices.claimAffinity) != 0 {
			t.Error("drained claims retain affinity ownership")
		}
	})
	t.Run("two-results-on-one-device", func(t *testing.T) {
		c := newSharingAffinityAccountingTracker(t, true)
		a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "blue")
		second := *a.Status.Allocation.Devices.Results[0].DeepCopy()
		second.ShareID = ptr.To(types.UID("second-share"))
		a.Status.Allocation.Devices.Results = append(a.Status.Allocation.Devices.Results, second)
		device := sharingAffinityAccountingDevice(a)
		c.allocatedDevices.onAdd(a)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 2, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onAdd(b)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 3, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onDelete(a)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onDelete(b)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
	})
	t.Run("allocation-updates-and-tombstone", func(t *testing.T) {
		c := newSharingAffinityAccountingTracker(t, true)
		allocated := sharingAffinityAccountingClaim("a", "blue")
		pending := allocated.DeepCopy()
		pending.Status.Allocation = nil
		device := sharingAffinityAccountingDevice(allocated)
		c.allocatedDevices.onAdd(pending)
		c.allocatedDevices.onUpdate(pending, pending.DeepCopy())
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
		c.allocatedDevices.onUpdate(pending, allocated)
		c.allocatedDevices.onUpdate(allocated, allocated.DeepCopy())
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.allocatedDevices.onUpdate(allocated, pending)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
		c.allocatedDevices.onUpdate(pending, allocated)
		c.allocatedDevices.onDelete(cache.DeletedFinalStateUnknown{Key: "default/a", Obj: allocated})
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
	})
	for _, update := range []bool{false, true} {
		t.Run(fmt.Sprintf("uid-replacement-update=%t", update), func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			old := sharingAffinityAccountingClaim("old", "blue")
			replacement := sharingAffinityAccountingClaim("new", "red")
			replacement.Name = old.Name
			device := sharingAffinityAccountingDevice(old)
			c.allocatedDevices.onAdd(old)
			if update {
				c.allocatedDevices.onUpdate(old, replacement)
			} else {
				c.allocatedDevices.onDelete(old)
				c.allocatedDevices.onAdd(replacement)
			}
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
			c.allocatedDevices.onDelete(cache.DeletedFinalStateUnknown{Key: "default/old", Obj: old})
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
			if _, found := c.allocatedDevices.claimAffinity[old.UID]; found {
				t.Error("replaced UID retains affinity ownership")
			}
			c.allocatedDevices.onDelete(replacement)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
		})
	}
	t.Run("removal-does-not-reextract", func(t *testing.T) {
		c := newSharingAffinityAccountingTracker(t, true)
		claim := sharingAffinityAccountingClaim("a", "blue")
		device := sharingAffinityAccountingDevice(claim)
		c.allocatedDevices.onAdd(claim)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		c.affinity.slices = accountingForbiddenSliceLister{t: t}
		deleted := claim.DeepCopy()
		deleted.Spec.Devices.Config = nil
		c.allocatedDevices.onDelete(deleted)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
	})
}

// Removing the offending owner does not make an ambiguous device trustworthy.
func TestSharingAffinityAmbiguityStickyUntilDrain(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	for _, unsatisfied := range []bool{false, true} {
		t.Run(fmt.Sprintf("unsatisfied=%t", unsatisfied), func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "red")
			if unsatisfied {
				b.Spec.Devices.Config = nil
			}
			device := sharingAffinityAccountingDevice(a)
			c.allocatedDevices.onAdd(a)
			c.allocatedDevices.onAdd(b)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 2, structured.AffinityStatusUnreconstructable, nil)
			if !unsatisfied {
				logs := c.logger.GetSink().(ktesting.Underlier).GetBuffer().String()
				if !strings.Contains(logs, errAffinityConflict.Error()) {
					t.Errorf("conflict was not logged: %s", logs)
				}
			}
			c.allocatedDevices.onDelete(b)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusUnreconstructable, nil)
			inFlight := sharingAffinityAccountingClaim("in-flight", "blue")
			signalSharingAffinityAccounting(t, c, inFlight)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusUnreconstructable, nil)
			c.MaybeRemoveClaimPendingAllocation(inFlight.UID, true)
			c.allocatedDevices.onDelete(a)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusClean, nil)
			reused := sharingAffinityAccountingClaim("reused", "red")
			c.allocatedDevices.onAdd(reused)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
		})
	}
}

// Affinity handoff deduplication must not repair legacy 5+5 capacity accounting.
func TestSharingAffinityGatherAllocatedStateHandoff(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
			setSharingAffinityAccountingGates(t, enabled, true)
			for _, observedFirst := range []bool{false, true} {
				t.Run(fmt.Sprintf("observed-first=%t", observedFirst), func(t *testing.T) {
					c := newSharingAffinityAccountingTracker(t, enabled)
					claim := sharingAffinityAccountingClaim("handoff", "blue")
					device := sharingAffinityAccountingDevice(claim)
					if observedFirst {
						c.allocatedDevices.onAdd(claim)
						assumecache.AddTestObject(c.cache, claim)
						if err := c.SignalClaimPendingAllocation(claim.UID, claim); err != nil {
							t.Fatal(err)
						}
						if len(c.inFlightAllocations) != 0 {
							t.Error("already observed allocation was added in flight")
						}
					} else {
						signalSharingAffinityAccounting(t, c, claim)
						initial := gatherSharingAffinityAccounting(t, c)
						assertSharingAffinityAccountingCapacity(t, initial, device, "5")
						if enabled {
							assertSharingAffinityAccountingState(t, initial, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
						}
						c.allocatedDevices.onAdd(claim.DeepCopy())
						assumecache.UpdateTestObject(c.cache, claim)
					}
					state := gatherSharingAffinityAccounting(t, c)
					wantCapacity := "10"
					if observedFirst {
						wantCapacity = "5"
					}
					assertSharingAffinityAccountingCapacity(t, state, device, wantCapacity)
					if !state.AllocatedSharedDeviceIDs.Equal(sets.New[structured.DeviceID](device)) || len(state.AllocatedDevices) != 0 {
						t.Errorf("public allocation sets = %v / %v, want one shared DeviceID", state.AllocatedDevices, state.AllocatedSharedDeviceIDs)
					}
					if enabled {
						assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
						assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					} else if c.affinity != nil || c.allocatedDevices.affinityAcct != nil || state.DeviceSharingAffinity.DeviceStates != nil {
						t.Error("gate-off handoff initialized affinity accounting or state")
					}
					deleted := c.MaybeRemoveClaimPendingAllocation(claim.UID, true)
					if deleted == observedFirst {
						t.Errorf("in-flight removal = %t, observed-first = %t", deleted, observedFirst)
					}
					state = gatherSharingAffinityAccounting(t, c)
					assertSharingAffinityAccountingCapacity(t, state, device, "5")
					if !state.AllocatedSharedDeviceIDs.Equal(sets.New(device)) {
						t.Error("in-flight removal changed the observed shared DeviceID")
					}
					if enabled {
						assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					} else if state.DeviceSharingAffinity.DeviceStates != nil {
						t.Error("gate-off snapshot contains affinity state after handoff")
					}
					c.allocatedDevices.onDelete(claim)
					state = gatherSharingAffinityAccounting(t, c)
					if len(state.AllocatedSharedDeviceIDs) != 0 || len(state.AggregatedCapacity) != 0 || len(state.DeviceSharingAffinity.DeviceStates) != 0 {
						t.Errorf("fully drained handoff retains state: %#v", state)
					}
				})
			}
		})
	}
}

func TestSharingAffinityGatherAllocatedStateIdentityConflicts(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	for _, identity := range []string{"same", "different-share", "different-claim", "different-device"} {
		t.Run(identity, func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			inFlight := sharingAffinityAccountingClaim("claim", "blue")
			observed := sharingAffinityAccountingClaim("claim", "red")
			// ResourceVersion is not an ordering proof for allocation contents.
			observed.ResourceVersion = "20"
			inFlight.ResourceVersion = "100"
			switch identity {
			case "different-share":
				observed.Status.Allocation.Devices.Results[0].ShareID = ptr.To(types.UID("other-share"))
			case "different-claim":
				observed.UID = "other-uid"
			case "different-device":
				observed.Status.Allocation.Devices.Results[0].Device = "other-device"
			}
			signalSharingAffinityAccounting(t, c, inFlight)
			c.allocatedDevices.onAdd(observed)
			observedDevice, inFlightDevice := sharingAffinityAccountingDevice(observed), sharingAffinityAccountingDevice(inFlight)
			state := gatherSharingAffinityAccounting(t, c)
			wantShared := sets.New[structured.DeviceID](observedDevice, inFlightDevice)
			if !state.AllocatedSharedDeviceIDs.Equal(wantShared) {
				t.Errorf("shared DeviceIDs = %v, want %v", state.AllocatedSharedDeviceIDs, wantShared)
			}
			if identity == "different-device" {
				assertSharingAffinityAccountingState(t, state, observedDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
				assertSharingAffinityAccountingState(t, state, inFlightDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				assertSharingAffinityAccountingCapacity(t, state, observedDevice, "5")
				assertSharingAffinityAccountingCapacity(t, state, inFlightDevice, "5")
			} else {
				assertSharingAffinityAccountingState(t, state, observedDevice, structured.AffinityStatusUnreconstructable, nil)
				assertSharingAffinityAccountingCapacity(t, state, observedDevice, "10")
			}
			// Both observed and in-flight backers now participate in the
			// semantic lock. A demonstrated conflict survives partial drain.
			owners := 1
			status, values := structured.AffinityStatusLocked, map[string]string{"subnet": "red"}
			if identity != "different-device" {
				status, values = structured.AffinityStatusUnreconstructable, nil
				if identity != "same" {
					owners = 2
				}
			}
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, observedDevice, owners, status, values)
			if !c.MaybeRemoveClaimPendingAllocation(inFlight.UID, true) {
				t.Fatal("in-flight allocation was not removed")
			}
			state = gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingState(t, state, observedDevice, status, values)
			assertSharingAffinityAccountingCapacity(t, state, observedDevice, "5")
			if len(state.DeviceSharingAffinity.DeviceStates) != 1 {
				t.Errorf("post-handoff affinity devices = %d, want 1", len(state.DeviceSharingAffinity.DeviceStates))
			}
			c.allocatedDevices.onDelete(observed)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), observedDevice, structured.AffinityStatusClean, nil)
			next := sharingAffinityAccountingClaim("after-drain", "green")
			next.Status.Allocation.Devices.Results[0].Device = observed.Status.Allocation.Devices.Results[0].Device
			c.allocatedDevices.onAdd(next)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), observedDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "green"})
		})
	}
}

func TestSharingAffinityGatherAllocatedStateSnapshotIsolation(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	c := newSharingAffinityAccountingTracker(t, true)
	observed := sharingAffinityAccountingClaim("observed", "blue")
	inFlight := sharingAffinityAccountingClaim("in-flight", "red")
	inFlight.Status.Allocation.Devices.Results[0].Device = "in-flight-device"
	observedDevice, inFlightDevice := sharingAffinityAccountingDevice(observed), sharingAffinityAccountingDevice(inFlight)
	c.allocatedDevices.onAdd(observed)
	signalSharingAffinityAccounting(t, c, inFlight)
	first := gatherSharingAffinityAccounting(t, c)
	for _, device := range []structured.DeviceID{observedDevice, inFlightDevice} {
		if len(first.DeviceSharingAffinity.DeviceStates[device].LockedAffinity) == 0 || first.AggregatedCapacity[device][accountingCapacity] == nil {
			t.Fatalf("snapshot lacks lock or capacity for %s: %#v", device, first)
		}
		first.DeviceSharingAffinity.DeviceStates[device].LockedAffinity["subnet"] = "mutated"
		first.AggregatedCapacity[device][accountingCapacity].Add(resource.MustParse("100"))
	}
	clear(first.DeviceSharingAffinity.DeviceStates)
	clear(first.AllocatedSharedDeviceIDs)
	clear(first.AggregatedCapacity)
	if first.AllocatedDevices == nil {
		t.Fatal("snapshot has a nil dedicated-device set")
	}
	first.AllocatedDevices.Insert(observedDevice)
	second := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingState(t, second, observedDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	assertSharingAffinityAccountingState(t, second, inFlightDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
	assertSharingAffinityAccountingCapacity(t, second, observedDevice, "5")
	assertSharingAffinityAccountingCapacity(t, second, inFlightDevice, "5")
	if len(second.AllocatedDevices) != 0 || !second.AllocatedSharedDeviceIDs.Equal(sets.New(observedDevice, inFlightDevice)) {
		t.Error("mutating public snapshot sets changed subsequent snapshots")
	}
	owners, _ := c.allocatedDevices.affinitySnapshot()
	record := owners[observedDevice]
	if record == nil || len(record.owners) != 1 {
		t.Fatalf("observed ownership snapshot = %#v, want one owner", record)
	}
	record.ambiguous = true
	for _, contribution := range record.owners {
		if len(contribution.values) == 0 {
			t.Fatalf("observed ownership snapshot has no lock values: %#v", contribution)
		}
		contribution.values["subnet"] = "mutated-owner"
	}
	clear(record.owners)
	delete(owners, observedDevice)
	assertSharingAffinityAccountingOwners(t, c.allocatedDevices, observedDevice, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	c.MaybeRemoveClaimPendingAllocation(inFlight.UID, true)
	c.allocatedDevices.onDelete(observed)
	assertSharingAffinityAccountingState(t, second, observedDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	assertSharingAffinityAccountingState(t, second, inFlightDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
}

func TestSharingAffinityGatherAllocatedStateLegacyModes(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
			setSharingAffinityAccountingGates(t, enabled, true)
			c := newSharingAffinityAccountingTracker(t, enabled)
			shared := sharingAffinityAccountingClaim("shared", "blue")
			dedicated := sharingAffinityAccountingClaim("dedicated", "")
			dedicated.Status.Allocation.Devices.Results[0].Device = "dedicated"
			dedicated.Status.Allocation.Devices.Results[0].ShareID = nil
			admin := sharingAffinityAccountingClaim("admin", "")
			admin.Status.Allocation.Devices.Results[0].Device = "admin"
			admin.Status.Allocation.Devices.Results[0].AdminAccess = ptr.To(true)
			for _, claim := range []*resourceapi.ResourceClaim{shared, dedicated, admin} {
				c.allocatedDevices.onAdd(claim)
				signalSharingAffinityAccounting(t, c, claim)
			}
			state := gatherSharingAffinityAccounting(t, c)
			if !state.AllocatedDevices.Equal(sets.New(sharingAffinityAccountingDevice(dedicated))) {
				t.Errorf("dedicated devices = %v", state.AllocatedDevices)
			}
			if !state.AllocatedSharedDeviceIDs.Equal(sets.New(sharingAffinityAccountingDevice(shared))) {
				t.Errorf("shared devices = %v", state.AllocatedSharedDeviceIDs)
			}
			if len(state.AggregatedCapacity) != 1 {
				t.Errorf("capacity devices = %d, want only the shared device", len(state.AggregatedCapacity))
			}
			assertSharingAffinityAccountingCapacity(t, state, sharingAffinityAccountingDevice(shared), "10")
			if enabled {
				if len(state.DeviceSharingAffinity.DeviceStates) != 1 {
					t.Errorf("affinity devices = %d, want only the shared device", len(state.DeviceSharingAffinity.DeviceStates))
				}
				assertSharingAffinityAccountingState(t, state, sharingAffinityAccountingDevice(shared), structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			} else if state.DeviceSharingAffinity.DeviceStates != nil {
				t.Error("gate-off legacy snapshot contains affinity state")
			}
		})
	}
}

func TestSharingAffinityGatherAllocatedStateRevisionRace(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	for _, remove := range []bool{false, true} {
		t.Run(fmt.Sprintf("remove=%t", remove), func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			claim := sharingAffinityAccountingClaim("revision", "blue")
			device := sharingAffinityAccountingDevice(claim)
			if remove {
				c.allocatedDevices.onAdd(claim)
			}
			previousHook := gatherAllocatedStateTestHook
			t.Cleanup(func() { gatherAllocatedStateTestHook = previousHook })
			var once sync.Once
			calls := 0
			gatherAllocatedStateTestHook = func() {
				once.Do(func() {
					calls++
					if remove {
						c.allocatedDevices.onDelete(claim)
					} else {
						c.allocatedDevices.onAdd(claim)
					}
				})
			}
			state, err := c.GatherAllocatedState()
			if !errors.Is(err, errClaimTrackerConcurrentModification) || state != nil {
				t.Fatalf("revision-invalid snapshot = %#v, %v; want nil, concurrent modification", state, err)
			}
			if calls != 1 {
				t.Fatalf("revision hook calls = %d, want 1", calls)
			}
			state = gatherSharingAffinityAccounting(t, c)
			if remove {
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusClean, nil)
				if state.AllocatedSharedDeviceIDs.Has(device) || len(state.AggregatedCapacity) != 0 {
					t.Error("retry retained allocation data removed by the revision hook")
				}
			} else {
				assertSharingAffinityAccountingState(t, state, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				assertSharingAffinityAccountingCapacity(t, state, device, "5")
				if !state.AllocatedSharedDeviceIDs.Equal(sets.New(device)) {
					t.Error("retry omitted allocation added by the revision hook")
				}
			}
		})
	}
}

func TestSharingAffinityConcurrentAccounting(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	c := newSharingAffinityAccountingTracker(t, true)
	// Keep a constrained allocation throughout the race so successful readers
	// cannot all observe an empty cache and vacuously pass the consistency checks.
	anchor := sharingAffinityAccountingClaim("anchor", "blue")
	anchor.Status.Allocation.Devices.Results[0].Device = "anchor-device"
	c.allocatedDevices.onAdd(anchor)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for writer := range 2 {
		claim := sharingAffinityAccountingClaim(types.UID(fmt.Sprintf("writer-%d", writer)), "blue")
		claim.Status.Allocation.Devices.Results[0].Device = fmt.Sprintf("device-%d", writer)
		wg.Go(func() {
			<-start
			for range 40 {
				c.allocatedDevices.onAdd(claim)
				c.allocatedDevices.onDelete(claim)
			}
		})
	}
	pending := sharingAffinityAccountingClaim("pending-writer", "blue")
	pending.Status.Allocation.Devices.Results[0].Device = "pending-device"
	unallocated := pending.DeepCopy()
	unallocated.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, unallocated)
	wg.Go(func() {
		<-start
		for range 40 {
			if err := c.SignalClaimPendingAllocation(pending.UID, pending); err != nil {
				t.Errorf("concurrent in-flight add: %v", err)
				return
			}
			c.MaybeRemoveClaimPendingAllocation(pending.UID, true)
		}
	})
	for range 2 {
		wg.Go(func() {
			<-start
			for range 80 {
				state, err := c.GatherAllocatedState()
				if errors.Is(err, errClaimTrackerConcurrentModification) {
					if state != nil {
						t.Error("revision-invalid concurrent snapshot is non-nil")
					}
					continue
				}
				if err != nil || state == nil {
					t.Errorf("concurrent GatherAllocatedState() = %#v, %v", state, err)
					continue
				}
				// An owner is installed before extraction begins. Such a
				// snapshot must exclude its pool until the lock is ready.
				for device := range state.AllocatedSharedDeviceIDs {
					if _, found := state.DeviceSharingAffinity.DeviceStates[device]; !found {
						pool := schedulerapi.PoolID{Driver: device.Driver.String(), Pool: device.Pool.String()}
						if !state.DeviceSharingAffinity.StalePools.Has(pool) {
							t.Errorf("shared device %s has neither affinity state nor pool exclusion in the same snapshot", device)
						}
					}
				}
				for device, lock := range state.DeviceSharingAffinity.DeviceStates {
					if !state.AllocatedSharedDeviceIDs.Has(device) {
						t.Errorf("affinity device %s is absent from the same snapshot's shared IDs", device)
					}
					if lock.Status != structured.AffinityStatusLocked || !maps.Equal(lock.LockedAffinity, map[string]string{"subnet": "blue"}) {
						t.Errorf("compatible concurrent allocation has invalid lock: %#v", lock)
					}
					assertSharingAffinityAccountingCapacity(t, state, device, "5")
				}
			}
		})
	}
	close(start)
	wg.Wait()
	state := gatherSharingAffinityAccounting(t, c)
	assertSharingAffinityAccountingState(t, state, sharingAffinityAccountingDevice(anchor), structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
	c.allocatedDevices.onDelete(anchor)
	state = gatherSharingAffinityAccounting(t, c)
	if len(state.DeviceSharingAffinity.DeviceStates) != 0 || len(state.AllocatedSharedDeviceIDs) != 0 || len(state.AggregatedCapacity) != 0 {
		t.Errorf("concurrent accounting did not drain: %#v", state)
	}
}

// Use the real constructor, with no slice tracker that extraction could access.
func TestSharingAffinityGateDisabled(t *testing.T) {
	for _, capacityEnabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("capacity-enabled=%t", capacityEnabled), func(t *testing.T) {
			setSharingAffinityAccountingGates(t, false, capacityEnabled)
			logger, ctx := ktesting.NewTestContext(t)
			claims := assumecache.NewAssumeCache(logger, nil, "sharing-affinity gate-off", "", nil)
			factory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			manager := NewDRAManager(ctx, claims, nil, factory)
			c := manager.resourceClaimTracker
			if c.affinity != nil || c.allocatedDevices.affinityAcct != nil || c.allocatedDevices.enabledSharingAffinity {
				t.Fatal("gate-off constructor initialized affinity accounting")
			}
			claim := sharingAffinityAccountingClaim("gate-off", "blue")
			claim.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`not JSON`)
			signalSharingAffinityAccounting(t, c, claim)
			assumecache.UpdateTestObject(claims, claim)
			state := gatherSharingAffinityAccounting(t, c)
			device := sharingAffinityAccountingDevice(claim)
			if state.DeviceSharingAffinity.DeviceStates != nil || c.inFlightAllocations[claim.UID].affinity != nil {
				t.Error("gate-off path evaluated or retained affinity contributions")
			}
			owners, _ := c.allocatedDevices.affinitySnapshot()
			if owners != nil || c.allocatedDevices.claimAffinity != nil || c.allocatedDevices.affinityDevices != nil {
				t.Error("gate-off path allocated affinity ownership maps")
			}
			if capacityEnabled {
				if len(state.AllocatedDevices) != 0 || !state.AllocatedSharedDeviceIDs.Equal(sets.New(device)) {
					t.Errorf("gate-off shared legacy sets = %v / %v", state.AllocatedDevices, state.AllocatedSharedDeviceIDs)
				}
				assertSharingAffinityAccountingCapacity(t, state, device, "10")
			} else if !state.AllocatedDevices.Equal(sets.New(device)) || len(state.AllocatedSharedDeviceIDs) != 0 || len(state.AggregatedCapacity) != 0 {
				t.Errorf("capacity-off dedicated legacy snapshot = %#v", state)
			}
		})
	}
}
