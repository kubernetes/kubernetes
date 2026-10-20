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
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

type reconstructionSliceLister struct {
	list func() ([]*resourceapi.ResourceSlice, error)
}

func (l reconstructionSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	return l.list()
}

// AssumeCache returns its upstream registration to the manager, so startup can
// be held independently of slice delivery without replacing a global sync hook.
type reconstructionClaimInformer struct {
	testInformer
	synced   chan struct{}
	observed chan struct{}
	once     sync.Once
}

func (i *reconstructionClaimInformer) AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error) {
	i.handler = handler
	return i, nil
}

func (i *reconstructionClaimInformer) HasSynced() bool {
	select {
	case <-i.synced:
		i.once.Do(func() { close(i.observed) })
		return true
	default:
		return false
	}
}

func (i *reconstructionClaimInformer) HasSyncedChecker() cache.DoneChecker { return i }
func (i *reconstructionClaimInformer) Done() <-chan struct{}               { return i.synced }
func (i *reconstructionClaimInformer) Name() string                        { return "reconstruction claims" }

func reconstructionPool(t *testing.T, results []poolExtraction, key poolName) poolExtraction {
	t.Helper()
	var matching []poolExtraction
	for _, result := range results {
		if result.pool == key {
			matching = append(matching, result)
		}
	}
	if len(matching) != 1 {
		t.Fatalf("pool %v extraction count = %d, want 1; results = %#v", key, len(matching), results)
	}
	return matching[0]
}

func reconstructionAssertContribution(t *testing.T, got affinityContribution, claim *resourceapi.ResourceClaim, result resourceapi.DeviceRequestAllocationResult, status structured.AffinityStatus, values map[string]string) {
	t.Helper()
	device := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
	if got.device != device || got.id != (affinityContributionID{claimUID: claim.UID, device: device, shareID: *result.ShareID}) ||
		got.pool.Driver != result.Driver || got.pool.Name != result.Pool ||
		got.request != result.Request || got.status != status || !maps.Equal(got.values, values) {
		t.Errorf("contribution = %#v, want claim %s request %s device %s status %v values %v", got, claim.UID, result.Request, device, status, values)
	}
	if status != structured.AffinityStatusLocked && got.values != nil {
		t.Errorf("unknown contribution retains a lock map: %v", got.values)
	}
}

// Exercise extraction separately from real cached-claim replay.
func TestSharingAffinityReconstruction(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	_, ctx := ktesting.NewTestContext(t)
	key := poolName{accountingDriver, accountingPool}
	blue := map[string]string{"subnet": "blue"}

	t.Run("extract-no-allocation", func(t *testing.T) {
		acct := newSharingAffinityAccounting()
		acct.slices = reconstructionSliceLister{list: func() ([]*resourceapi.ResourceSlice, error) {
			t.Error("empty claim extraction must not list ResourceSlices")
			return nil, nil
		}}
		unallocated := sharingAffinityAccountingClaim("unallocated", "blue")
		unallocated.Status.Allocation = nil
		for _, claim := range []*resourceapi.ResourceClaim{nil, unallocated} {
			if got := acct.extractByPool(ctx, claim); len(got) != 0 {
				t.Errorf("claim without allocation produced pool extraction: %#v", got)
			}
		}
		var disabled *affinityAccounting
		if got := disabled.extractByPool(ctx, sharingAffinityAccountingClaim("disabled", "blue")); len(got) != 0 {
			t.Errorf("disabled accounting produced pool extraction: %#v", got)
		}
	})

	t.Run("extract-single-list-multi-device-request-pool", func(t *testing.T) {
		acct := newSharingAffinityAccounting()
		all := acct.slices.(accountingSliceLister).slices
		other := all[0].DeepCopy()
		other.Name, other.Spec.Pool.Name, other.Spec.Pool.ResourceSliceCount = "other-pool", "other-pool", 1
		other.Spec.Pool.Generation = 7
		all = append(all, other)
		otherDriver := other.DeepCopy()
		otherDriver.Name, otherDriver.Spec.Driver, otherDriver.Spec.Pool.Name = "other-driver", "other.example.com", accountingPool
		all = append(all, otherDriver)
		// A driver's device attribute must not override the claim's opaque config.
		all[1].Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"subnet": {StringValue: ptr.To("attribute-is-not-config")},
		}
		claim := sharingAffinityAccountingClaim("multi", "blue")
		claim.Spec.Devices.Requests = []resourceapi.DeviceRequest{
			{Name: "blue", Exactly: &resourceapi.ExactDeviceRequest{DeviceClassName: "network"}},
			{Name: "red", Exactly: &resourceapi.ExactDeviceRequest{DeviceClassName: "network"}},
		}
		claim.Spec.Devices.Config[0].Requests = []string{"blue"}
		red := sharingAffinityAccountingClaim("red-config", "red").Spec.Devices.Config[0]
		red.Requests = []string{"red"}
		claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, red)
		otherConfig := sharingAffinityAccountingClaim("other-driver-config", "blue").Spec.Devices.Config[0]
		otherConfig.Opaque.Driver = otherDriver.Spec.Driver
		claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, otherConfig)
		first := &claim.Status.Allocation.Devices.Results[0]
		first.Request = "blue"
		second := *first
		second.Request, second.Device, second.ShareID = "red", "red-device", ptr.To(types.UID("red-share"))
		third := *first
		third.Pool, third.Device, third.ShareID = other.Spec.Pool.Name, "other-device", ptr.To(types.UID("other-share"))
		fourth := *first
		fourth.Driver, fourth.ShareID = otherDriver.Spec.Driver, ptr.To(types.UID("other-driver-share"))
		claim.Status.Allocation.Devices.Results = append(claim.Status.Allocation.Devices.Results, second, third, fourth)
		before := claim.DeepCopy()
		calls := 0
		acct.slices = reconstructionSliceLister{list: func() ([]*resourceapi.ResourceSlice, error) {
			calls++
			return all, nil
		}}
		got := acct.extractByPool(ctx, claim)
		if calls != 1 {
			t.Errorf("cached slice lists = %d, want exactly one for all pools and requests", calls)
		}
		if len(got) != 3 {
			t.Errorf("pool extractions = %d, want 3", len(got))
		}
		for _, pool := range []poolName{key, {accountingDriver, other.Spec.Pool.Name}, {otherDriver.Spec.Driver, accountingPool}} {
			extracted := reconstructionPool(t, got, pool)
			if extracted.pending || extracted.reason != "" || extracted.view != metadataView(all, pool) {
				t.Errorf("complete pool result = %#v, want settled with same-list provenance", extracted)
			}
			var wantDevices []structured.DeviceID
			var wantResults []resourceapi.DeviceRequestAllocationResult
			for _, result := range claim.Status.Allocation.Devices.Results {
				if result.Driver == pool.driver && result.Pool == pool.pool {
					wantResults = append(wantResults, result)
					wantDevices = append(wantDevices, structured.MakeDeviceID(result.Driver, result.Pool, result.Device))
				}
			}
			if !sets.New(extracted.devices...).Equal(sets.New(wantDevices...)) || len(extracted.contributions) != len(wantResults) {
				t.Fatalf("pool %v devices/contributions = %v/%v, want devices %v and %d contributions", pool, extracted.devices, extracted.contributions, wantDevices, len(wantResults))
			}
			for _, contribution := range extracted.contributions {
				index := slices.IndexFunc(wantResults, func(result resourceapi.DeviceRequestAllocationResult) bool {
					return structured.MakeDeviceID(result.Driver, result.Pool, result.Device) == contribution.device
				})
				if index < 0 {
					t.Fatalf("unexpected contribution: %#v", contribution)
				}
				values := blue
				if wantResults[index].Request == "red" {
					values = map[string]string{"subnet": "red"}
				}
				reconstructionAssertContribution(t, contribution, claim, wantResults[index], structured.AffinityStatusLocked, values)
				if contribution.pool.Generation != extracted.view.generation {
					t.Errorf("contribution generation %d differs from extraction view %d", contribution.pool.Generation, extracted.view.generation)
				}
			}
		}
		if !reflect.DeepEqual(claim, before) {
			t.Error("extraction mutated cached claim")
		}
	})

	for _, mode := range []string{"incomplete", "listError", "highest-generation-incomplete"} {
		t.Run("extract-pending-"+mode, func(t *testing.T) {
			acct := newSharingAffinityAccounting()
			all := acct.slices.(accountingSliceLister).slices
			claim := sharingAffinityAccountingClaim("pending", "blue")
			other := all[0].DeepCopy()
			other.Name, other.Spec.Pool.Name, other.Spec.Pool.ResourceSliceCount = "ready", "ready", 1
			result := claim.Status.Allocation.Devices.Results[0]
			result.Pool, result.ShareID = "ready", ptr.To(types.UID("ready-share"))
			claim.Status.Allocation.Devices.Results = append(claim.Status.Allocation.Devices.Results, result)
			var listErr error
			switch mode {
			case "incomplete":
				all = all[:1]
			case "highest-generation-incomplete":
				next := all[0].DeepCopy()
				next.Name, next.Spec.Pool.Generation = "next-generation", 2
				all = append(all, next)
			case "listError":
				listErr = errors.New("cached slice listing unavailable")
			}
			all = append(all, other)
			acct.slices = metadataSliceList{slices: all, err: listErr}
			got := acct.extractByPool(ctx, claim)
			for _, pool := range []poolName{key, {accountingDriver, "ready"}} {
				extracted := reconstructionPool(t, got, pool)
				wantPending := pool == key || listErr != nil
				if extracted.pending != wantPending {
					t.Errorf("pool %v pending = %t, want %t", pool, extracted.pending, wantPending)
				}
				if listErr == nil && extracted.view != metadataView(all, pool) {
					t.Errorf("pool %v extraction view = %#v, want same-list view %#v", pool, extracted.view, metadataView(all, pool))
				}
				if wantPending {
					reason := "incomplete"
					if listErr != nil {
						reason = "listError"
					}
					if extracted.reason != reason || len(extracted.contributions) != 0 {
						t.Errorf("transient blocker = %#v, want reason %s and no sticky contribution", extracted, reason)
					}
				} else if len(extracted.contributions) != 1 || extracted.contributions[0].status != structured.AffinityStatusLocked {
					t.Errorf("incomplete sibling quarantined ready pool: %#v", extracted)
				}
				if !reflect.DeepEqual(extracted.devices, []structured.DeviceID{structured.MakeDeviceID(pool.driver, pool.pool, accountingDevice)}) {
					t.Errorf("pending/settled pool lost allocation device: %#v", extracted)
				}
			}
		})
	}

	t.Run("extract-admin-dedicated-share-id-only-unconstrained", func(t *testing.T) {
		acct := newSharingAffinityAccounting()
		all := acct.slices.(accountingSliceLister).slices
		plain := all[1].DeepCopy()
		plain.Name, plain.Spec.Pool.Name, plain.Spec.Pool.ResourceSliceCount = "plain", "plain", 1
		acct.slices = accountingSliceLister{slices: append(all, plain)}
		claim := sharingAffinityAccountingClaim("ownership", "blue")
		result := &claim.Status.Allocation.Devices.Results[0]
		result.ConsumedCapacity = nil
		dedicated, admin, unconstrained := *result, *result, *result
		dedicated.Device, dedicated.ShareID = "dedicated", nil
		admin.Pool, admin.Device, admin.AdminAccess = "admin-only", "admin", ptr.To(true)
		unconstrained.Pool, unconstrained.Device = "plain", "plain"
		claim.Status.Allocation.Devices.Results = append(claim.Status.Allocation.Devices.Results, dedicated, admin, unconstrained)
		got := acct.extractByPool(ctx, claim)
		if len(got) != 2 {
			t.Errorf("extraction count = %d, want constrained and unconstrained pools only", len(got))
		}
		extracted := reconstructionPool(t, got, key)
		if len(extracted.contributions) != 1 {
			t.Fatalf("affinity owners = %v, want only share-ID-only result", extracted.contributions)
		}
		reconstructionAssertContribution(t, extracted.contributions[0], claim, *result, structured.AffinityStatusLocked, blue)
		if !sets.New(extracted.devices...).Equal(sets.New(sharingAffinityAccountingDevice(claim), structured.MakeDeviceID(accountingDriver, accountingPool, "dedicated"))) {
			t.Errorf("pool device tracking lost dedicated/nil-capacity ownership: %v", extracted.devices)
		}
		plainResult := reconstructionPool(t, got, poolName{accountingDriver, "plain"})
		if plainResult.pending || len(plainResult.contributions) != 0 || len(plainResult.devices) != 1 {
			t.Errorf("no-affinity result = %#v, want tracked device without affinity", plainResult)
		}
	})

	t.Run("extract-immutable-results-and-inputs", func(t *testing.T) {
		acct := newSharingAffinityAccounting()
		all := acct.slices.(accountingSliceLister).slices
		beforeSlices := make([]*resourceapi.ResourceSlice, len(all))
		for i, slice := range all {
			beforeSlices[i] = slice.DeepCopy()
		}
		claim := sharingAffinityAccountingClaim("immutable", "blue")
		beforeClaim := claim.DeepCopy()
		first := reconstructionPool(t, acct.extractByPool(ctx, claim), key)
		if len(first.contributions) != 1 || len(first.devices) != 1 {
			t.Fatalf("extraction = %#v, want one device and one Locked owner", first)
		}
		first.contributions[0].values["subnet"] = "caller-mutation"
		first.devices[0] = structured.MakeDeviceID("other.example.com", "other", "other")
		second := reconstructionPool(t, acct.extractByPool(ctx, claim), key)
		if len(second.contributions) != 1 {
			t.Fatalf("second extraction contributions = %#v, want one Locked owner", second.contributions)
		}
		reconstructionAssertContribution(t, second.contributions[0], claim, claim.Status.Allocation.Devices.Results[0], structured.AffinityStatusLocked, blue)
		if !reflect.DeepEqual(all, beforeSlices) || !reflect.DeepEqual(claim, beforeClaim) {
			t.Error("extraction or caller mutation changed cached claim/slice inputs")
		}
	})

	for _, source := range []string{"claim", "allocation-claim", "allocation-class", "other-driver", "sibling-subrequest", "parent-request"} {
		t.Run("extract-config-source-"+source, func(t *testing.T) {
			acct := newSharingAffinityAccounting()
			claim := sharingAffinityAccountingClaim("config-source", "blue")
			claim.Status.Allocation.Devices.Results[0].Request = "req/sub"
			config := claim.Spec.Devices.Config[0]
			wantStatus, wantValues := structured.AffinityStatusUnreconstructable, map[string]string(nil)
			switch source {
			case "claim":
				claim.Spec.Devices.Config[0].Requests = []string{"req/sub"}
				wantStatus, wantValues = structured.AffinityStatusLocked, blue
			case "parent-request":
				claim.Spec.Devices.Config[0].Requests = []string{"req"}
				wantStatus, wantValues = structured.AffinityStatusLocked, blue
			case "sibling-subrequest":
				claim.Spec.Devices.Config[0].Requests = []string{"req/sibling"}
			case "other-driver":
				claim.Spec.Devices.Config[0].Opaque.Driver = "unrelated.example.com"
			default:
				claim.Spec.Devices.Config = nil
				allocationSource := resourceapi.AllocationConfigSourceClaim
				if source == "allocation-class" {
					allocationSource = resourceapi.AllocationConfigSourceClass
				}
				claim.Status.Allocation.Devices.Config = []resourceapi.DeviceAllocationConfiguration{{
					Source: allocationSource, DeviceConfiguration: config.DeviceConfiguration,
				}}
			}
			extracted := reconstructionPool(t, acct.extractByPool(ctx, claim), key)
			if extracted.pending || len(extracted.contributions) != 1 {
				t.Fatalf("config-source extraction = %#v, want one settled contribution", extracted)
			}
			reconstructionAssertContribution(t, extracted.contributions[0], claim, claim.Status.Allocation.Devices.Results[0], wantStatus, wantValues)
		})
	}

	for _, count := range []int{1, 2} {
		t.Run(fmt.Sprintf("cached-replay-%d-owners", count), func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			for i := 0; i < count; i++ {
				// No Pod or ReservedFor entry is required for an allocated claim to own a lock.
				assumecache.AddTestObject(c.cache, sharingAffinityAccountingClaim(types.UID(fmt.Sprintf("owner-%d", i)), "blue"))
			}
			lists := 0
			original := c.affinity.slices
			c.affinity.slices = reconstructionSliceLister{list: func() ([]*resourceapi.ResourceSlice, error) {
				lists++
				if !c.allocatedDevices.mutex.TryLock() {
					t.Error("cached slice listing called under allocatedDevices.mutex")
				} else {
					c.allocatedDevices.mutex.Unlock()
				}
				if !c.inFlightMutex.TryLock() {
					t.Error("cached slice listing called under inFlightMutex")
				} else {
					c.inFlightMutex.Unlock()
				}
				return original.ListPatchedResourceSlices()
			}}
			c.allocatedDevices.observedAfterComputeHook = func() {
				if !c.allocatedDevices.mutex.TryLock() {
					t.Error("extraction completion called under allocatedDevices.mutex")
				} else {
					c.allocatedDevices.mutex.Unlock()
				}
				if !c.inFlightMutex.TryLock() {
					t.Error("extraction completion called under inFlightMutex")
				} else {
					c.inFlightMutex.Unlock()
				}
			}
			var celPolls, celDone atomic.Int64
			c.affinity.evalContextHook = func(ctx context.Context) context.Context {
				return lockProbeContext{Context: ctx, done: &celDone, probe: func() {
					celPolls.Add(1)
					if !c.allocatedDevices.mutex.TryLock() {
						t.Error("CEL evaluation ran under allocatedDevices.mutex")
					} else {
						c.allocatedDevices.mutex.Unlock()
					}
					if !c.inFlightMutex.TryLock() {
						t.Error("CEL evaluation ran under inFlightMutex")
					} else {
						c.inFlightMutex.Unlock()
					}
				}}
			}
			registration := c.cache.AddEventHandler(c.allocatedDevices.handlers())
			if celPolls.Load() == 0 {
				t.Error("CEL evaluation never polled the probe context; lock boundary unverified")
			}
			// Only cel-go's ExecutionFrame.SetContext (inside ContextEval) calls
			// Done; extractor-level ctxError checks only call Err.
			if celDone.Load() == 0 {
				t.Error("CEL interpreter never entered ContextEval with the probe context; in-CEL lock boundary unverified")
			}
			if !registration.HasSynced() || lists != count {
				t.Errorf("cached replay synced=%t lists=%d, want synced and %d lists", registration.HasSynced(), lists, count)
			}
			device := structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, count, structured.AffinityStatusLocked, blue)
			state := gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingCapacity(t, state, device, fmt.Sprint(count*5))
			state.DeviceSharingAffinity.DeviceStates[device].LockedAffinity["subnet"] = "mutated"
			delete(state.DeviceSharingAffinity.DeviceStates, device)
			state.AggregatedCapacity[device][accountingCapacity].Set(999)
			again := gatherSharingAffinityAccounting(t, c)
			assertSharingAffinityAccountingState(t, again, device, structured.AffinityStatusLocked, blue)
			assertSharingAffinityAccountingCapacity(t, again, device, fmt.Sprint(count*5))
		})
	}

	for _, reversed := range []bool{false, true} {
		t.Run(fmt.Sprintf("conflicting-historical-owners-reversed=%t", reversed), func(t *testing.T) {
			c := newSharingAffinityAccountingTracker(t, true)
			a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "red")
			owners := []*resourceapi.ResourceClaim{a, b}
			if reversed {
				slices.Reverse(owners)
			}
			// Fixture-seeded historical conflict, not a state a correct allocator creates.
			c.cache.AddEventHandler(c.allocatedDevices.handlers())
			for _, owner := range owners {
				assumecache.AddTestObject(c.cache, owner)
			}
			device := sharingAffinityAccountingDevice(a)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 2, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "10")
			assumecache.DeleteTestObject(c.cache, a)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusUnreconstructable, nil)
			assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
			assumecache.DeleteTestObject(c.cache, b)
			assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusClean, nil)
			fresh := sharingAffinityAccountingClaim("fresh", "green")
			assumecache.AddTestObject(c.cache, fresh)
			assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "green"})
		})
	}

	for _, claimsFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("manager-startup-claims-first=%t", claimsFirst), func(t *testing.T) {
			logger, baseCtx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithTimeout(baseCtx, 10*time.Second)
			defer cancel()
			objects := []runtime.Object{}
			for _, slice := range newSharingAffinityAccounting().slices.(accountingSliceLister).slices {
				objects = append(objects, slice)
			}
			client := fake.NewClientset(objects...)
			factory := informers.NewSharedInformerFactory(client, 0)
			tracker, err := resourceslicetracker.StartTracker(ctx, resourceslicetracker.Options{
				SliceInformer: factory.Resource().V1().ResourceSlices(), KubeClient: client,
			})
			if err != nil {
				t.Fatalf("StartTracker: %v", err)
			}
			claims := assumecache.NewAssumeCache(logger, nil, "startup claims", "", nil)
			claim := sharingAffinityAccountingClaim("startup-owner", "blue")
			before := claim.DeepCopy()
			if claimsFirst {
				assumecache.AddTestObject(claims, claim)
			}
			manager := NewDRAManager(ctx, claims, tracker, factory)
			c := manager.resourceClaimTracker
			entered, release, exited := make(chan struct{}), make(chan struct{}), make(chan struct{})
			// Install the instance seam before starting the informer; no global hook is replaced.
			c.metadataSyncedHook = func() {
				close(entered)
				select {
				case <-release:
				case <-ctx.Done():
				}
				close(exited)
			}
			factory.Start(ctx.Done())
			defer factory.Shutdown()
			defer cancel()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("metadata handler registration did not reach its startup barrier")
			}
			device := sharingAffinityAccountingDevice(claim)
			// If metadata arrived first, deliver the claim with the final startup barrier held.
			if !claimsFirst {
				assumecache.AddTestObject(claims, claim)
			}
			state, metadata, err := c.gatherAllocatedStateWithMetadata()
			if err != nil {
				t.Fatalf("gather during startup: %v", err)
			}
			if metadata.synced {
				t.Error("metadata became ready before its registration barrier was released")
			}
			if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}) {
				t.Error("startup owner pool not marked stale while registration is held")
			}
			c.allocatedDevices.mutex.RLock()
			fenced := c.allocatedDevices.fenced.Has(device)
			c.allocatedDevices.mutex.RUnlock()
			if fenced {
				t.Error("transient startup ordering created a sticky device fence")
			}
			assertSharingAffinityAccountingCapacity(t, state, device, "5")
			close(release)
			<-exited
			// No subsequent claim or slice event: readiness alone must finish reconstruction.
			if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
				state, metadata, err := c.gatherAllocatedStateWithMetadata()
				if err != nil {
					return false, nil
				}
				record := state.DeviceSharingAffinity.DeviceStates[device]
				return metadata.synced && !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}) &&
					record.Status == structured.AffinityStatusLocked && maps.Equal(record.LockedAffinity, blue), nil
			}); err != nil {
				t.Errorf("startup owner did not recover to Locked without another event: %v; state=%#v", err, gatherSharingAffinityAccounting(t, c).DeviceSharingAffinity.DeviceStates)
			}
			assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
			if !reflect.DeepEqual(claim, before) {
				t.Error("startup reconstruction changed the bound claim")
			}
		})
	}

	for _, lastSync := range []string{"metadata", "claims"} {
		for _, intermediate := range []bool{false, true} {
			name := "sync-order-last-" + lastSync
			if intermediate {
				name = "startup-intermediate-generation-last-" + lastSync
			}
			t.Run(name, func(t *testing.T) {
				reconstructionStartupRegistration(t, lastSync, intermediate)
			})
		}
	}

	t.Run("real-manager-cached-replay-no-api-calls", func(t *testing.T) {
		logger, baseCtx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithTimeout(baseCtx, 10*time.Second)
		defer cancel()
		objects := []runtime.Object{}
		for _, slice := range newSharingAffinityAccounting().slices.(accountingSliceLister).slices {
			objects = append(objects, slice)
		}
		client := fake.NewClientset(objects...)
		factory := informers.NewSharedInformerFactory(client, 0)
		tracker, err := resourceslicetracker.StartTracker(ctx, resourceslicetracker.Options{
			SliceInformer: factory.Resource().V1().ResourceSlices(), KubeClient: client,
		})
		if err != nil {
			t.Fatalf("StartTracker: %v", err)
		}
		factory.Start(ctx.Done())
		defer factory.Shutdown()
		defer cancel()
		if !cache.WaitForCacheSync(ctx.Done(), tracker.HasSynced) {
			t.Fatal("cached ResourceSlices failed to sync")
		}
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			for _, action := range client.Actions() {
				if action.Matches("watch", "resourceslices") {
					return true, nil
				}
			}
			return false, nil
		}); err != nil {
			t.Fatalf("initial ResourceSlice watch did not start: %v", err)
		}
		claims := assumecache.NewAssumeCache(logger, nil, "reconstruction claims", "", nil)
		claim := sharingAffinityAccountingClaim("already-allocated", "blue")
		assumecache.AddTestObject(claims, claim)
		before := len(client.Actions())
		manager := NewDRAManager(ctx, claims, tracker, factory)
		c := manager.resourceClaimTracker
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
			c.allocatedDevices.mutex.RLock()
			defer c.allocatedDevices.mutex.RUnlock()
			return c.metadata.synced, nil
		}); err != nil {
			t.Fatalf("manager metadata registration did not sync: %v", err)
		}
		device := sharingAffinityAccountingDevice(claim)
		// Registration sync may enqueue replay without completing it.
		reconcileWaitState(t, ctx, c, device, structured.AffinityStatusLocked, blue)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, blue)
		assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
		if after := len(client.Actions()); after != before {
			t.Errorf("manager replay added API actions: %v", client.Actions()[before:])
		}
		cached, err := claims.Get(claim.Namespace + "/" + claim.Name)
		if err != nil || !reflect.DeepEqual(cached, claim) {
			t.Errorf("replay changed cached allocation: %#v, %v", cached, err)
		}
	})
}

func reconstructionStartupRegistration(t *testing.T, lastSync string, intermediate bool) string {
	t.Helper()
	setSharingAffinityAccountingGates(t, true, true)
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(5)))
	ctx, cancel := context.WithTimeout(klog.NewContext(context.Background(), logger), 15*time.Second)
	defer cancel()
	key := poolName{accountingDriver, accountingPool}
	all := newSharingAffinityAccounting().slices.(accountingSliceLister).slices
	objects := make([]runtime.Object, len(all))
	for i, slice := range all {
		objects[i] = slice
	}
	client := fake.NewClientset(objects...)
	factory := informers.NewSharedInformerFactory(client, 0)
	tracker, err := resourceslicetracker.StartTracker(ctx, resourceslicetracker.Options{
		SliceInformer: factory.Resource().V1().ResourceSlices(), KubeClient: client,
	})
	if err != nil {
		t.Fatalf("StartTracker: %v", err)
	}
	claimInformer := &reconstructionClaimInformer{synced: make(chan struct{}), observed: make(chan struct{})}
	claims := assumecache.NewAssumeCache(logger, claimInformer, "startup registrations", "", nil)
	manager := NewDRAManager(ctx, claims, tracker, factory)
	c := manager.resourceClaimTracker
	metadataEntered, metadataRelease := make(chan struct{}), make(chan struct{})
	c.metadataSyncedHook = func() {
		close(metadataEntered)
		select {
		case <-metadataRelease:
		case <-ctx.Done():
		}
	}
	factory.Start(ctx.Done())
	defer factory.Shutdown()
	defer cancel()
	select {
	case <-metadataEntered:
	case <-ctx.Done():
		t.Fatal("metadata registration did not reach the held startup barrier")
	}
	claim := sharingAffinityAccountingClaim("initial-owner", "blue")
	before := claim.DeepCopy()
	device := sharingAffinityAccountingDevice(claim)
	claimInformer.handler.OnAdd(claim, true)

	checkHeld := func(phase string) {
		t.Helper()
		state := gatherSharingAffinityAccounting(t, c)
		if !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}) {
			t.Errorf("%s: pool became ready with a startup registration still held", phase)
		}
		if got := state.DeviceSharingAffinity.DeviceStates[device]; got.Status == structured.AffinityStatusUnreconstructable {
			t.Errorf("%s: transient startup metadata established ambiguity: %#v", phase, got)
		}
		c.allocatedDevices.mutex.RLock()
		fenced := c.allocatedDevices.fenced.Has(device)
		c.allocatedDevices.mutex.RUnlock()
		if fenced {
			t.Errorf("%s: startup owner was fenced before both registrations synced", phase)
		}
		assertSharingAffinityAccountingCapacity(t, state, device, "5")
	}
	waitMetadata := func() {
		t.Helper()
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
			c.allocatedDevices.mutex.RLock()
			defer c.allocatedDevices.mutex.RUnlock()
			return c.metadata.synced, nil
		}); err != nil {
			t.Fatalf("released metadata registration never synced: %v", err)
		}
	}
	waitClaims := func() {
		t.Helper()
		select {
		case <-claimInformer.observed:
		case <-time.After(time.Second):
			t.Error("manager never checked the released claim-handler registration")
		}
	}
	checkHeld("initial generation")
	if lastSync == "claims" {
		close(metadataRelease)
		waitMetadata()
	} else {
		close(claimInformer.synced)
		waitClaims()
	}
	checkHeld("first registration synced")

	wantValues := map[string]string{"subnet": "blue"}
	if intermediate {
		var next []*resourceapi.ResourceSlice
		for _, slice := range all {
			updated := slice.DeepCopy()
			updated.Spec.Pool.Generation++
			for i := range updated.Spec.SharingAffinity {
				updated.Spec.SharingAffinity[i].Expression = `"green"`
			}
			next = append(next, updated)
			if _, err := client.ResourceV1().ResourceSlices().Update(ctx, updated, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("publish final startup generation: %v", err)
			}
		}
		expected := metadataView(next, key)
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
			c.allocatedDevices.mutex.RLock()
			defer c.allocatedDevices.mutex.RUnlock()
			return sameMetadata(c.metadata.pools[key], expected), nil
		}); err != nil {
			t.Fatalf("final generation not published before last registration release: %v", err)
		}
		checkHeld("final generation before startup")
		wantValues = map[string]string{"subnet": "green"}
	}
	if lastSync == "claims" {
		close(claimInformer.synced)
		waitClaims()
	} else {
		close(metadataRelease)
		waitMetadata()
	}
	// Releasing the last registration is the only remaining event.
	if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 2*time.Second, true, func(context.Context) (bool, error) {
		state, err := c.GatherAllocatedState()
		if err != nil {
			return false, nil
		}
		if state.DeviceSharingAffinity.DeviceStates[device].Status == structured.AffinityStatusUnreconstructable {
			return false, fmt.Errorf("startup produced sticky ambiguity instead of current-generation reconstruction")
		}
		return !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: accountingDriver, Pool: accountingPool}) &&
			state.DeviceSharingAffinity.DeviceStates[device].Status == structured.AffinityStatusLocked &&
			maps.Equal(state.DeviceSharingAffinity.DeviceStates[device].LockedAffinity, wantValues), nil
	}); err != nil {
		t.Errorf("last registration release did not reconstruct current map %v: %v", wantValues, err)
	}
	assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
	if !reflect.DeepEqual(claim, before) {
		t.Error("startup changed the cached allocated claim")
	}
	logs := logger.GetSink().(ktesting.Underlier).GetBuffer().String()
	if strings.Contains(logs, "metadataChanged") || strings.Contains(logs, `reason="lateCommit"`) {
		t.Errorf("startup metadata evolution was logged as semantic ambiguity:\n%s", logs)
	}
	return logs
}

// Unknown owners quarantine only their device and remain capacity-accounted.
func TestSharingAffinityRecovery(t *testing.T) {
	setSharingAffinityAccountingGates(t, true, true)
	_, ctx := ktesting.NewTestContext(t)
	for _, mode := range []string{"legacy-missing-config", "unsatisfied-extractor", "extraction-error", "mixed-valid-and-unknown"} {
		t.Run(mode, func(t *testing.T) {
			unknown := sharingAffinityAccountingClaim("unknown", "blue")
			switch mode {
			case "legacy-missing-config", "mixed-valid-and-unknown":
				unknown.Spec.Devices.Config = nil
			case "unsatisfied-extractor":
				unknown.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`{"apiVersion":"config.example.com/v1","kind":"Network"}`)
			case "extraction-error":
				unknown.Spec.Devices.Config[0].Opaque.Parameters.Raw = []byte(`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":42}`)
			}
			t.Run("cached-replay-drain", func(t *testing.T) {
				c := newSharingAffinityAccountingTracker(t, true)
				good := sharingAffinityAccountingClaim("good", "blue")
				owners := []*resourceapi.ResourceClaim{unknown, good}
				if mode == "mixed-valid-and-unknown" {
					second := unknown.DeepCopy()
					second.Name, second.UID = "second-unknown", "second-unknown"
					second.Status.Allocation.Devices.Results[0].ShareID = ptr.To(types.UID("second-share"))
					owners = []*resourceapi.ResourceClaim{unknown, second, good}
				}
				sibling := sharingAffinityAccountingClaim("sibling", "blue")
				sibling.Status.Allocation.Devices.Results[0].Device = "sibling"
				c.cache.AddEventHandler(c.allocatedDevices.handlers())
				originals := make([]*resourceapi.ResourceClaim, 0, len(owners)+1)
				for _, claim := range append(slices.Clone(owners), sibling) {
					originals = append(originals, claim.DeepCopy())
					assumecache.AddTestObject(c.cache, claim)
				}
				device, siblingDevice := sharingAffinityAccountingDevice(unknown), sharingAffinityAccountingDevice(sibling)
				for i, owner := range owners {
					state := gatherSharingAffinityAccounting(t, c)
					assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, len(owners)-i, structured.AffinityStatusUnreconstructable, nil)
					assertSharingAffinityAccountingCapacity(t, state, device, fmt.Sprint((len(owners)-i)*5))
					assertSharingAffinityAccountingState(t, state, siblingDevice, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					assertSharingAffinityAccountingCapacity(t, state, siblingDevice, "5")
					assertSharingAffinityAccountingState(t, state, structured.MakeDeviceID(accountingDriver, "unused-pool", "unused"), structured.AffinityStatusClean, nil)
					if !state.AllocatedSharedDeviceIDs.Has(device) || !state.AllocatedSharedDeviceIDs.Has(siblingDevice) {
						t.Errorf("quarantine discarded allocated shared devices: %v", state.AllocatedSharedDeviceIDs)
					}
					assumecache.DeleteTestObject(c.cache, owner)
				}
				assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 0, structured.AffinityStatusClean, nil)
				after := gatherSharingAffinityAccounting(t, c)
				if len(after.AggregatedCapacity[device]) != 0 || after.AllocatedSharedDeviceIDs.Has(device) {
					t.Errorf("full drain retained capacity/shared ownership: %#v", after)
				}
				fresh := sharingAffinityAccountingClaim("new-after-drain", "green")
				assumecache.AddTestObject(c.cache, fresh)
				assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "green"})
				for i, claim := range append(slices.Clone(owners), sibling) {
					if !reflect.DeepEqual(claim, originals[i]) {
						t.Errorf("recovery mutated existing claim %s", claim.Name)
					}
				}
			})
			if mode != "mixed-valid-and-unknown" {
				t.Run("pool-extraction-is-semantic-not-pending", func(t *testing.T) {
					extracted := reconstructionPool(t, newSharingAffinityAccounting().extractByPool(ctx, unknown), poolName{accountingDriver, accountingPool})
					if extracted.pending || extracted.reason != "" || len(extracted.contributions) != 1 {
						t.Fatalf("genuine extraction failure = %#v, want settled unknown contribution", extracted)
					}
					reconstructionAssertContribution(t, extracted.contributions[0], unknown, unknown.Status.Allocation.Devices.Results[0], structured.AffinityStatusUnreconstructable, nil)
				})
			}
		})
	}

	t.Run("nil-capacity-owner-prevents-drain", func(t *testing.T) {
		c := newSharingAffinityAccountingTracker(t, true)
		unknown := sharingAffinityAccountingClaim("unknown", "blue")
		unknown.Spec.Devices.Config = nil
		nilCapacity := sharingAffinityAccountingClaim("share-id-only", "blue")
		nilCapacity.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
		c.cache.AddEventHandler(c.allocatedDevices.handlers())
		assumecache.AddTestObject(c.cache, unknown)
		assumecache.AddTestObject(c.cache, nilCapacity)
		assumecache.DeleteTestObject(c.cache, unknown)
		device := sharingAffinityAccountingDevice(unknown)
		state := gatherSharingAffinityAccounting(t, c)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusUnreconstructable, nil)
		if !state.AllocatedSharedDeviceIDs.Has(device) || len(state.AggregatedCapacity[device]) != 0 {
			t.Errorf("share-ID-only owner has incorrect ownership/capacity: %#v", state)
		}
		assumecache.DeleteTestObject(c.cache, nilCapacity)
		assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusClean, nil)
		assumecache.AddTestObject(c.cache, sharingAffinityAccountingClaim("fresh", "red"))
		assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, c), device, structured.AffinityStatusLocked, map[string]string{"subnet": "red"})
	})
}

// lockProbeContext runs probe whenever affinity extraction polls Err, which
// happens inside the per-extractor CEL evaluation loop.
type lockProbeContext struct {
	context.Context
	probe func()
	// done counts Done calls, which come only from the CEL interpreter.
	done *atomic.Int64
}

func (c lockProbeContext) Done() <-chan struct{} {
	if c.done != nil {
		c.done.Add(1)
	}
	c.probe()
	return c.Context.Done()
}

func (c lockProbeContext) Err() error {
	c.probe()
	return c.Context.Err()
}
