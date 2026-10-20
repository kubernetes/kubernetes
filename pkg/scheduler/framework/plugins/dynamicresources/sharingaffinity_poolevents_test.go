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
	"sort"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

func poolEventsGates(t *testing.T, sharing, capacity bool) {
	t.Helper()
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRASharingAffinity:    sharing,
		features.DRAConsumableCapacity: capacity,
	})
}

func poolEventsSlices() []*resourceapi.ResourceSlice {
	slices := newSharingAffinityAccounting().slices.(accountingSliceLister).slices
	for _, slice := range slices {
		slice.UID = types.UID(slice.Name)
		slice.ResourceVersion = "1"
	}
	slices[1].Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
		"units": {Value: resource.MustParse("20")},
	}
	return slices
}

func poolEventsSetup(t *testing.T, slices []*resourceapi.ResourceSlice, claims ...*resourceapi.ResourceClaim) (ktesting.TContext, *testContext) {
	t.Helper()
	poolEventsGates(t, true, true)
	tCtx := ktesting.Init(t)
	objects := make([]runtime.Object, 0, len(slices))
	for _, slice := range slices {
		objects = append(objects, slice)
	}
	node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "pool-events-node"}}
	class := &resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}}
	if len(claims) == 0 {
		claim := sharingAffinityAccountingClaim("readiness", "blue")
		claim.UID, claim.ResourceVersion = "", ""
		claim.Status.Allocation = nil
		claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
		claim.Spec.Devices.Requests[0].Exactly.Count = 1
		claims = []*resourceapi.ResourceClaim{claim}
	}
	var createClaims []*resourceapi.ResourceClaim
	for _, claim := range claims {
		if claim.Status.Allocation != nil {
			// Restart fixtures represent objects already persisted before the
			// new manager starts, not fresh API creates with preassigned UIDs.
			objects = append(objects, claim.DeepCopy())
		} else {
			createClaims = append(createClaims, claim)
		}
	}
	tc := setup(tCtx, nil, []*v1.Node{node}, createClaims, []*resourceapi.DeviceClass{class}, nil, objects,
		feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate), false, nil)
	// Informer readiness does not guarantee that an asynchronously registered
	// metadata handler has published its initial view. PreFilter is the public
	// coherence barrier for a pod that still needs an allocation.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "metadata-readiness", Namespace: claims[0].Namespace, UID: "metadata-readiness"},
		Spec: v1.PodSpec{ResourceClaims: []v1.PodResourceClaim{{
			Name: "resource", ResourceClaimName: ptr.To(claims[0].Name),
		}}},
	}
	var status *fwk.Status
	if err := wait.PollUntilContextTimeout(tCtx, 5*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		a := tc.draManager.resourceClaimTracker.allocatedDevices
		a.mutex.RLock()
		ready := a.startupReady
		a.mutex.RUnlock()
		if !ready {
			return false, nil
		}
		_, status = tc.p.PreFilter(tCtx, framework.NewCycleState(), pod, tc.nodeInfos)
		return status.IsSuccess(), nil
	}); err != nil {
		t.Fatalf("initial metadata did not become coherent: %v, PreFilter status: %v", err, status)
	}
	return tCtx, tc
}

// Observe real tracker callbacks rather than simulating the manager's missing
// metadata handler. The callback only synchronizes the test; it changes no state.
func poolEventsChange(t *testing.T, ctx context.Context, tc *testContext, slice *resourceapi.ResourceSlice, remove bool) {
	t.Helper()
	delivered := make(chan struct{}, 1)
	notify := func(obj interface{}) {
		if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
			obj = tombstone.Obj
		}
		got, ok := obj.(*resourceapi.ResourceSlice)
		if !ok || got.Name != slice.Name || (!remove && !reflect.DeepEqual(got.Spec, slice.Spec)) {
			return
		}
		select {
		case delivered <- struct{}{}:
		default:
		}
	}
	handler := cache.ResourceEventHandlerFuncs{}
	if remove {
		handler.DeleteFunc = notify
	} else {
		handler.AddFunc = notify
		handler.UpdateFunc = func(_, obj interface{}) { notify(obj) }
	}
	if _, err := tc.draManager.resourceSliceLister.tracker.AddEventHandler(handler); err != nil {
		t.Fatalf("register ResourceSlice delivery observer: %v", err)
	}
	client := tc.client.ResourceV1().ResourceSlices()
	if remove {
		if err := client.Delete(ctx, slice.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("delete ResourceSlice %s: %v", slice.Name, err)
		}
	} else {
		current, err := client.Get(ctx, slice.Name, metav1.GetOptions{})
		updated := slice.DeepCopy()
		if apierrors.IsNotFound(err) {
			updated.UID = ""
			updated.ResourceVersion = ""
			if _, err := client.Create(ctx, updated, metav1.CreateOptions{}); err != nil {
				t.Fatalf("create ResourceSlice %s: %v", slice.Name, err)
			}
		} else {
			if err != nil {
				t.Fatalf("get ResourceSlice %s: %v", slice.Name, err)
			}
			updated.UID = current.UID
			updated.ResourceVersion = current.ResourceVersion
			if _, err := client.Update(ctx, updated, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("update ResourceSlice %s: %v", slice.Name, err)
			}
		}
	}
	select {
	case <-delivered:
	case <-ctx.Done():
		t.Fatalf("waiting for ResourceSlice %s: %v", slice.Name, ctx.Err())
	case <-time.After(5 * time.Second):
		t.Fatalf("ResourceSlice %s change was not delivered", slice.Name)
	}
}

func poolEventsAssertState(t *testing.T, ctx context.Context, tc *testContext, device structured.DeviceID, status structured.AffinityStatus, values map[string]string) {
	t.Helper()
	// Tracker handlers are independent; allow the manager to finish its callback.
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		state, err := tc.draManager.resourceClaimTracker.GatherAllocatedState()
		if err != nil {
			return false, err
		}
		got, present := state.DeviceSharingAffinity.DeviceStates[device]
		if status == structured.AffinityStatusClean {
			return !present, nil
		}
		return present && got.Status == status && maps.Equal(got.LockedAffinity, values), nil
	})
	assertSharingAffinityAccountingState(t, gatherSharingAffinityAccounting(t, tc.draManager.resourceClaimTracker), device, status, values)
}

func poolEventsAssertRevision(t *testing.T, ctx context.Context, tc *testContext, before int64) {
	t.Helper()
	a := tc.draManager.resourceClaimTracker.allocatedDevices
	if err := wait.PollUntilContextTimeout(ctx, 5*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return a.Revision() > before, nil
	}); err != nil {
		t.Errorf("metadata publication did not invalidate accounting revision: before=%d, after=%d", before, a.Revision())
	}
}

// Accounting revision is not the pool epoch; Reserve fences on the latter.
func poolEventsEpoch(c *claimTracker, slice *resourceapi.ResourceSlice) uint64 {
	return publishRaceCommitted(c).pools[poolName{slice.Spec.Driver, slice.Spec.Pool.Name}].epoch
}

func poolEventsAssertEpochBumped(t *testing.T, c *claimTracker, slice *resourceapi.ResourceSlice, before uint64) {
	t.Helper()
	if got := poolEventsEpoch(c, slice); got != before+1 {
		t.Errorf("pool %s epoch = %d, want %d", slice.Spec.Pool.Name, got, before+1)
	}
}

func poolEventsProgramCacheSnapshot(t *testing.T, extractor *sharingaffinity.Extractor) (int, []uintptr) {
	t.Helper()
	cache := reflect.ValueOf(extractor).Elem().FieldByName("cache").Elem()
	inner := cache.FieldByName("cache").Elem()
	entries := inner.FieldByName("cache")
	ptrs := make([]uintptr, 0, entries.Len())
	iter := entries.MapRange()
	for iter.Next() {
		element := iter.Value().Elem()
		entry := element.FieldByName("Value").Elem().Elem()
		ptrs = append(ptrs, entry.FieldByName("value").Elem().Pointer())
	}
	sort.Slice(ptrs, func(i, j int) bool { return ptrs[i] < ptrs[j] })
	return entries.Len(), ptrs
}

func newPoolEventsUnitTracker(t *testing.T, slices ...*resourceapi.ResourceSlice) (*claimTracker, *mutableSliceLister) {
	t.Helper()
	setSharingAffinityAccountingGates(t, true, true)
	lister := &mutableSliceLister{slices: slices}
	c := newSharingAffinityAccountingTracker(t, true)
	c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
	c.allocatedDevices.metadata = c.metadata
	c.metadataSlices = lister
	c.affinity.slices = lister
	return c, lister
}

func TestSharingAffinityPoolEvents(t *testing.T) {
	t.Run("events-to-register-delete", func(t *testing.T) {
		for _, sharing := range []bool{false, true} {
			for _, capacity := range []bool{false, true} {
				t.Run(fmt.Sprintf("sharing=%t/capacity=%t", sharing, capacity), func(t *testing.T) {
					// The global gate enforces its dependency, but event registration
					// must also handle independently constructed scheduler flags.
					poolEventsGates(t, sharing && capacity, capacity)
					pl := &DynamicResources{fts: feature.Features{
						EnableDRASharingAffinity: sharing, EnableDRAConsumableCapacity: capacity,
					}}
					events, err := pl.EventsToRegister(ktesting.Init(t))
					if err != nil {
						t.Fatalf("EventsToRegister: %v", err)
					}
					found := 0
					for _, event := range events {
						if event.Event.Resource != fwk.ResourceSlice {
							continue
						}
						found++
						if want := fwk.Add | fwk.Update | fwk.Delete; event.Event.ActionType != want {
							t.Errorf("ResourceSlice actions = %v, want %v", event.Event.ActionType, want)
						}
						if event.QueueingHintFn != nil || event.PreQueueingHintFn != nil {
							t.Error("ResourceSlice events must requeue unconditionally, without hints")
						}
					}
					if found != 1 {
						t.Errorf("ResourceSlice registrations = %d, want 1", found)
					}
				})
			}
		}
	})

	changes := []struct {
		name   string
		mutate func(*resourceapi.SharingAffinityExtractor)
	}{
		{"expression", func(e *resourceapi.SharingAffinityExtractor) { e.Expression = `"changed"` }},
		{"name", func(e *resourceapi.SharingAffinityExtractor) { e.Name = "network" }},
		{"group", func(e *resourceapi.SharingAffinityExtractor) { e.APIGroup = "other.example.com" }},
		{"kind", func(e *resourceapi.SharingAffinityExtractor) { e.Kind = "OtherNetwork" }},
	}
	for _, change := range changes {
		t.Run("same-generation-"+change.name+"-change", func(t *testing.T) {
			for _, owner := range []string{"observed", "in-flight", "share-id-only"} {
				t.Run(owner, func(t *testing.T) {
					slices := poolEventsSlices()
					ctx, tc := poolEventsSetup(t, slices)
					c := tc.draManager.resourceClaimTracker
					claim := sharingAffinityAccountingClaim("owner", "blue")
					if owner == "share-id-only" {
						claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
					}
					if owner == "in-flight" {
						signalSharingAffinityAccounting(t, c, claim)
					} else {
						assumecache.AddTestObject(c.cache, claim)
					}
					device := sharingAffinityAccountingDevice(claim)
					// Add returns synchronously, but a superseding worker token may still be settling the owner; snapshot only once it is Locked.
					poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					before := gatherSharingAffinityAccounting(t, c)
					assertSharingAffinityAccountingState(t, before, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					updated := slices[0].DeepCopy()
					change.mutate(&updated.Spec.SharingAffinity[0])
					epoch := poolEventsEpoch(c, updated)
					poolEventsChange(t, ctx, tc, updated, false)
					poolEventsAssertEpochBumped(t, c, updated, epoch)
					poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
					assertSharingAffinityAccountingState(t, before, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
					if owner == "in-flight" {
						c.MaybeRemoveClaimPendingAllocation(claim.UID, true)
					} else {
						assumecache.DeleteTestObject(c.cache, claim)
					}
					poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusClean, nil)
				})
			}
		})
	}

	for _, changed := range []bool{false, true} {
		name := "generation-roll-unchanged-digest"
		if changed {
			name = "generation-roll-changed-digest"
		}
		t.Run(name, func(t *testing.T) {
			slices := poolEventsSlices()
			ctx, tc := poolEventsSetup(t, slices)
			c := tc.draManager.resourceClaimTracker
			claim := sharingAffinityAccountingClaim("owner", "blue")
			assumecache.AddTestObject(c.cache, claim)
			device := sharingAffinityAccountingDevice(claim)
			for i, slice := range slices {
				before := c.allocatedDevices.Revision()
				updated := slice.DeepCopy()
				updated.Spec.Pool.Generation++
				if i == 0 && changed {
					updated.Spec.SharingAffinity[0].Expression = `"changed"`
				}
				epoch := poolEventsEpoch(c, updated)
				poolEventsChange(t, ctx, tc, updated, false)
				poolEventsAssertEpochBumped(t, c, updated, epoch)
				poolEventsAssertRevision(t, ctx, tc, before)
			}
			if changed {
				poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
			} else {
				poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
			}
		})
	}

	t.Run("pool-incomplete-then-restored", func(t *testing.T) {
		slices := poolEventsSlices()
		ctx, tc := poolEventsSetup(t, slices)
		c := tc.draManager.resourceClaimTracker
		claim := sharingAffinityAccountingClaim("owner", "blue")
		assumecache.AddTestObject(c.cache, claim)
		device := sharingAffinityAccountingDevice(claim)
		before := c.allocatedDevices.Revision()
		poolEventsChange(t, ctx, tc, slices[0], true)
		poolEventsAssertRevision(t, ctx, tc, before)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		restored := slices[0].DeepCopy()
		restored.Spec.SharingAffinity[0].Expression = `"changed"`
		poolEventsChange(t, ctx, tc, restored, false)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
	})

	t.Run("pool-deleted-and-restored", func(t *testing.T) {
		slices := poolEventsSlices()
		ctx, tc := poolEventsSetup(t, slices)
		c := tc.draManager.resourceClaimTracker
		claim := sharingAffinityAccountingClaim("owner", "blue")
		assumecache.AddTestObject(c.cache, claim)
		device := sharingAffinityAccountingDevice(claim)
		for _, slice := range slices {
			before := c.allocatedDevices.Revision()
			epoch := poolEventsEpoch(c, slice)
			poolEventsChange(t, ctx, tc, slice, true)
			poolEventsAssertEpochBumped(t, c, slice, epoch)
			poolEventsAssertRevision(t, ctx, tc, before)
		}
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		for i, slice := range slices {
			restored := slice.DeepCopy()
			if i == 0 {
				restored.Spec.SharingAffinity[0].Expression = `"changed"`
			}
			poolEventsChange(t, ctx, tc, restored, false)
		}
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
	})

	for _, nilCapacity := range []bool{false, true} {
		t.Run(fmt.Sprintf("metadata-removed/without-capacity=%t", nilCapacity), func(t *testing.T) {
			slices := poolEventsSlices()
			ctx, tc := poolEventsSetup(t, slices)
			c := tc.draManager.resourceClaimTracker
			a, b := sharingAffinityAccountingClaim("a", "blue"), sharingAffinityAccountingClaim("b", "blue")
			if nilCapacity {
				a.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
				b.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
			}
			assumecache.AddTestObject(c.cache, a)
			assumecache.AddTestObject(c.cache, b)
			device := sharingAffinityAccountingDevice(a)
			poolEventsChange(t, ctx, tc, slices[0], true)
			devices := slices[1].DeepCopy()
			devices.Spec.Pool.ResourceSliceCount = 1
			poolEventsChange(t, ctx, tc, devices, false)
			poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
			state := gatherSharingAffinityAccounting(t, c)
			if state.AllocatedDevices.Has(device) {
				t.Error("shared device became dedicated after metadata removal")
			}
			if !nilCapacity {
				assertSharingAffinityAccountingCapacity(t, state, device, "10")
			}
			assumecache.DeleteTestObject(c.cache, a)
			poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
			if !nilCapacity {
				assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
			}
			before := c.allocatedDevices.Revision()
			assumecache.DeleteTestObject(c.cache, b)
			poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusClean, nil)
			if c.allocatedDevices.Revision() <= before {
				t.Error("last-owner drain did not advance accounting revision")
			}
		})
	}

	t.Run("metadata-added-without-prior-contribution", func(t *testing.T) {
		slices := poolEventsSlices()
		devices := slices[1].DeepCopy()
		devices.Spec.Pool.ResourceSliceCount = 1
		ctx, tc := poolEventsSetup(t, []*resourceapi.ResourceSlice{devices})
		c := tc.draManager.resourceClaimTracker
		claim := sharingAffinityAccountingClaim("owner", "blue")
		claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
		assumecache.AddTestObject(c.cache, claim)
		device := sharingAffinityAccountingDevice(claim)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusClean, nil)
		poolEventsChange(t, ctx, tc, slices[1], false)
		poolEventsChange(t, ctx, tc, slices[0], false)
		// No semantic baseline existed while the pool was unconstrained.
		// A reconstructable owner therefore establishes its first lock.
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		assumecache.DeleteTestObject(c.cache, claim)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusClean, nil)
	})

	t.Run("nil-capacity-handoff-and-drain", func(t *testing.T) {
		slices := poolEventsSlices()
		ctx, tc := poolEventsSetup(t, slices)
		c := tc.draManager.resourceClaimTracker
		claim := sharingAffinityAccountingClaim("owner", "blue")
		claim.Status.Allocation.Devices.Results[0].ConsumedCapacity = nil
		device := sharingAffinityAccountingDevice(claim)
		signalSharingAffinityAccounting(t, c, claim)
		assumecache.UpdateTestObject(c.cache, claim)
		assertSharingAffinityAccountingOwners(t, c.allocatedDevices, device, 1, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		c.MaybeRemoveClaimPendingAllocation(claim.UID, true)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
		updated := slices[0].DeepCopy()
		updated.Spec.SharingAffinity[0].Expression = `"changed"`
		poolEventsChange(t, ctx, tc, updated, false)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusUnreconstructable, nil)
		state := gatherSharingAffinityAccounting(t, c)
		if state.AllocatedDevices.Has(device) || len(state.AggregatedCapacity[device]) != 0 {
			t.Errorf("nil-capacity shared owner acquired dedicated ownership or capacity: %#v", state)
		}
		assumecache.DeleteTestObject(c.cache, claim)
		poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusClean, nil)
	})

	t.Run("unrelated-pool-does-not-fence", func(t *testing.T) {
		for _, identity := range []string{"driver", "pool"} {
			t.Run(identity, func(t *testing.T) {
				slices := poolEventsSlices()
				other := poolEventsSlices()
				for _, slice := range other {
					slice.Name += "-other"
					slice.UID = types.UID(slice.Name)
					if identity == "driver" {
						slice.Spec.Driver = "other.example.com"
					} else {
						slice.Spec.Pool.Name = "other"
					}
				}
				ctx, tc := poolEventsSetup(t, append(slices, other...))
				c := tc.draManager.resourceClaimTracker
				claim := sharingAffinityAccountingClaim("owner", "blue")
				assumecache.AddTestObject(c.cache, claim)
				before := c.allocatedDevices.Revision()
				updated := other[0].DeepCopy()
				updated.Spec.SharingAffinity[0].Expression = `"changed"`
				poolEventsChange(t, ctx, tc, updated, false)
				poolEventsAssertRevision(t, ctx, tc, before)
				device := sharingAffinityAccountingDevice(claim)
				poolEventsAssertState(t, ctx, tc, device, structured.AffinityStatusLocked, map[string]string{"subnet": "blue"})
				assertSharingAffinityAccountingCapacity(t, gatherSharingAffinityAccounting(t, c), device, "5")
			})
		}
	})

	t.Run("stale-filter-result", func(t *testing.T) {
		for _, mutation := range []string{"expression", "name", "group", "kind", "generation", "deleted", "metadata-added"} {
			t.Run(mutation, func(t *testing.T) {
				slices := poolEventsSlices()
				initial := slices
				if mutation == "metadata-added" {
					devices := slices[1].DeepCopy()
					devices.Spec.Pool.ResourceSliceCount = 1
					initial = []*resourceapi.ResourceSlice{devices}
				}
				claim := sharingAffinityAccountingClaim("pending", "blue")
				claim.UID = ""
				claim.ResourceVersion = ""
				claim.Status.Allocation = nil
				claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
				claim.Spec.Devices.Requests[0].Exactly.Count = 1
				claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
					Requests: map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("5")},
				}
				ctx, tc := poolEventsSetup(t, initial, claim)
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "consumer", Namespace: claim.Namespace, UID: "consumer"},
					Spec: v1.PodSpec{ResourceClaims: []v1.PodResourceClaim{{
						Name: "resource", ResourceClaimName: ptr.To(claim.Name),
					}}},
				}
				if _, status := tc.p.PreFilter(ctx, tc.state, pod, tc.nodeInfos); !status.IsSuccess() {
					t.Fatalf("PreFilter before metadata mutation: %v", status)
				}
				if status := tc.p.Filter(ctx, tc.state, pod, tc.nodeInfos[0]); !status.IsSuccess() {
					t.Fatalf("Filter before metadata mutation: %v", status)
				}
				updated := slices[0].DeepCopy()
				switch mutation {
				case "generation":
					for _, slice := range slices {
						next := slice.DeepCopy()
						next.Spec.Pool.Generation++
						poolEventsChange(t, ctx, tc, next, false)
					}
				case "deleted":
					poolEventsChange(t, ctx, tc, updated, true)
				case "metadata-added":
					poolEventsChange(t, ctx, tc, slices[1], false)
					poolEventsChange(t, ctx, tc, slices[0], false)
				default:
					for _, change := range changes {
						if change.name == mutation {
							change.mutate(&updated.Spec.SharingAffinity[0])
						}
					}
					poolEventsChange(t, ctx, tc, updated, false)
				}
				status := tc.p.Reserve(ctx, tc.state, pod, tc.nodeInfos[0].Node().Name)
				if status.Code() != fwk.Unschedulable {
					t.Errorf("Reserve after %s mutation = %v, want Unschedulable", mutation, status)
				}
				claims, err := tc.draManager.resourceClaimTracker.List()
				if err != nil || len(claims) != 1 {
					t.Fatalf("list claims after Reserve: %v, %v", claims, err)
				}
				if pending := tc.draManager.resourceClaimTracker.GetPendingAllocation(claims[0].UID); pending != nil {
					t.Errorf("stale Reserve published pending allocation: %#v", pending)
				}
				state := gatherSharingAffinityAccounting(t, tc.draManager.resourceClaimTracker)
				if len(state.DeviceSharingAffinity.DeviceStates) != 0 || len(state.AggregatedCapacity) != 0 {
					t.Errorf("stale Reserve leaked ownership: affinity=%v capacity=%v", state.DeviceSharingAffinity.DeviceStates, state.AggregatedCapacity)
				}
				tc.p.Unreserve(ctx, tc.state, pod, tc.nodeInfos[0].Node().Name)
			})
		}
	})

	t.Run("unsynced-registry", func(t *testing.T) {
		for _, taintRules := range []bool{false, true} {
			t.Run(fmt.Sprintf("taint-rules=%t", taintRules), func(t *testing.T) {
				slices := poolEventsSlices()
				unconstrained := sharingAffinityTestSlice("unsynced-unconstrained", "unconstrained.example.com", "pool", 1, 1)
				unconstrained.Spec.AllNodes = ptr.To(true)
				slices = append(slices, unconstrained)
				claim := sharingAffinityAccountingClaim("unsynced", "blue")
				claim.UID, claim.ResourceVersion = "", ""
				claim.Status.Allocation = nil
				claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
				claim.Spec.Devices.Requests[0].Exactly.Count = 1
				ctx, tc := poolEventsSetup(t, slices, claim)
				tc.p.fts.EnableDRADeviceTaints = taintRules
				c := tc.draManager.resourceClaimTracker
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "unsynced", Namespace: claim.Namespace, UID: "unsynced"},
					Spec: v1.PodSpec{ResourceClaims: []v1.PodResourceClaim{{
						Name: "resource", ResourceClaimName: ptr.To(claim.Name),
					}}},
				}
				// A cycle filtered while synced must not commit once the
				// registry loses coherence: retryable Error, not Unschedulable.
				filtered := framework.NewCycleState()
				if _, status := tc.p.PreFilter(ctx, filtered, pod, tc.nodeInfos); !status.IsSuccess() {
					t.Fatalf("synced PreFilter: %v", status)
				}
				if status := tc.p.Filter(ctx, filtered, pod, tc.nodeInfos[0]); !status.IsSuccess() {
					t.Fatalf("synced Filter: %v", status)
				}
				c.allocatedDevices.mutex.Lock()
				c.metadata.synced = false
				c.allocatedDevices.mutex.Unlock()
				if status := tc.p.Reserve(ctx, filtered, pod, tc.nodeInfos[0].Node().Name); status.Code() != fwk.Error || !errors.Is(status.AsError(), errAffinityMetadataNotCoherent) {
					t.Fatalf("unsynced plugin Reserve = %v, want Error wrapping metadata-not-coherent", status)
				}
				if pending := c.GetPendingAllocation(claim.UID); pending != nil {
					t.Fatalf("unsynced plugin Reserve published pending allocation: %#v", pending)
				}
				tc.p.Unreserve(ctx, filtered, pod, tc.nodeInfos[0].Node().Name)
				_, status := tc.p.PreFilter(ctx, framework.NewCycleState(), pod, tc.nodeInfos)
				if status.Code() != fwk.Error || !errors.Is(status.AsError(), errAffinityMetadataNotCoherent) {
					t.Fatalf("unsynced PreFilter = %v, want Error wrapping metadata-not-coherent", status)
				}
				allocation := &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
					{Request: "req", Driver: "unconstrained.example.com", Pool: "pool", Device: "device", ShareID: ptr.To(types.UID("share"))},
				}}}
				claimWithAllocation := claim.DeepCopy()
				claimWithAllocation.UID = "unsynced-uid"
				claimWithAllocation.ResourceVersion = "1"
				claimWithAllocation.Status.Allocation = allocation
				current := claimWithAllocation.DeepCopy()
				current.Status.Allocation = nil
				assumecache.AddTestObject(c.cache, current)
				err := c.ReserveClaimWithAffinity(ctx, affinityReservation{
					claimUID: claimWithAllocation.UID, claimRV: claimWithAllocation.ResourceVersion,
					claim: claimWithAllocation, token: reservationToken("token"),
					provenance: []structured.DeviceAffinity{{
						ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID("unconstrained.example.com", "pool", "device"),
						Pool:   sharingaffinity.PoolKey{Driver: "unconstrained.example.com", Name: "pool", Generation: 1},
						Digest: structured.EmptyExtractorDigest,
					}},
					fenceEpochs: map[poolName]uint64{{"unconstrained.example.com", "pool"}: c.metadata.pools[poolName{"unconstrained.example.com", "pool"}].epoch},
				})
				if !errors.Is(err, errAffinityMetadataNotCoherent) {
					t.Fatalf("unsynced unconstrained Reserve = %v, want metadata-not-coherent", err)
				}
				c.publishMetadata(true, nil)
				if _, status := tc.p.PreFilter(ctx, framework.NewCycleState(), pod, tc.nodeInfos); !status.IsSuccess() {
					t.Fatalf("PreFilter after sync = %v, want success without another slice event", status)
				}
				err = c.ReserveClaimWithAffinity(ctx, affinityReservation{
					claimUID: claimWithAllocation.UID, claimRV: claimWithAllocation.ResourceVersion,
					claim: claimWithAllocation, token: reservationToken("retry"),
					provenance: []structured.DeviceAffinity{{
						ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID("unconstrained.example.com", "pool", "device"),
						Pool:   sharingaffinity.PoolKey{Driver: "unconstrained.example.com", Name: "pool", Generation: 1},
						Digest: structured.EmptyExtractorDigest,
					}},
					fenceEpochs: map[poolName]uint64{{"unconstrained.example.com", "pool"}: c.metadata.pools[poolName{"unconstrained.example.com", "pool"}].epoch},
				})
				if err != nil {
					t.Fatalf("Reserve retry after metadata sync = %v, want success without another slice event", err)
				}
			})
		}
	})

	t.Run("late-observed-empty-contribution", func(t *testing.T) {
		// Reconciliation moves from pending through replay to Locked.
		// CEL may run again, but never under accounting locks.
		reconcileLateObservedMetadata(t, false)
	})

	t.Run("stale-observed-contribution", func(t *testing.T) {
		reconcileLateObservedMetadata(t, true)
	})

	t.Run("observed-commit-before-handler", func(t *testing.T) {
		reconcileObservedAheadOfPublication(t)
	})

	t.Run("reordered-and-duplicate-callbacks", func(t *testing.T) {
		oldSlice := publishRaceSlice("p", 1, `"old"`)
		newSlice := publishRaceSlice("p", 2, `"new"`)
		lister := &mutableSliceLister{slices: []*resourceapi.ResourceSlice{oldSlice}}
		c := newPublishRaceTracker(t, lister)
		c.publishMetadata(true, nil)
		key := poolName{"driver.example.com", "p"}
		before := publishRaceCommitted(c).pools[key].epoch
		lister.set(newSlice)
		c.publishMetadata(false, affectedPools(oldSlice, newSlice))
		afterNew := publishRaceCommitted(c)
		c.publishMetadata(false, affectedPools(oldSlice))
		c.publishMetadata(false, affectedPools(newSlice))
		afterDup := publishRaceCommitted(c)
		if !sameMetadata(afterDup.pools[key], metadataView([]*resourceapi.ResourceSlice{newSlice}, key)) {
			t.Fatalf("duplicate/reordered callbacks committed %+v, want current new slice", afterDup.pools[key])
		}
		if afterDup.pools[key].epoch != before+1 || afterDup.pools[key].epoch != afterNew.pools[key].epoch {
			t.Fatalf("epoch after duplicate callbacks = %d, want one increment from %d", afterDup.pools[key].epoch, before)
		}
	})

	t.Run("program-cache-retained-across-metadata-events", func(t *testing.T) {
		slices := poolEventsSlices()
		ctx, tc := poolEventsSetup(t, slices)
		c := tc.draManager.resourceClaimTracker
		owner := sharingAffinityAccountingClaim("cache-owner", "blue")
		assumecache.AddTestObject(c.cache, owner)
		wantLen, wantPrograms := poolEventsProgramCacheSnapshot(t, c.affinity.extractor)
		if wantLen != 1 {
			t.Fatalf("program cache length after observed owner = %d, want one compiled expression", wantLen)
		}

		added := sharingAffinityTestSlice("cache-added-metadata", accountingDriver, "cache-added", 1, 1, slices[0].Spec.SharingAffinity...)
		added.Spec.AllNodes = ptr.To(true)
		poolEventsChange(t, ctx, tc, added, false)
		gotLen, gotPrograms := poolEventsProgramCacheSnapshot(t, c.affinity.extractor)
		if gotLen != wantLen || !reflect.DeepEqual(gotPrograms, wantPrograms) {
			t.Fatalf("program cache after metadata add = len %d programs %v, want len %d programs %v", gotLen, gotPrograms, wantLen, wantPrograms)
		}

		updated := slices[0].DeepCopy()
		updated.Labels = map[string]string{"unchanged-expression": "true"}
		poolEventsChange(t, ctx, tc, updated, false)
		gotLen, gotPrograms = poolEventsProgramCacheSnapshot(t, c.affinity.extractor)
		if gotLen != wantLen || !reflect.DeepEqual(gotPrograms, wantPrograms) {
			t.Fatalf("program cache after metadata update = len %d programs %v, want len %d programs %v", gotLen, gotPrograms, wantLen, wantPrograms)
		}

		// Reconciliation must retain previously compiled programs.
		assertRetained := func(event string) {
			t.Helper()
			gotLen, gotPrograms := poolEventsProgramCacheSnapshot(t, c.affinity.extractor)
			got := sets.New(gotPrograms...)
			if gotLen < wantLen || !got.HasAll(wantPrograms...) {
				t.Fatalf("program cache after %s = len %d programs %v, want retained %v", event, gotLen, gotPrograms, wantPrograms)
			}
		}
		changed := updated.DeepCopy()
		changed.Spec.SharingAffinity[0].Expression = `"changed"`
		reconcilePublish(t, ctx, tc, []*resourceapi.ResourceSlice{changed, slices[1]})
		// Reflection cannot acquire the CEL cache's private mutex. Wait for
		// the only owner to finish compiling before inspecting its contents.
		reconcileWaitState(t, ctx, c, sharingAffinityAccountingDevice(owner), structured.AffinityStatusUnreconstructable, nil)
		assertRetained("extractor-changing update")
		poolEventsChange(t, ctx, tc, changed, true)
		assertRetained("metadata delete")
	})
}

// The metadata registry must stay unsynced until the manager's own handler
// registration reports sync, regardless of whether the slice tracker
// delegates directly to the informer (taint rules off) or patches slices.
func TestAffinityMetadataRegistrationReadiness(t *testing.T) {
	for _, taintRules := range []bool{false, true} {
		t.Run(fmt.Sprintf("taint-rules=%t", taintRules), func(t *testing.T) {
			poolEventsGates(t, true, true)
			tCtx := ktesting.Init(t)
			objects := make([]runtime.Object, 0, 2)
			for _, slice := range poolEventsSlices() {
				objects = append(objects, slice)
			}
			client := fake.NewSimpleClientset(objects...)
			factory := informers.NewSharedInformerFactory(client, 0)
			sliceTracker, err := resourceslicetracker.StartTracker(tCtx, resourceslicetracker.Options{
				EnableDeviceTaintRules: taintRules,
				SliceInformer:          factory.Resource().V1().ResourceSlices(),
				TaintInformer:          factory.Resource().V1().DeviceTaintRules(),
				KubeClient:             client,
			})
			if err != nil {
				t.Fatalf("StartTracker: %v", err)
			}
			registrationReady := make(chan struct{})
			var registrationReleased atomic.Bool
			original := metadataRegistrationSynced
			metadataRegistrationSynced = func(registration cache.ResourceEventHandlerRegistration) cache.InformerSynced {
				return func() bool {
					select {
					case <-registrationReady:
						return registration.HasSynced()
					default:
						return false
					}
				}
			}
			t.Cleanup(func() { metadataRegistrationSynced = original })
			claims := assumecache.NewAssumeCache(tCtx.Logger(), factory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
			manager := NewDRAManager(tCtx, claims, sliceTracker, factory)
			c := manager.resourceClaimTracker
			entered := make(chan struct{})
			release := make(chan struct{})
			// Set before the informers start: the sync goroutine cannot reach
			// the hook until the registration reports sync.
			c.metadataSyncedHook = func() {
				close(entered)
				<-release
			}
			synced := func() bool {
				c.allocatedDevices.mutex.RLock()
				defer c.allocatedDevices.mutex.RUnlock()
				return c.metadata.synced
			}
			if synced() {
				t.Fatal("metadata synced before the handler registration synced")
			}
			factory.Start(tCtx.Done())
			tCtx.Cleanup(factory.Shutdown)
			// Informer and tracker readiness alone must not start initialization.
			if err := wait.PollUntilContextTimeout(tCtx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
				return sliceTracker.HasSynced(), nil
			}); err != nil {
				t.Fatalf("slice tracker never synced: %v", err)
			}
			select {
			case <-entered:
				t.Fatal("initialization entered while the handler registration was not ready")
			case <-time.After(100 * time.Millisecond):
			}
			if synced() {
				t.Fatal("metadata synced while only the informer/tracker was ready")
			}
			registrationReleased.Store(true)
			close(registrationReady)
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("handler registration did not sync")
			}
			if synced() {
				t.Fatal("metadata synced before initial publication completed")
			}
			if !registrationReleased.Load() {
				t.Fatal("initialization entered before registration readiness")
			}
			close(release)
			if err := wait.PollUntilContextTimeout(tCtx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
				return synced(), nil
			}); err != nil {
				t.Fatalf("metadata never synced after registration sync: %v", err)
			}
		})
	}
}
