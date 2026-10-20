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
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	cgotesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

type reserveFixture struct {
	ctx      ktesting.TContext
	tc       *testContext
	metadata *resourceapi.ResourceSlice
	devices  *resourceapi.ResourceSlice
}

// Narrowing to the public tracker interface models a consumer without optional
// affinity reservation support while delegating all accounting to the real manager.
type reserveLegacyTracker struct{ fwk.ResourceClaimTracker }

type reserveLegacyManager struct {
	fwk.SharedDRAManager
	claims fwk.ResourceClaimTracker
}

func (m reserveLegacyManager) ResourceClaims() fwk.ResourceClaimTracker { return m.claims }

func reserveClaim(name, subnet string) *resourceapi.ResourceClaim {
	claim := sharingAffinityAccountingClaim(types.UID(name), subnet)
	claim.Status = resourceapi.ResourceClaimStatus{}
	claim.Finalizers = []string{resourceapi.Finalizer}
	claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
	claim.Spec.Devices.Requests[0].Exactly.Count = 1
	claim.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
		Requests: map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("5")},
	}
	return claim
}

func reservePod(name string, claims ...*resourceapi.ResourceClaim) *v1.Pod {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: name, UID: types.UID(name)}}
	for i, claim := range claims {
		pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims, v1.PodResourceClaim{
			Name: fmt.Sprintf("claim-%d", i), ResourceClaimName: ptr.To(claim.Name),
		})
	}
	return pod
}

func newReserveFixture(t *testing.T, enabled bool, claims ...*resourceapi.ResourceClaim) *reserveFixture {
	t.Helper()
	return newReserveFixtureWithContext(t, ktesting.Init(t), enabled, claims...)
}

func newReserveFixtureWithExtractionLog(t *testing.T, claims ...*resourceapi.ResourceClaim) (*reserveFixture, func() int) {
	t.Helper()
	logContext, buffer := sharingAffinityTestLogContext(t, 6)
	f := newReserveFixtureWithContext(t, ktesting.Init(t).WithLogger(klog.FromContext(logContext)), true, claims...)
	return f, func() int {
		count := 0
		for _, entry := range buffer.Data() {
			if entry.Message == "Extracting sharing affinity" {
				count++
			}
		}
		return count
	}
}

func newReserveFixtureWithContext(t *testing.T, ctx ktesting.TContext, enabled bool, claims ...*resourceapi.ResourceClaim) *reserveFixture {
	t.Helper()
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRASharingAffinity:    enabled,
	})
	slices := newSharingAffinityAccounting().slices.(accountingSliceLister).slices
	metadata, devices := slices[0].DeepCopy(), slices[1].DeepCopy()
	for _, slice := range []*resourceapi.ResourceSlice{metadata, devices} {
		slice.UID = types.UID(slice.Name)
		slice.ResourceVersion = "1"
	}
	devices.Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
		"units": {Value: resource.MustParse("100")},
	}
	devices.Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"other": {BoolValue: ptr.To(false)},
	}
	if !enabled {
		// Gate-off rollback must exercise a permissive pool, not bypass a declaring pool.
		metadata.Spec.SharingAffinity = nil
	}
	objs := []apiruntime.Object{metadata, devices}
	for _, claim := range claims {
		objs = append(objs, claim.DeepCopy())
	}
	class := &resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}}
	tc := setup(ctx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, nil, []*resourceapi.DeviceClass{class}, nil, objs,
		feature.Features{EnableDRAConsumableCapacity: true, EnableDRASharingAffinity: enabled}, false, nil)
	if enabled && len(claims) > 0 {
		// setup waits for baseline handlers, not the manager's asynchronous
		// initial metadata publication. Probe readiness without running Filter.
		probe := reservePod("reservation-metadata-readiness", claims[0])
		var status *fwk.Status
		if err := wait.PollUntilContextTimeout(ctx, time.Millisecond, 5*time.Second, true, func(pollCtx context.Context) (bool, error) {
			_, status = tc.p.PreFilter(pollCtx, framework.NewCycleState(), probe, tc.nodeInfos)
			if status.IsSuccess() {
				return true, nil
			}
			if status.Code() == fwk.Error && strings.Contains(status.Message(), "sharing affinity metadata not yet") {
				return false, nil
			}
			return false, fmt.Errorf("non-readiness PreFilter failure: %v", status)
		}); err != nil {
			t.Fatalf("initial sharing-affinity metadata did not become coherent: %v; PreFilter: %v", err, status)
		}
	}
	return &reserveFixture{ctx: ctx, tc: tc, metadata: metadata, devices: devices}
}

func (f *reserveFixture) filter(t *testing.T, pod *v1.Pod, group fwk.PodGroupCycleState) fwk.CycleState {
	t.Helper()
	state := framework.NewCycleState()
	if group != nil {
		state.SetPodGroupCycleState(group)
	}
	_, status := f.tc.p.PreFilter(f.ctx, state, pod, f.tc.nodeInfos)
	if !status.IsSuccess() {
		t.Fatalf("PreFilter(%s): %v", pod.Name, status)
	}
	if status := f.tc.p.Filter(f.ctx, state, pod, f.tc.nodeInfos[0]); !status.IsSuccess() {
		t.Fatalf("Filter(%s): %v", pod.Name, status)
	}
	return state
}

func (f *reserveFixture) reserve(t *testing.T, state fwk.CycleState, pod *v1.Pod) {
	t.Helper()
	if status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name); !status.IsSuccess() {
		t.Fatalf("Reserve(%s): %v", pod.Name, status)
	}
}

func (f *reserveFixture) snapshot(t *testing.T) *structured.AllocatedState {
	t.Helper()
	return gatherSharingAffinityAccounting(t, f.tc.draManager.resourceClaimTracker)
}

// Aggregate accounting cannot see a ghost same-claim token, so compare tokens directly.
func (f *reserveFixture) pendingTokens(uid types.UID) sets.Set[reservationToken] {
	c := f.tc.draManager.resourceClaimTracker
	c.inFlightMutex.RLock()
	defer c.inFlightMutex.RUnlock()
	entry, found := c.inFlightAllocations[uid]
	if !found {
		return nil
	}
	return entry.tokens.Clone()
}

func reserveStateToken(t *testing.T, state fwk.CycleState, uid types.UID) reservationToken {
	t.Helper()
	s, err := getStateData(state)
	if err != nil {
		t.Fatal(err)
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()
	token, ok := s.affinityReservations[uid]
	if !ok {
		t.Fatalf("no reservation token recorded for claim %s", uid)
	}
	return token
}

func (f *reserveFixture) unreserve(state fwk.CycleState, pod *v1.Pod) {
	f.tc.p.Unreserve(f.ctx, state, pod, workerNode.Name)
}

func reserveAssertState(t *testing.T, state *structured.AllocatedState, subnet, capacity string) {
	t.Helper()
	device := structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice)
	status := structured.AffinityStatusLocked
	var values map[string]string
	if subnet == "" {
		status = structured.AffinityStatusClean
	} else {
		values = map[string]string{"subnet": subnet}
	}
	assertSharingAffinityAccountingState(t, state, device, status, values)
	if capacity != "" {
		assertSharingAffinityAccountingCapacity(t, state, device, capacity)
	} else if len(state.AggregatedCapacity) != 0 || len(state.AllocatedSharedDeviceIDs) != 0 {
		t.Errorf("drained allocation retains capacity or shared devices: %#v", state)
	}
}

func reserveAssertUnchanged(t *testing.T, before, after *structured.AllocatedState) {
	t.Helper()
	if !reflect.DeepEqual(before, after) {
		t.Errorf("reservation changed another owner's state:\nbefore: %#v\nafter:  %#v", before, after)
	}
}

func (f *reserveFixture) updateSlice(t *testing.T, slice *resourceapi.ResourceSlice) {
	t.Helper()
	if _, err := f.tc.client.ResourceV1().ResourceSlices().Update(f.ctx, slice, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("update ResourceSlice: %v", err)
	}
	if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		slices, err := f.tc.draManager.ResourceSlices().ListWithDeviceTaintRules()
		for _, current := range slices {
			if current.Name == slice.Name && reflect.DeepEqual(current.Spec, slice.Spec) {
				return true, nil
			}
		}
		return false, err
	}); err != nil {
		t.Fatalf("wait for ResourceSlice tracker update: %v", err)
	}
}

func TestSharingAffinityReserve(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("missing-tracker-interface/gate=%t", enabled), func(t *testing.T) {
			claim := reserveClaim("legacy-tracker", "blue")
			f := newReserveFixture(t, enabled, claim)
			metadata := f.metadata.DeepCopy()
			metadata.Spec.SharingAffinity = nil
			f.updateSlice(t, metadata)
			pod := reservePod("legacy-tracker-pod", claim)
			state := f.filter(t, pod, nil)
			f.tc.p.draManager = reserveLegacyManager{
				SharedDRAManager: f.tc.draManager,
				claims:           reserveLegacyTracker{f.tc.draManager.ResourceClaims()},
			}
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if enabled {
				if status.Code() != fwk.Error {
					t.Errorf("Reserve = %v, want Error when tracker lacks affinity reservation support, even without contributions", status)
				}
				reserveAssertState(t, f.snapshot(t), "", "")
			} else {
				if !status.IsSuccess() {
					t.Fatalf("legacy gate-off Reserve: %v", status)
				}
				reserveAssertState(t, f.snapshot(t), "", "5")
			}
		})
	}

	t.Run("legacy-signal-cannot-increment-affinity-reservation", func(t *testing.T) {
		claim := reserveClaim("tokenized", "blue")
		f := newReserveFixture(t, true, claim)
		pod := reservePod("tokenized-pod", claim)
		state := f.filter(t, pod, nil)
		f.reserve(t, state, pod)
		allocated := claim.DeepCopy()
		allocated.Status.Allocation = f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID).DeepCopy()
		before := f.snapshot(t)
		if err := f.tc.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, allocated); err == nil {
			t.Error("legacy SignalClaimPendingAllocation accepted an affinity reservation")
		}
		reserveAssertUnchanged(t, before, f.snapshot(t))
		f.unreserve(state, pod)
		reserveAssertState(t, f.snapshot(t), "", "")
	})

	t.Run("legacy-entry-token-boundaries", func(t *testing.T) {
		claim := reserveClaim("legacy", "blue")
		f := newReserveFixture(t, true, claim)
		allocated := sharingAffinityAccountingClaim(claim.UID, "blue")
		allocated.Namespace, allocated.Name, allocated.ResourceVersion = claim.Namespace, claim.Name, claim.ResourceVersion
		if err := f.tc.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, allocated); err != nil {
			t.Fatalf("create legacy pending allocation: %v", err)
		}
		before := f.snapshot(t)
		key := poolName{accountingDriver, accountingPool}
		err := f.tc.draManager.resourceClaimTracker.ReserveClaimWithAffinity(f.ctx, affinityReservation{
			claimUID: claim.UID, claimRV: "0", claim: allocated.DeepCopy(), token: reservationToken("token"),
			provenance: []structured.DeviceAffinity{{
				ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice),
				Pool:   sharingaffinity.PoolKey{Driver: accountingDriver, Name: accountingPool, Generation: 1},
				Digest: structured.ExtractorDigest(f.metadata.Spec.SharingAffinity), Constrained: true,
				Values: map[string]string{"subnet": "blue"}, Contributes: true,
			}},
			fenceEpochs: map[poolName]uint64{key: f.tc.draManager.resourceClaimTracker.metadata.pools[key].epoch},
		})
		if !errors.Is(err, errAffinityStale) {
			t.Fatalf("ReserveClaimWithAffinity joining legacy entry = %v, want stale", err)
		}
		reserveAssertUnchanged(t, before, f.snapshot(t))
		if deleted := f.tc.draManager.resourceClaimTracker.ReleaseClaimPendingAllocation(claim.UID, reservationToken("not-a-legacy-token")); deleted {
			t.Fatal("token release deleted a legacy in-flight entry")
		}
		if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) == nil {
			t.Fatal("legacy entry disappeared after token-release no-op")
		}
		revision := f.tc.draManager.resourceClaimTracker.allocatedDevices.Revision()
		if deleted := f.tc.draManager.resourceClaimTracker.MaybeRemoveClaimPendingAllocation(claim.UID, false); !deleted {
			t.Fatal("legacy removal did not delete the legacy in-flight entry")
		}
		if got := f.tc.draManager.resourceClaimTracker.allocatedDevices.Revision(); got <= revision {
			t.Errorf("revision after legacy in-flight removal = %d, want > %d", got, revision)
		}
		reserveAssertState(t, f.snapshot(t), "", "")
	})

	t.Run("first-owner-immediately-visible", func(t *testing.T) {
		claim := reserveClaim("first", "blue")
		f := newReserveFixture(t, true, claim)
		pod := reservePod("first-pod", claim)
		state := f.filter(t, pod, nil)
		reserveAssertState(t, f.snapshot(t), "", "")
		f.reserve(t, state, pod)
		reserveAssertState(t, f.snapshot(t), "blue", "5")
		if pending := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); pending == nil {
			t.Fatal("successful Reserve did not publish the allocation")
		}
	})

	for _, concurrent := range []bool{false, true} {
		for _, equal := range []bool{false, true} {
			t.Run(fmt.Sprintf("atomic/concurrent=%t/equal=%t", concurrent, equal), func(t *testing.T) {
				a, b := reserveClaim("a", "blue"), reserveClaim("b", "red")
				if equal {
					b = reserveClaim("b", "blue")
				}
				f := newReserveFixture(t, true, a, b)
				pods := []*v1.Pod{reservePod("pod-a", a), reservePod("pod-b", b)}
				states := []fwk.CycleState{f.filter(t, pods[0], nil), f.filter(t, pods[1], nil)}
				reserveAssertState(t, f.snapshot(t), "", "")
				statuses := make([]*fwk.Status, 2)
				if concurrent {
					start, done := make(chan struct{}), make(chan int, 2)
					for i := range pods {
						go func(i int) {
							<-start
							statuses[i] = f.tc.p.Reserve(f.ctx, states[i], pods[i], workerNode.Name)
							done <- i
						}(i)
					}
					close(start)
					timeout := time.NewTimer(5 * time.Second)
					defer timeout.Stop()
					for range pods {
						select {
						case <-done:
						case <-timeout.C:
							t.Fatal("concurrent Reserve did not finish")
						}
					}
				} else {
					statuses[0] = f.tc.p.Reserve(f.ctx, states[0], pods[0], workerNode.Name)
					before := f.snapshot(t)
					statuses[1] = f.tc.p.Reserve(f.ctx, states[1], pods[1], workerNode.Name)
					if !equal {
						reserveAssertUnchanged(t, before, f.snapshot(t))
					}
				}
				successes := 0
				winner := 0
				for i, status := range statuses {
					if status.IsSuccess() {
						successes++
						winner = i
					} else if status.Code() != fwk.Unschedulable {
						t.Errorf("Reserve(%s) status = %v, want Unschedulable conflict", pods[i].Name, status)
					}
				}
				wantSuccesses := 1
				if equal {
					wantSuccesses = 2
				}
				if successes != wantSuccesses {
					t.Errorf("successful reservations = %d, want %d; statuses = %v", successes, wantSuccesses, statuses)
				}
				if equal {
					reserveAssertState(t, f.snapshot(t), "blue", "10")
				} else if successes == 1 {
					subnet := "blue"
					if winner == 1 {
						subnet = "red"
					}
					reserveAssertState(t, f.snapshot(t), subnet, "5")
				}
			})
		}
	}

	for _, subnet := range []string{"blue", "red", ""} {
		t.Run("current-observed-owner/"+fmt.Sprintf("subnet=%q", subnet), func(t *testing.T) {
			claim := reserveClaim("candidate", "blue")
			f := newReserveFixture(t, true, claim)
			pod := reservePod("candidate-pod", claim)
			state := f.filter(t, pod, nil)
			observed := sharingAffinityAccountingClaim("observed", subnet)
			assumecache.AddTestObject(f.tc.draManager.resourceClaimTracker.cache, observed)
			reconcileWaitForIdle(t, f.ctx, f.tc.draManager.resourceClaimTracker)
			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if subnet == "blue" {
				if !status.IsSuccess() {
					t.Fatalf("compatible observed owner cannot be joined: %v", status)
				}

				t.Run("claim-changed-between-precheck-and-lock", func(t *testing.T) {
					claim := reserveClaim("changed-inside-reserve", "blue")
					f := newReserveFixture(t, true, claim)
					pod := reservePod("changed-inside-reserve-pod", claim)
					state := f.filter(t, pod, nil)
					f.tc.draManager.resourceClaimTracker.reserveAfterListHook = func() {
						changed := claim.DeepCopy()
						changed.ResourceVersion = "2"
						assumecache.UpdateTestObject(f.tc.draManager.resourceClaimTracker.cache, changed)
					}
					defer func() { f.tc.draManager.resourceClaimTracker.reserveAfterListHook = nil }()
					before := f.snapshot(t)
					status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
					if status.Code() != fwk.Error || !errors.Is(status.AsError(), errAffinityClaimVersionChanged) {
						t.Fatalf("Reserve = %v, want Error wrapping ResourceVersion change from second identity check", status)
					}
					if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
						t.Fatal("claim changed during Reserve published an in-flight allocation")
					}
					reserveAssertUnchanged(t, before, f.snapshot(t))
				})
				reserveAssertState(t, f.snapshot(t), "blue", "10")
			} else {
				if status.Code() != fwk.Unschedulable {
					t.Errorf("Reserve = %v, want Unschedulable for current conflicting or unreconstructable owner", status)
				}
				reserveAssertUnchanged(t, before, f.snapshot(t))
			}
		})
	}

	for _, subnet := range []string{"blue", "red", ""} {
		t.Run("admin-access-current-owner/"+fmt.Sprintf("subnet=%q", subnet), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, true)
			claim := reserveClaim("admin", "blue")
			claim.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			claim.Spec.Devices.Requests[0].Exactly.Capacity = nil
			f := newReserveFixture(t, true, claim)
			f.tc.p.fts.EnableDRAAdminAccess = true
			pod := reservePod("admin-pod", claim)
			state := f.filter(t, pod, nil)
			pluginState, err := getStateData(state)
			if err != nil {
				t.Fatal(err)
			}
			results := pluginState.nodeAllocations[workerNode.Name].allocationResults[0].Devices.Results
			if len(results) != 1 || !ptr.Deref(results[0].AdminAccess, false) {
				t.Fatalf("Filter did not produce a non-owning admin allocation: %#v", results)
			}
			owner := sharingAffinityAccountingClaim("observed", subnet)
			assumecache.AddTestObject(f.tc.draManager.resourceClaimTracker.cache, owner)
			reconcileWaitForIdle(t, f.ctx, f.tc.draManager.resourceClaimTracker)
			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if subnet == "blue" {
				if !status.IsSuccess() {
					t.Errorf("compatible admin Reserve: %v", status)
				}
			} else if status.Code() != fwk.Unschedulable {
				t.Errorf("admin Reserve = %v, want Unschedulable for conflicting or unreconstructable current owner", status)
			}
			reserveAssertUnchanged(t, before, f.snapshot(t))
			if subnet != "blue" && f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
				t.Error("rejected admin reservation was published")
			}
		})
	}

	t.Run("no-second-cel", func(t *testing.T) {
		claim := reserveClaim("cached", "blue")
		f, extractions := newReserveFixtureWithExtractionLog(t, claim)
		pod := reservePod("cached-pod", claim)
		state := f.filter(t, pod, nil)
		before := extractions()
		if before != 1 {
			t.Errorf("Filter extraction calls = %d, want exactly one for the selected pool/request", before)
		}
		// This lister is used by claim-accounting extraction, not the publication fence.
		lister := &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{f.metadata, f.devices}}
		f.tc.draManager.resourceClaimTracker.affinity.slices = lister
		f.reserve(t, state, pod)
		if after := extractions(); after != before {
			t.Errorf("Reserve called CEL extraction again: before=%d, after=%d", before, after)
		}
		if calls := lister.callCount(); calls != 0 {
			t.Errorf("Reserve repeated claim-affinity extraction (%d accounting metadata lookups); reuse Filter's keys", calls)
		}
		reserveAssertState(t, f.snapshot(t), "blue", "5")
	})

	t.Run("same-claim-different-allocation-identity", func(t *testing.T) {
		claim := reserveClaim("racing-claim", "blue")
		f := newReserveFixture(t, true, claim)
		pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
		sa, sb := f.filter(t, pa, nil), f.filter(t, pb, nil)
		a, err := getStateData(sa)
		if err != nil {
			t.Fatal(err)
		}
		b, err := getStateData(sb)
		if err != nil {
			t.Fatal(err)
		}
		if reflect.DeepEqual(a.nodeAllocations[workerNode.Name].allocationResults, b.nodeAllocations[workerNode.Name].allocationResults) {
			t.Fatal("independent Filter attempts did not produce different sharing allocation identities")
		}
		f.reserve(t, sa, pa)
		before := f.snapshot(t)
		tokensBefore := f.pendingTokens(claim.UID)
		if want := sets.New(reserveStateToken(t, sa, claim.UID)); !tokensBefore.Equal(want) {
			t.Fatalf("tokens after first Reserve = %v, want %v", sets.List(tokensBefore), sets.List(want))
		}
		if status := f.tc.p.Reserve(f.ctx, sb, pb, workerNode.Name); status.Code() != fwk.Unschedulable {
			t.Errorf("Reserve = %v, want Unschedulable for a different allocation of the same claim", status)
		}
		if got := f.pendingTokens(claim.UID); !got.Equal(tokensBefore) {
			t.Errorf("rejected Reserve changed tokens to %v, want %v", sets.List(got), sets.List(tokensBefore))
		}
		f.unreserve(sb, pb)
		f.unreserve(sb, pb)
		reserveAssertUnchanged(t, before, f.snapshot(t))
		if got := f.pendingTokens(claim.UID); !got.Equal(tokensBefore) {
			t.Errorf("rejected-attempt Unreserve changed tokens to %v, want %v", sets.List(got), sets.List(tokensBefore))
		}
		f.unreserve(sa, pa)
		reserveAssertState(t, f.snapshot(t), "", "")
		if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
			t.Error("releasing the only legitimate owner left a pending allocation")
		}
	})

	for _, equalRV := range []bool{false, true} {
		t.Run(fmt.Sprintf("pending-replaced-by-observed-different-allocation/equal-rv=%t", equalRV), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			claim := reserveClaim("replaced-pending", "blue")
			f := newReserveFixture(t, true, claim)
			devices := f.devices.DeepCopy()
			other := devices.Spec.Devices[0].DeepCopy()
			other.Name = "other"
			devices.Spec.Devices = append(devices.Spec.Devices, *other)
			f.updateSlice(t, devices)

			group := framework.NewCycleState()
			pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
			sa := f.filter(t, pa, group)
			f.reserve(t, sa, pa)
			sb := f.filter(t, pb, group)
			f.unreserve(sa, pa)

			replacement := sharingAffinityAccountingClaim(claim.UID, "blue")
			replacement.Namespace, replacement.Name = claim.Namespace, claim.Name
			if equalRV {
				replacement.ResourceVersion = claim.ResourceVersion
			} else {
				replacement.ResourceVersion = "2"
			}
			replacement.Status.Allocation.Devices.Results[0].Device = "other"
			replacement.Status.Allocation.Devices.Results[0].ShareID = ptr.To(types.UID("replacement-share"))
			assumecache.DeleteTestObject(f.tc.draManager.resourceClaimTracker.cache, claim)
			assumecache.AddTestObject(f.tc.draManager.resourceClaimTracker.cache, replacement)
			reconcileWaitForIdle(t, f.ctx, f.tc.draManager.resourceClaimTracker)

			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, sb, pb, workerNode.Name)
			if status.Code() != fwk.Unschedulable {
				t.Fatalf("Reserve = %v, want Unschedulable for observed replacement with different allocation", status)
			}
			if equalRV && !strings.Contains(status.Message(), errAffinityConflict.Error()) {
				t.Errorf("Reserve status %q does not report %v for equal-RV replacement", status.Message(), errAffinityConflict)
			}
			if pending := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); pending != nil {
				t.Fatalf("rejected replacement Reserve republished pending allocation: %#v", pending)
			}
			reserveAssertUnchanged(t, before, f.snapshot(t))
		})
	}

	for _, persisted := range []bool{false, true} {
		t.Run(fmt.Sprintf("pending-owner-disappeared/persisted=%t", persisted), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			claim := reserveClaim("shared", "blue")
			f, extractions := newReserveFixtureWithExtractionLog(t, claim)
			pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
			group := framework.NewCycleState()
			sa := f.filter(t, pa, group)
			f.reserve(t, sa, pa)
			original := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID).DeepCopy()
			beforePendingFilter := extractions()
			sb := f.filter(t, pb, group)
			if after := extractions(); after != beforePendingFilter {
				t.Errorf("pending-claim Filter repeated extraction: before=%d, after=%d", beforePendingFilter, after)
			}
			originalAffinity, found := f.tc.draManager.resourceClaimTracker.GetPendingAffinity(claim.UID)
			if !found || len(originalAffinity.provenance) == 0 {
				t.Fatalf("pending entry lacks affinity provenance: %#v", originalAffinity)
			}
			pendingState, err := getStateData(sb)
			if err != nil {
				t.Fatal(err)
			}
			// Request/result index, pool generation, digest, flags and values
			// must survive the pending-only Filter unchanged.
			if plan := pendingState.nodeAllocations[workerNode.Name].affinityByClaim[0]; !reflect.DeepEqual(plan.devices, originalAffinity.provenance) {
				t.Errorf("pending-only Filter provenance = %#v, want %#v", plan.devices, originalAffinity.provenance)
			}

			t.Run("handcrafted-provenance-generation-and-digest", func(t *testing.T) {
				for _, field := range []string{"generation", "digest"} {
					t.Run(field, func(t *testing.T) {
						claim := reserveClaim("bad-"+field, "blue")
						f := newReserveFixture(t, true, claim)
						pod := reservePod("bad-"+field+"-pod", claim)
						state := f.filter(t, pod, nil)
						data, err := getStateData(state)
						if err != nil {
							t.Fatal(err)
						}
						plan := data.nodeAllocations[workerNode.Name].affinityByClaim[0]
						if len(plan.devices) != 1 {
							t.Fatalf("plan provenance = %#v, want one entry", plan.devices)
						}
						if field == "generation" {
							plan.devices[0].Pool.Generation++
						} else {
							plan.devices[0].Digest = "wrong-digest"
						}
						data.nodeAllocations[workerNode.Name].affinityByClaim[0] = plan
						before := f.snapshot(t)
						status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
						if status.Code() != fwk.Unschedulable {
							t.Fatalf("Reserve = %v, want Unschedulable stale provenance", status)
						}
						reserveAssertUnchanged(t, before, f.snapshot(t))
					})
				}
			})

			t.Run("metadata-visible-handler-blocked", func(t *testing.T) {
				for _, added := range []bool{false, true} {
					t.Run(fmt.Sprintf("added=%t", added), func(t *testing.T) {
						claim := reserveClaim("blocked-handler", "blue")
						f := newReserveFixture(t, true, claim)
						if added {
							metadata := f.metadata.DeepCopy()
							metadata.Spec.SharingAffinity = nil
							f.updateSlice(t, metadata)
						}
						pod := reservePod("blocked-handler-pod", claim)
						state := f.filter(t, pod, nil)
						block := make(chan struct{})
						var unblock sync.Once
						f.tc.draManager.resourceClaimTracker.publishBeforeCommitHook = func(string, string) {
							<-block
						}
						defer func() {
							unblock.Do(func() { close(block) })
						}()
						changed := f.metadata.DeepCopy()
						if added {
							changed.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityTestExtractor("subnet", `"red"`)}
						} else {
							changed.Spec.SharingAffinity[0].Expression = `"red"`
						}
						current, err := f.tc.client.ResourceV1().ResourceSlices().Get(f.ctx, changed.Name, metav1.GetOptions{})
						if err != nil {
							t.Fatalf("get ResourceSlice: %v", err)
						}
						changed.UID, changed.ResourceVersion = current.UID, current.ResourceVersion
						if _, err := f.tc.client.ResourceV1().ResourceSlices().Update(f.ctx, changed, metav1.UpdateOptions{}); err != nil {
							t.Fatalf("update ResourceSlice: %v", err)
						}
						if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
							slices, err := f.tc.draManager.ResourceSlices().ListWithDeviceTaintRules()
							for _, current := range slices {
								if current.Name == changed.Name && reflect.DeepEqual(current.Spec, changed.Spec) {
									return true, nil
								}
							}
							return false, err
						}); err != nil {
							t.Fatalf("wait for lister-visible update: %v", err)
						}
						before := f.snapshot(t)
						status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
						if status.Code() != fwk.Unschedulable {
							t.Fatalf("Reserve = %v, want Unschedulable from list mismatch while handler blocked", status)
						}
						reserveAssertUnchanged(t, before, f.snapshot(t))
						unblock.Do(func() { close(block) })
					})
				}
			})

			t.Run("metadata-update-during-reserve", func(t *testing.T) {
				for _, hook := range []string{"after-list", "after-commit"} {
					t.Run(hook, func(t *testing.T) {
						claim := reserveClaim("during-"+hook, "blue")
						f := newReserveFixture(t, true, claim)
						pod := reservePod("during-"+hook+"-pod", claim)
						state := f.filter(t, pod, nil)
						changed := f.metadata.DeepCopy()
						changed.Spec.SharingAffinity[0].Expression = `"red"`
						changeOnce := sync.OnceFunc(func() { f.updateSlice(t, changed) })
						switch hook {
						case "after-list":
							f.tc.draManager.resourceClaimTracker.reserveAfterListHook = changeOnce
							defer func() { f.tc.draManager.resourceClaimTracker.reserveAfterListHook = nil }()
						case "after-commit":
							f.tc.draManager.resourceClaimTracker.reserveAfterCommitHook = changeOnce
							defer func() { f.tc.draManager.resourceClaimTracker.reserveAfterCommitHook = nil }()
						}
						before := f.snapshot(t)
						status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
						if status.Code() != fwk.Unschedulable {
							t.Fatalf("Reserve = %v, want Unschedulable after metadata mutation inside Reserve", status)
						}
						reserveAssertUnchanged(t, before, f.snapshot(t))
						if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
							t.Fatal("stale Reserve left pending allocation behind")
						}
					})
				}
			})

			t.Run("metadata-update-after-commit-handler-blocked", func(t *testing.T) {
				claim := reserveClaim("after-commit-blocked", "blue")
				f := newReserveFixture(t, true, claim)
				pod := reservePod("after-commit-blocked-pod", claim)
				state := f.filter(t, pod, nil)
				tracker := f.tc.draManager.resourceClaimTracker
				block := make(chan struct{})
				var unblock sync.Once
				defer unblock.Do(func() { close(block) })
				tracker.publishMutex.Lock()
				tracker.publishBeforeCommitHook = func(string, string) { <-block }
				tracker.publishMutex.Unlock()
				changed := f.metadata.DeepCopy()
				changed.Spec.SharingAffinity[0].Expression = `"red"`
				var epochAtCommit uint64
				key := poolName{accountingDriver, accountingPool}
				tracker.reserveAfterCommitHook = sync.OnceFunc(func() {
					tracker.allocatedDevices.mutex.RLock()
					epochAtCommit = tracker.metadata.pools[key].epoch
					tracker.allocatedDevices.mutex.RUnlock()
					// Waits only for lister visibility; the registry handler stays blocked.
					f.updateSlice(t, changed)
				})
				defer func() { tracker.reserveAfterCommitHook = nil }()
				before := f.snapshot(t)
				status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
				tracker.allocatedDevices.mutex.RLock()
				epochAfter := tracker.metadata.pools[key].epoch
				tracker.allocatedDevices.mutex.RUnlock()
				if epochAfter != epochAtCommit {
					t.Fatalf("registry published during Reserve (epoch %d -> %d); handler was meant to stay blocked", epochAtCommit, epochAfter)
				}
				if status.Code() != fwk.Unschedulable {
					t.Fatalf("Reserve = %v, want Unschedulable from post-commit relist while handler blocked", status)
				}
				reserveAssertUnchanged(t, before, f.snapshot(t))
				if tracker.GetPendingAllocation(claim.UID) != nil {
					t.Fatal("post-commit relist rejection left pending allocation behind")
				}
				unblock.Do(func() { close(block) })
			})

			t.Run("metadata-update-after-relist", func(t *testing.T) {
				claim := reserveClaim("post-relist", "blue")
				f := newReserveFixture(t, true, claim)
				pod := reservePod("post-relist-pod", claim)
				state := f.filter(t, pod, nil)
				f.reserve(t, state, pod)
				changed := f.metadata.DeepCopy()
				changed.Spec.SharingAffinity[0].Expression = `"red"`
				f.updateSlice(t, changed)
				reconcileWaitForIdle(t, f.ctx, f.tc.draManager.resourceClaimTracker)
				assertSharingAffinityAccountingState(t, f.snapshot(t), structured.MakeDeviceID(accountingDriver, accountingPool, accountingDevice), structured.AffinityStatusUnreconstructable, nil)
				f.unreserve(state, pod)
				reserveAssertState(t, f.snapshot(t), "", "")
			})
			if persisted {
				if status := f.tc.p.PreBind(f.ctx, sa, pa, workerNode.Name); !status.IsSuccess() {
					t.Fatalf("PreBind original owner: %v", status)
				}
			} else {
				f.unreserve(sa, pa)
			}
			before := f.snapshot(t)
			lister := &sharingAffinityTestLister{slices: []*resourceapi.ResourceSlice{f.metadata, f.devices}}
			f.tc.draManager.resourceClaimTracker.affinity.slices = lister
			beforePendingReserve := extractions()
			status := f.tc.p.Reserve(f.ctx, sb, pb, workerNode.Name)
			if !persisted {
				if after := extractions(); after != beforePendingReserve {
					t.Errorf("pending-plan Reserve repeated CEL extraction: before=%d, after=%d", beforePendingReserve, after)
				}
			}
			if calls := lister.callCount(); calls != 0 {
				t.Errorf("pending-plan Reserve repeated accounting extraction (%d metadata lookups)", calls)
			}
			if persisted {
				if status.Code() != fwk.Unschedulable {
					t.Errorf("Reserve = %v, want Unschedulable after pending allocation was persisted", status)
				}
				reserveAssertUnchanged(t, before, f.snapshot(t))
			} else {
				if !status.IsSuccess() {
					t.Fatalf("Reserve must reacquire released, still-unallocated pending claim: %v", status)
				}
				if got := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); !reflect.DeepEqual(got, original) {
					t.Errorf("pending claim allocation identity changed: got %#v, want %#v", got, original)
				}
				if got, found := f.tc.draManager.resourceClaimTracker.GetPendingAffinity(claim.UID); !found || !reflect.DeepEqual(got.provenance, originalAffinity.provenance) {
					t.Errorf("recreated pending provenance = %#v, want %#v", got.provenance, originalAffinity.provenance)
				}
				reserveAssertState(t, f.snapshot(t), "blue", "5")
			}
		})
	}

	for _, mutation := range []string{"resource-version", "uid", "allocated-elsewhere", "deleted"} {
		t.Run("stale-claim/"+mutation, func(t *testing.T) {
			claim := reserveClaim("stale", "blue")
			f := newReserveFixture(t, true, claim)
			pod := reservePod("stale-pod", claim)
			state := f.filter(t, pod, nil)
			changed := claim.DeepCopy()
			changed.ResourceVersion = "2"
			switch mutation {
			case "uid":
				changed.UID = "replacement"
			case "allocated-elsewhere":
				changed.Status.Allocation = sharingAffinityAccountingClaim(claim.UID, "blue").Status.Allocation
			}
			if mutation == "deleted" {
				assumecache.DeleteTestObject(f.tc.draManager.resourceClaimTracker.cache, claim)
			} else {
				assumecache.UpdateTestObject(f.tc.draManager.resourceClaimTracker.cache, changed)
			}
			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if mutation == "resource-version" {
				// The claim queueing hint skips RV-only updates, so only an
				// Error (backoff retry) re-queues the pod without another event.
				if status.Code() != fwk.Error {
					t.Errorf("Reserve = %v, want Error for ResourceVersion-only change", status)
				}
			} else if status.Code() != fwk.Unschedulable {
				t.Errorf("Reserve = %v, want Unschedulable for stale claim", status)
			}
			if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
				t.Error("stale Filter result published an in-flight allocation")
			}
			reserveAssertUnchanged(t, before, f.snapshot(t))
		})
	}

	for _, mutation := range []string{"expression", "name", "group", "kind", "generation", "added", "removed", "deleted"} {
		t.Run("stale-metadata/"+mutation, func(t *testing.T) {
			claim := reserveClaim("stale", "blue")
			f := newReserveFixture(t, true, claim)
			pod := reservePod("stale-pod", claim)
			if mutation == "added" {
				unconstrained := f.metadata.DeepCopy()
				unconstrained.Spec.SharingAffinity = nil
				f.updateSlice(t, unconstrained)
			}
			state := f.filter(t, pod, nil)
			changed := f.metadata.DeepCopy()
			switch mutation {
			case "expression":
				changed.Spec.SharingAffinity[0].Expression = `"red"`
			case "name":
				changed.Spec.SharingAffinity[0].Name = "renamed"
			case "group":
				changed.Spec.SharingAffinity[0].APIGroup = "*"
			case "kind":
				changed.Spec.SharingAffinity[0].Kind = "*"
			case "generation":
				changed.Spec.Pool.Generation++
				devices := f.devices.DeepCopy()
				devices.Spec.Pool.Generation++
				f.updateSlice(t, devices)
			case "removed":
				changed.Spec.SharingAffinity = nil
			}
			if mutation == "deleted" {
				if err := f.tc.client.ResourceV1().ResourceSlices().Delete(f.ctx, changed.Name, metav1.DeleteOptions{}); err != nil {
					t.Fatal(err)
				}
				if err := wait.PollUntilContextTimeout(f.ctx, time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
					slices, err := f.tc.draManager.ResourceSlices().ListWithDeviceTaintRules()
					for _, current := range slices {
						if current.Name == changed.Name {
							return false, err
						}
					}
					return true, err
				}); err != nil {
					t.Fatal(err)
				}
			} else {
				f.updateSlice(t, changed)
			}
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if status.Code() != fwk.Unschedulable {
				t.Errorf("Reserve = %v, want Unschedulable for changed metadata", status)
			}
			if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
				t.Error("stale Filter result published an in-flight allocation")
			}
			reserveAssertState(t, f.snapshot(t), "", "")
		})
	}
}

func TestSharingAffinityUnreserve(t *testing.T) {
	t.Run("last-owner-release-idempotent", func(t *testing.T) {
		claim := reserveClaim("single", "blue")
		f := newReserveFixture(t, true, claim)
		pod := reservePod("single-pod", claim)
		state := f.filter(t, pod, nil)
		f.reserve(t, state, pod)
		c := f.tc.draManager.resourceClaimTracker
		token := reserveStateToken(t, state, claim.UID)
		revision := c.allocatedDevices.Revision()
		if deleted := c.ReleaseClaimPendingAllocation(claim.UID, token); !deleted {
			t.Fatal("last tokenized release did not delete the in-flight entry")
		}
		if got := c.allocatedDevices.Revision(); got <= revision {
			t.Errorf("revision after tokenized in-flight removal = %d, want > %d", got, revision)
		}
		revision = c.allocatedDevices.Revision()
		if deleted := c.ReleaseClaimPendingAllocation(claim.UID, token); deleted {
			t.Error("repeated tokenized release reported a deletion")
		}
		if got := c.allocatedDevices.Revision(); got != revision {
			t.Errorf("revision after repeated release = %d, want unchanged %d", got, revision)
		}
		f.unreserve(state, pod)
		reserveAssertState(t, f.snapshot(t), "", "")
		f.unreserve(state, pod)
		reserveAssertState(t, f.snapshot(t), "", "")
	})

	for _, observed := range []bool{false, true} {
		t.Run(fmt.Sprintf("other-owner-preserved/observed=%t", observed), func(t *testing.T) {
			a, b := reserveClaim("a", "blue"), reserveClaim("b", "blue")
			f := newReserveFixture(t, true, a, b)
			pa, pb := reservePod("pod-a", a), reservePod("pod-b", b)
			sa, sb := f.filter(t, pa, nil), f.filter(t, pb, nil)
			if observed {
				owner := sharingAffinityAccountingClaim(a.UID, "blue")
				owner.ResourceVersion = "2"
				assumecache.UpdateTestObject(f.tc.draManager.resourceClaimTracker.cache, owner)
			} else {
				f.reserve(t, sa, pa)
			}
			before := f.snapshot(t)
			f.reserve(t, sb, pb)
			f.unreserve(sb, pb)
			reserveAssertUnchanged(t, before, f.snapshot(t))
			f.unreserve(sb, pb)
			reserveAssertUnchanged(t, before, f.snapshot(t))
		})
	}

	t.Run("podgroup-sharer-preserved-on-repeated-unreserve", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
		claim := reserveClaim("shared", "blue")
		f := newReserveFixture(t, true, claim)
		pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
		for _, pod := range []*v1.Pod{pa, pb} {
			pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("reservation-group")}
		}
		group := framework.NewCycleState()
		sa := f.filter(t, pa, group)
		f.reserve(t, sa, pa)
		sb := f.filter(t, pb, group)
		f.reserve(t, sb, pb)
		reserveAssertState(t, f.snapshot(t), "blue", "5")
		f.unreserve(sa, pa)
		reserveAssertState(t, f.snapshot(t), "blue", "5")
		f.unreserve(sa, pa)
		reserveAssertState(t, f.snapshot(t), "blue", "5")
		if pending := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); pending == nil {
			t.Error("repeated cleanup of pod A removed pod B's pending allocation")
		}
		f.unreserve(sb, pb)
		reserveAssertState(t, f.snapshot(t), "", "")
	})

	t.Run("attempt-without-reserve-preserves-podgroup-owner", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
		claim := reserveClaim("shared", "blue")
		f := newReserveFixture(t, true, claim)
		pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
		group := framework.NewCycleState()
		sa := f.filter(t, pa, group)
		f.reserve(t, sa, pa)
		sb := f.filter(t, pb, group)
		before := f.snapshot(t)
		// A preceding Reserve plugin may fail before this plugin acquires anything.
		f.unreserve(sb, pb)
		reserveAssertUnchanged(t, before, f.snapshot(t))
		if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) == nil {
			t.Error("cleanup without an acquired reservation removed another pod's ownership")
		}
	})

	for _, pendingJoin := range []bool{false, true} {
		t.Run(fmt.Sprintf("partial-multi-claim-reserve-rollback/pending-join=%t", pendingJoin), func(t *testing.T) {
			if pendingJoin {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			}
			a, b := reserveClaim("first", "blue"), reserveClaim("second", "blue")
			a.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["accounting.example.com"].other`},
			}}
			b.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{
				CEL: &resourceapi.CELDeviceSelector{Expression: `!device.attributes["accounting.example.com"].other`},
			}}
			f := newReserveFixture(t, true, a, b)
			devices := f.devices.DeepCopy()
			other := devices.Spec.Devices[0].DeepCopy()
			other.Name = "other"
			other.Attributes["other"] = resourceapi.DeviceAttribute{BoolValue: ptr.To(true)}
			devices.Spec.Devices = append(devices.Spec.Devices, *other)
			f.updateSlice(t, devices)
			var group fwk.PodGroupCycleState
			var ownerPod *v1.Pod
			var ownerState fwk.CycleState
			if pendingJoin {
				group = framework.NewCycleState()
				ownerPod = reservePod("pending-owner", a)
				ownerState = f.filter(t, ownerPod, group)
				f.reserve(t, ownerState, ownerPod)
			}
			ownerTokens := f.pendingTokens(a.UID)
			pod := reservePod("two-claims", a, b)
			state := f.filter(t, pod, group)
			owner := sharingAffinityAccountingClaim("racer", "red")
			assumecache.AddTestObject(f.tc.draManager.resourceClaimTracker.cache, owner)
			before := f.snapshot(t)
			status := f.tc.p.Reserve(f.ctx, state, pod, workerNode.Name)
			if status.Code() != fwk.Unschedulable {
				t.Errorf("Reserve = %v, want Unschedulable for second-claim conflict", status)
			}
			reserveAssertUnchanged(t, before, f.snapshot(t))
			for _, claim := range []*resourceapi.ResourceClaim{a, b} {
				wantPending := pendingJoin && claim == a
				if gotPending := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil; gotPending != wantPending {
					t.Errorf("after failed Reserve claim %s pending = %t, want %t", claim.Name, gotPending, wantPending)
				}
			}
			f.unreserve(state, pod)
			f.unreserve(state, pod)
			reserveAssertUnchanged(t, before, f.snapshot(t))
			if got := f.pendingTokens(a.UID); !got.Equal(ownerTokens) {
				t.Errorf("failed attempt left tokens %v for claim %s, want only %v", sets.List(got), a.Name, sets.List(ownerTokens))
			}
			if got := f.pendingTokens(b.UID); got != nil {
				t.Errorf("failed attempt left tokens %v for claim %s", sets.List(got), b.Name)
			}
			if pendingJoin {
				f.unreserve(ownerState, ownerPod)
				if f.tc.draManager.ResourceClaims().GetPendingAllocation(a.UID) != nil {
					t.Error("releasing the original owner did not drain the joined entry; failed attempt token leaked")
				}
			}
		})
	}

	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("failed-bind-after-successful-prebind/gate=%t", enabled), func(t *testing.T) {
			claim := reserveClaim("persisted", "blue")
			f := newReserveFixture(t, enabled, claim)
			pod := reservePod("failed-bind-pod", claim)
			state := f.filter(t, pod, nil)
			f.reserve(t, state, pod)
			if status := f.tc.p.PreBind(f.ctx, state, pod, workerNode.Name); !status.IsSuccess() {
				t.Fatalf("PreBind: %v", status)
			}
			before := f.snapshot(t)
			f.unreserve(state, pod)
			f.unreserve(state, pod)
			reserveAssertUnchanged(t, before, f.snapshot(t))
			stored, err := f.tc.client.ResourceV1().ResourceClaims(claim.Namespace).Get(f.ctx, claim.Name, metav1.GetOptions{})
			if err != nil || stored.Status.Allocation == nil || len(stored.Status.ReservedFor) != 0 {
				t.Errorf("failed Bind must retain allocation but release pod reservation: %#v, err=%v", stored, err)
			}
			subnet := ""
			if enabled {
				subnet = "blue"
			}
			reserveAssertState(t, f.snapshot(t), subnet, "5")
		})

		t.Run(fmt.Sprintf("failed-binding/gate=%t", enabled), func(t *testing.T) {
			claim := reserveClaim("failed", "blue")
			f := newReserveFixture(t, enabled, claim)
			pod := reservePod("failed-pod", claim)
			state := f.filter(t, pod, nil)
			before := f.snapshot(t)
			f.reserve(t, state, pod)
			f.tc.client.PrependReactor("update", "resourceclaims", func(action cgotesting.Action) (bool, apiruntime.Object, error) {
				if action.GetSubresource() == "status" {
					return true, nil, errors.New("injected claim status persistence failure")
				}
				return false, nil, nil
			})
			if status := f.tc.p.PreBind(f.ctx, state, pod, workerNode.Name); status.Code() != fwk.Error {
				t.Fatalf("PreBind = %v, want injected persistence Error", status)
			}
			f.unreserve(state, pod)
			f.unreserve(state, pod)
			reserveAssertUnchanged(t, before, f.snapshot(t))
			stored, err := f.tc.client.ResourceV1().ResourceClaims(claim.Namespace).Get(f.ctx, claim.Name, metav1.GetOptions{})
			if err != nil || stored.Status.Allocation != nil {
				t.Errorf("failed persistence allocated claim: %#v, err=%v", stored, err)
			}
		})

		for _, grouped := range []bool{false, true} {
			t.Run(fmt.Sprintf("partial-prebind-persisted-claim/gate=%t/group=%t", enabled, grouped), func(t *testing.T) {
				if grouped {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
				}
				a, b := reserveClaim("persisted", "blue"), reserveClaim("failed", "blue")
				c, d := reserveClaim("incompatible", "red"), reserveClaim("compatible", "blue")
				f := newReserveFixture(t, enabled, a, b, c, d)
				pod := reservePod("partial-pod", a, b)
				var group fwk.PodGroupCycleState
				if grouped {
					group = framework.NewCycleState()
					pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("reservation-group")}
				}
				state := f.filter(t, pod, group)
				f.reserve(t, state, pod)
				// Separate API storage withholds informer events while retaining real fake-client
				// status persistence. Only the plugin's AssumeClaimAfterAPICall updates its cache.
				backend := fake.NewSimpleClientset(a.DeepCopy(), b.DeepCopy())
				reaction := cgotesting.ObjectReaction(backend.Tracker())
				f.tc.client.PrependReactor("*", "resourceclaims", func(action cgotesting.Action) (bool, apiruntime.Object, error) {
					if update, ok := action.(cgotesting.UpdateAction); ok {
						claim := update.GetObject().(*resourceapi.ResourceClaim)
						if claim.Name == b.Name && action.GetSubresource() == "status" {
							return true, nil, errors.New("injected second claim persistence failure")
						}
						claim.ResourceVersion = "2"
					}
					return reaction(action)
				})
				if status := f.tc.p.PreBind(f.ctx, state, pod, workerNode.Name); status.Code() != fwk.Error ||
					!strings.Contains(status.Message(), "injected second claim persistence failure") {
					t.Fatalf("PreBind = %v, want second claim persistence Error", status)
				}
				tracker := f.tc.draManager.resourceClaimTracker
				cached, err := tracker.Get(a.Namespace, a.Name)
				if err != nil || cached.Status.Allocation == nil {
					t.Fatalf("first claim was not assumed after persistence: %#v, err=%v", cached, err)
				}
				apiCopy, err := tracker.cache.GetAPIObj(a.Namespace + "/" + a.Name)
				if err != nil || apiCopy.(*resourceapi.ResourceClaim).Status.Allocation != nil {
					t.Fatalf("test did not withhold allocated informer object: %#v, err=%v", apiCopy, err)
				}
				data, err := getStateData(state)
				if err != nil {
					t.Fatal(err)
				}
				data.mutex.Lock()
				persisted := sets.List(data.persistedClaims)
				data.mutex.Unlock()
				if want := []types.UID{a.UID}; !reflect.DeepEqual(persisted, want) {
					t.Errorf("persistedClaims after partial PreBind = %v, want %v", persisted, want)
				}
				f.unreserve(state, pod)
				f.unreserve(state, pod)
				cached, err = tracker.Get(a.Namespace, a.Name)
				if err != nil || cached.Status.Allocation == nil {
					t.Errorf("Unreserve restored a persisted claim to an unallocated informer object: %#v, err=%v", cached, err)
				}
				subnet := ""
				if enabled {
					subnet = "blue"
				}
				reserveAssertState(t, f.snapshot(t), subnet, "5")
				for _, claim := range []*resourceapi.ResourceClaim{a, b} {
					if tracker.GetPendingAllocation(claim.UID) != nil {
						t.Errorf("Unreserve left claim %s in-flight", claim.Name)
					}
				}
				if grouped {
					groupState, err := getPodGroupStateData(state)
					if err != nil {
						t.Fatal(err)
					}
					if _, found := groupState.pendingAllocations[a.UID]; !found {
						t.Error("Unreserve removed the persisted allocation from PodGroup state")
					}
					if _, found := groupState.pendingAllocations[b.UID]; found {
						t.Error("Unreserve retained the failed allocation in PodGroup state")
					}
					member := reservePod("next-member", a)
					member.Spec.SchedulingGroup = pod.Spec.SchedulingGroup.DeepCopy()
					cycle := f.filter(t, member, group)
					memberState, err := getStateData(cycle)
					if err != nil {
						t.Fatal(err)
					}
					if memberState.claims.get(0).Status.Allocation == nil {
						t.Error("next PodGroup member must reuse the persisted allocation")
					}
				}
				for _, claim := range []*resourceapi.ResourceClaim{c, d} {
					candidate := reservePod("pod-"+claim.Name, claim)
					cycle := framework.NewCycleState()
					if _, status := f.tc.p.PreFilter(f.ctx, cycle, candidate, f.tc.nodeInfos); !status.IsSuccess() {
						t.Fatalf("fresh PreFilter for %s: %v", claim.Name, status)
					}
					status := f.tc.p.Filter(f.ctx, cycle, candidate, f.tc.nodeInfos[0])
					if enabled && claim == c {
						if status.Code() != fwk.Unschedulable {
							t.Errorf("incompatible claim Filter = %v, want Unschedulable while persisted allocation remains", status)
						}
					} else if !status.IsSuccess() {
						t.Errorf("compatible or gate-off claim Filter failed: %v", status)
					}
				}
				stored, err := f.tc.client.ResourceV1().ResourceClaims(a.Namespace).Get(f.ctx, a.Name, metav1.GetOptions{})
				if err != nil || stored.Status.Allocation == nil || len(stored.Status.ReservedFor) != 0 {
					t.Fatalf("persisted allocation must remain, without pod reservation: %#v, err=%v", stored, err)
				}
				before := f.snapshot(t)
				assumecache.UpdateTestObject(tracker.cache, stored)
				reserveAssertUnchanged(t, before, f.snapshot(t))
			})
		}
	}

	t.Run("capacity-handoff-remains-five-plus-five", func(t *testing.T) {
		setSharingAffinityAccountingGates(t, true, true)
		tracker := newSharingAffinityAccountingTracker(t, true)
		claim := sharingAffinityAccountingClaim("handoff", "blue")
		signalSharingAffinityAccounting(t, tracker, claim)
		tracker.allocatedDevices.addDevices(claim)
		reserveAssertState(t, gatherSharingAffinityAccounting(t, tracker), "blue", "10")
		if !tracker.MaybeRemoveClaimPendingAllocation(claim.UID, true) {
			t.Fatal("handoff did not remove the pending allocation")
		}
		reserveAssertState(t, gatherSharingAffinityAccounting(t, tracker), "blue", "5")
		if tracker.MaybeRemoveClaimPendingAllocation(claim.UID, true) {
			t.Error("repeated handoff removed another owner")
		}
		reserveAssertState(t, gatherSharingAffinityAccounting(t, tracker), "blue", "5")
	})
}

// A PodGroup whose PreBind wrote the claim and then failed must hand the
// pending allocation off exactly like successful PreBind cleanup (force
// remove), with or without the sharing-affinity gate.
func TestSharingAffinityUnreservePersistedPodGroupClaim(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("gate=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)
			claim := reserveClaim("shared", "blue")
			f := newReserveFixture(t, enabled, claim)
			pa, pb := reservePod("pod-a", claim), reservePod("pod-b", claim)
			for _, pod := range []*v1.Pod{pa, pb} {
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("prebind-group")}
			}
			group := framework.NewCycleState()
			sa := f.filter(t, pa, group)
			f.reserve(t, sa, pa)
			sb := f.filter(t, pb, group)
			f.reserve(t, sb, pb)
			if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) == nil {
				t.Fatal("expected pending allocation after Reserve")
			}

			// Simulate PreBind for pod A succeeding at the API write and
			// failing afterwards.
			data, err := getStateData(sa)
			if err != nil {
				t.Fatalf("getStateData: %v", err)
			}
			data.mutex.Lock()
			if data.persistedClaims == nil {
				data.persistedClaims = sets.New[types.UID]()
			}
			data.persistedClaims.Insert(claim.UID)
			data.mutex.Unlock()

			f.unreserve(sa, pa)
			if got := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); got != nil {
				t.Errorf("persisted claim: pending allocation = %v, want removed for all sharers", got)
			}
			// Idempotent: second Unreserve of A and Unreserve of B must not panic or resurrect state.
			f.unreserve(sa, pa)
			f.unreserve(sb, pb)
			if got := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); got != nil {
				t.Errorf("after all Unreserve: pending allocation = %v, want nil", got)
			}
		})
	}
}

// A pod may reference the same claim under two names; Reserve must acquire
// once so a single Unreserve drains capacity and any affinity lock.
func TestSharingAffinityUnreserveDuplicateClaimReference(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, grouped := range []bool{false, true} {
			t.Run(fmt.Sprintf("gate=%t/podgroup=%t", enabled, grouped), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, grouped)
				claim := reserveClaim("shared", "blue")
				f := newReserveFixture(t, enabled, claim)
				pod := reservePod("dup-pod", claim, claim)
				var group fwk.PodGroupCycleState
				if grouped {
					pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("dup-group")}
					group = framework.NewCycleState()
				}
				state := f.filter(t, pod, group)
				f.reserve(t, state, pod)
				if f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID) == nil {
					t.Fatal("expected pending allocation after Reserve")
				}
				f.unreserve(state, pod)
				if got := f.tc.draManager.ResourceClaims().GetPendingAllocation(claim.UID); got != nil {
					t.Errorf("pending allocation leaked after Unreserve: %v", got)
				}
				after := f.snapshot(t)
				if len(after.AggregatedCapacity) != 0 || len(after.AllocatedSharedDeviceIDs) != 0 {
					t.Errorf("capacity leaked after Unreserve: %#v", after)
				}
				if enabled {
					reserveAssertState(t, after, "", "")
				}
			})
		}
	}
}
