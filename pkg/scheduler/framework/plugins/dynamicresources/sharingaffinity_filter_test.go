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
	"strings"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/utils/ptr"
)

type filterAffinityClasses struct {
	class *resourceapi.DeviceClass
}

func (l filterAffinityClasses) List() ([]*resourceapi.DeviceClass, error) {
	return []*resourceapi.DeviceClass{l.class}, nil
}

func (l filterAffinityClasses) Get(name string) (*resourceapi.DeviceClass, error) {
	if name != l.class.Name {
		return nil, fmt.Errorf("unexpected device class %q", name)
	}
	return l.class, nil
}

type filterAffinityLegacyAllocator struct {
	err     error
	results []resourceapi.AllocationResult
}

type filterAffinityAllocator struct {
	results  []resourceapi.AllocationResult
	affinity []structured.ClaimAffinity
	err      error
	called   bool
}

var _ structured.Allocator = (*filterAffinityAllocator)(nil)
var _ structured.AffinityAllocator = (*filterAffinityAllocator)(nil)

func (a *filterAffinityAllocator) Allocate(context.Context, *v1.Node, []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, error) {
	return a.results, a.err
}

func (a *filterAffinityAllocator) AllocateWithAffinity(context.Context, *v1.Node, []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, []structured.ClaimAffinity, error) {
	a.called = true
	return a.results, a.affinity, a.err
}

// Embedding only the public interfaces deliberately hides optional affinity
// methods which framework test doubles and out-of-tree callers may not have.
type filterAffinityLegacyTracker struct {
	fwk.ResourceClaimTracker
}

type filterAffinityLegacyManager struct {
	fwk.SharedDRAManager
	tracker fwk.ResourceClaimTracker
}

func (m filterAffinityLegacyManager) ResourceClaims() fwk.ResourceClaimTracker {
	return m.tracker
}

func (a filterAffinityLegacyAllocator) Allocate(context.Context, *v1.Node, []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, error) {
	return a.results, a.err
}

func filterAffinityState(claims []*resourceapi.ResourceClaim, allocator structured.Allocator) (fwk.CycleState, *stateData) {
	state := &stateData{
		claims:               newClaimStore(claims, nil, nil),
		allocator:            allocator,
		informationsForClaim: make([]informationForClaim, len(claims)),
		nodeAllocations:      make(map[string]nodeAllocation),
	}
	cycle := framework.NewCycleState()
	cycle.Write(stateKey, state)
	return cycle, state
}

func filterAffinityFixture() (*resourceapi.ResourceClaim, *resourceapi.ResourceSlice, *resourceapi.DeviceClass) {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "claim", Namespace: "default", UID: "claim", ResourceVersion: "1"},
		Spec: resourceapi.ResourceClaimSpec{Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name: "req",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: "network", AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				},
			}},
			Config: []resourceapi.DeviceClaimConfiguration{
				sharingAffinityTestConfig(sharingAffinityTestDriver, `{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"opaque-secret"}`),
			},
		}},
	}
	slice := sharingAffinityTestSlice("devices", sharingAffinityTestDriver, sharingAffinityTestPool, 1, 1,
		resourceapi.SharingAffinityExtractor{
			Name: "subnet", APIGroup: "config.example.com", Kind: "Network",
			Expression: `has(parameters.subnet) ? parameters.subnet : ""`,
		})
	slice.Spec.AllNodes = ptr.To(true)
	slice.Spec.Devices = []resourceapi.Device{{Name: "device", AllowMultipleAllocations: ptr.To(true)}}
	class := &resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}}
	return claim, slice, class
}

// Exercise Filter itself, not just the extractor's classification.
func TestSharingAffinityFilter(t *testing.T) {
	type testCase struct {
		name       string
		expression string
		raw        string
		mutate     func(*resourceapi.ResourceClaim, *resourceapi.ResourceSlice, *resourceapi.DeviceClass)
		lock       *structured.DeviceAffinityState
		want       fwk.Code
		cause      error
		message    string
	}
	cases := []testCase{
		{name: "status/clean", want: fwk.Success},
		{name: "status/nonmatch", expression: `""`, want: fwk.Unschedulable, message: "claim configuration does not produce a value for every sharing affinity extractor"},
		{name: "status/partial-satisfaction", mutate: func(_ *resourceapi.ResourceClaim, s *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			s.Spec.SharingAffinity = append(s.Spec.SharingAffinity, sharingAffinityTestExtractor("other", `""`))
		}, want: fwk.Unschedulable},
		{name: "status/no-config", mutate: func(c *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			c.Spec.Devices.Config = nil
		}, want: fwk.Unschedulable},
		{name: "status/invalid-pool", mutate: func(_ *resourceapi.ResourceClaim, s *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			s.Spec.SharingAffinity = append(s.Spec.SharingAffinity, s.Spec.SharingAffinity[0])
		}, want: fwk.Unschedulable, message: "invalid resource pools"},
		{name: "status/runtime-fatal/cel", expression: `string(int(parameters.subnet))`, want: fwk.Error, cause: sharingaffinity.ErrCELEval},
		{name: "status/runtime-fatal/missing-field", expression: `parameters[parameters.subnet]`, want: fwk.Error, cause: sharingaffinity.ErrMissingField},
		{name: "status/runtime-fatal/non-string", expression: `parameters.subnet`, raw: `{"apiVersion":"config.example.com/v1","kind":"Network","subnet":42,"private":"opaque-secret"}`, want: fwk.Error, cause: sharingaffinity.ErrNonStringReturn},
		{name: "status/runtime-fatal/parameters", raw: `{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"opaque-secret",}`, want: fwk.Error, cause: sharingaffinity.ErrParameters},
		{name: "status/claim-inconsistent", mutate: func(c *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			c.Spec.Devices.Config = append(c.Spec.Devices.Config, sharingAffinityTestConfig(sharingAffinityTestDriver,
				`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"other-secret"}`))
		}, want: fwk.UnschedulableAndUnresolvable, message: "claim self-inconsistent"},
		{name: "lock/equal", lock: &structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "opaque-secret"}}, want: fwk.Success},
		{name: "lock/mismatch", lock: &structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "other-secret"}}, want: fwk.Unschedulable},
		{name: "lock/superset", lock: &structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "opaque-secret", "extra": "value"}}, want: fwk.Unschedulable},
		{name: "lock/unreconstructable", lock: &structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}, want: fwk.Unschedulable},
		{name: "metadata/permissive", mutate: func(c *resourceapi.ResourceClaim, s *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			s.Spec.SharingAffinity = nil
			c.Spec.Devices.Config = nil
		}, want: fwk.Success},
		{name: "metadata/permissive-but-fenced", mutate: func(_ *resourceapi.ResourceClaim, s *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			s.Spec.SharingAffinity = nil
		}, lock: &structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}, want: fwk.Unschedulable},
		{name: "metadata/version-independent", raw: `{"apiVersion":"config.example.com/v999","kind":"Network","subnet":"opaque-secret"}`, want: fwk.Success},
		{name: "metadata/extra-fields-ignored", raw: `{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"opaque-secret","ignored":{"private":[1,2,3]}}`, want: fwk.Success},
		{name: "metadata/group-mismatch", raw: `{"apiVersion":"other.example.com/v1","kind":"Network","subnet":"opaque-secret"}`, want: fwk.Unschedulable},
		{name: "metadata/kind-mismatch", raw: `{"apiVersion":"config.example.com/v1","kind":"Other","subnet":"opaque-secret"}`, want: fwk.Unschedulable},
		{name: "metadata/wildcard", mutate: func(_ *resourceapi.ResourceClaim, s *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			s.Spec.SharingAffinity[0].APIGroup = "*"
			s.Spec.SharingAffinity[0].Kind = "*"
		}, raw: `{"subnet":"opaque-secret"}`, want: fwk.Success},
		{name: "scope/wrong-driver-before-decode", mutate: func(c *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			c.Spec.Devices.Config = append(c.Spec.Devices.Config, sharingAffinityTestConfig("other.example.com", `{"malformed":`))
		}, want: fwk.Success},
		{name: "scope/wrong-request-before-decode", mutate: func(c *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, _ *resourceapi.DeviceClass) {
			c.Spec.Devices.Config = append(c.Spec.Devices.Config, sharingAffinityTestConfig(sharingAffinityTestDriver, `{"malformed":`, "other"))
		}, want: fwk.Success},
		{name: "scope/class-config-excluded", mutate: func(c *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, class *resourceapi.DeviceClass) {
			class.Spec.Config = []resourceapi.DeviceClassConfiguration{{DeviceConfiguration: c.Spec.Devices.Config[0].DeviceConfiguration}}
			c.Spec.Devices.Config = nil
		}, want: fwk.Unschedulable},
		{name: "scope/class-conflict-excluded", mutate: func(_ *resourceapi.ResourceClaim, _ *resourceapi.ResourceSlice, class *resourceapi.DeviceClass) {
			class.Spec.Config = []resourceapi.DeviceClassConfiguration{{DeviceConfiguration: sharingAffinityTestConfig(sharingAffinityTestDriver,
				`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"other-secret"}`).DeviceConfiguration}}
		}, want: fwk.Success},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setSharingAffinityAccountingGates(t, true, true)
			ctx, logs := sharingAffinityTestLogContext(t, 10)
			claim, slice, class := filterAffinityFixture()
			if tc.expression != "" {
				slice.Spec.SharingAffinity[0].Expression = tc.expression
			}
			if tc.raw != "" {
				claim.Spec.Devices.Config[0] = sharingAffinityTestConfig(sharingAffinityTestDriver, tc.raw)
			}
			if tc.mutate != nil {
				tc.mutate(claim, slice, class)
			}
			snapshot := structured.AllocatedState{}
			if tc.lock != nil {
				snapshot.DeviceSharingAffinity.DeviceStates = structured.AffinityStateCollection{
					structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "device"): *tc.lock,
				}
			}
			allocator, err := structured.NewAllocator(ctx, structured.Features{SharingAffinity: true, ConsumableCapacity: true},
				snapshot, filterAffinityClasses{class}, []*resourceapi.ResourceSlice{slice}, cel.NewCache(10, cel.Features{}))
			if err != nil {
				t.Fatalf("NewAllocator: %v", err)
			}
			cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
			node := framework.NewNodeInfo()
			node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
			plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
			status := plugin.Filter(ctx, cycle, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default"}}, node)
			if got := status.Code(); got != tc.want {
				t.Errorf("Filter code = %v (%v), want %v", got, status, tc.want)
			}
			if tc.message != "" && !strings.Contains(status.Message(), tc.message) {
				t.Errorf("Filter message = %q, want substring %q", status.Message(), tc.message)
			}
			if tc.cause != nil && !errors.Is(status.AsError(), tc.cause) {
				t.Errorf("Filter error = %v, want wrapped %v", status.AsError(), tc.cause)
			}
			if tc.want != fwk.Success && len(state.nodeAllocations) != 0 {
				t.Errorf("failed Filter published a node allocation: %#v", state.nodeAllocations)
			}
			if tc.want == fwk.Success {
				allocation, found := state.nodeAllocations["node"]
				if !found || len(allocation.allocationResults) != 1 || len(allocation.allocationResults[0].Devices.Results) != 1 {
					t.Errorf("successful Filter did not cache exactly one device: %#v", allocation)
				}
			}
			for _, secret := range []string{"opaque-secret", "other-secret"} {
				if strings.Contains(status.Message(), secret) || strings.Contains(logs.String(), secret) {
					t.Errorf("Filter status or logs disclose opaque value %q", secret)
				}
			}
		})
	}

	t.Run("fallback-allocator-without-affinity", func(t *testing.T) {
		for _, enabled := range []bool{false, true} {
			t.Run(fmt.Sprintf("enabled=%t", enabled), func(t *testing.T) {
				setSharingAffinityAccountingGates(t, enabled, true)
				_, ctx := ktesting.NewTestContext(t)
				claim, _, _ := filterAffinityFixture()
				cycle, _ := filterAffinityState([]*resourceapi.ResourceClaim{claim}, filterAffinityLegacyAllocator{
					results: []resourceapi.AllocationResult{{}},
				})

				t.Run("affinity-allocator-interface", func(t *testing.T) {
					setSharingAffinityAccountingGates(t, true, true)
					_, ctx := ktesting.NewTestContext(t)
					claim, _, _ := filterAffinityFixture()
					result := resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
						sharingAffinityTestResult("req", "device", "share"),
					}}}
					prov := structured.DeviceAffinity{
						ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "device"),
						Pool:   sharingaffinity.PoolKey{Driver: sharingAffinityTestDriver, Name: sharingAffinityTestPool, Generation: 1},
						Digest: structured.EmptyExtractorDigest,
					}
					allocator := &filterAffinityAllocator{results: []resourceapi.AllocationResult{result}, affinity: []structured.ClaimAffinity{{Devices: []structured.DeviceAffinity{prov}}}}
					cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
					node := framework.NewNodeInfo()
					node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
					plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
					status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
					if !status.IsSuccess() || !allocator.called {
						t.Fatalf("Filter with optional AffinityAllocator = %v, called=%t; want success via AllocateWithAffinity", status, allocator.called)
					}
					if len(state.nodeAllocations["node"].affinityByClaim) != 1 {
						t.Fatalf("affinityByClaim = %#v, want one plan", state.nodeAllocations["node"].affinityByClaim)
					}
				})

				t.Run("affinity-length-mismatch", func(t *testing.T) {
					baseResult := resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
						sharingAffinityTestResult("req", "device", "share"),
					}}}
					baseAffinity := structured.ClaimAffinity{Devices: []structured.DeviceAffinity{{
						ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, "device"),
						Pool:   sharingaffinity.PoolKey{Driver: sharingAffinityTestDriver, Name: sharingAffinityTestPool, Generation: 1},
						Digest: structured.EmptyExtractorDigest,
					}}}
					for _, tc := range []struct {
						name     string
						results  []resourceapi.AllocationResult
						affinity []structured.ClaimAffinity
					}{
						{name: "claim-length-short", results: []resourceapi.AllocationResult{baseResult}, affinity: nil},
						{name: "device-length-short", results: []resourceapi.AllocationResult{baseResult}, affinity: []structured.ClaimAffinity{{}}},
						{name: "negative-index", results: []resourceapi.AllocationResult{baseResult}, affinity: []structured.ClaimAffinity{{Devices: []structured.DeviceAffinity{{ResultIndex: -1}}}}},
						{name: "out-of-range-index", results: []resourceapi.AllocationResult{baseResult}, affinity: []structured.ClaimAffinity{{Devices: []structured.DeviceAffinity{{ResultIndex: 1}}}}},
						{name: "duplicate-index", results: []resourceapi.AllocationResult{{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
							sharingAffinityTestResult("req", "device", "share"), sharingAffinityTestResult("req", "device", "share-2"),
						}}}}, affinity: []structured.ClaimAffinity{{Devices: []structured.DeviceAffinity{baseAffinity.Devices[0], baseAffinity.Devices[0]}}}},
					} {
						t.Run(tc.name, func(t *testing.T) {
							setSharingAffinityAccountingGates(t, true, true)
							_, ctx := ktesting.NewTestContext(t)
							claim, _, _ := filterAffinityFixture()
							cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, &filterAffinityAllocator{results: tc.results, affinity: tc.affinity})
							node := framework.NewNodeInfo()
							node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
							plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
							status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
							if status.Code() != fwk.Error || !strings.Contains(status.Message(), "affinity results do not match allocation results") {
								t.Fatalf("Filter = %v, want affinity mismatch Error", status)
							}
							if len(state.nodeAllocations) != 0 {
								t.Fatalf("mismatched affinity published node allocations: %#v", state.nodeAllocations)
							}
						})

						t.Run("podgroup/mixed-new-and-pending", func(t *testing.T) {
							setSharingAffinityAccountingGates(t, true, true)
							_, ctx := ktesting.NewTestContext(t)
							a, _, _ := filterAffinityFixture()
							b := a.DeepCopy()
							c := a.DeepCopy()
							a.Name, a.UID = "a", "a"
							b.Name, b.UID = "b", "b"
							c.Name, c.UID = "c", "c"
							resultFor := func(device, share string) resourceapi.AllocationResult {
								return resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
									sharingAffinityTestResult("req", device, types.UID(share)),
								}}}
							}
							provFor := func(device string) structured.ClaimAffinity {
								return structured.ClaimAffinity{Devices: []structured.DeviceAffinity{{
									ResultIndex: 0, Request: "req", Device: structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, device),
									Pool:   sharingaffinity.PoolKey{Driver: sharingAffinityTestDriver, Name: sharingAffinityTestPool, Generation: 1},
									Digest: structured.EmptyExtractorDigest,
								}}}
							}
							allocator := &filterAffinityAllocator{
								results:  []resourceapi.AllocationResult{resultFor("new-a", "share-a"), resultFor("new-c", "share-c")},
								affinity: []structured.ClaimAffinity{provFor("new-a"), provFor("new-c")},
							}
							cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{a, b, c}, allocator)
							state.informationsForClaim[1].allocation = ptr.To(resultFor("pending-b", "share-b"))
							state.informationsForClaim[1].pendingAffinity = &pendingAffinity{allocation: state.informationsForClaim[1].allocation, provenance: provFor("pending-b").Devices, claimUID: b.UID}
							node := framework.NewNodeInfo()
							node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
							plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
							status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
							if !status.IsSuccess() {
								t.Fatalf("Filter = %v, want success", status)
							}
							plans := state.nodeAllocations["node"].affinityByClaim
							if got, want := sets.List(sets.KeySet(plans)), []int{0, 1, 2}; !reflect.DeepEqual(got, want) {
								t.Fatalf("plan keys = %v, want store indices %v", got, want)
							}
							for index, wantDevice := range map[int]string{0: "new-a", 1: "pending-b", 2: "new-c"} {
								plan := plans[index]
								if len(plan.devices) != 1 || plan.devices[0].Device != structured.MakeDeviceID(sharingAffinityTestDriver, sharingAffinityTestPool, wantDevice) {
									t.Errorf("plan[%d] provenance = %#v, want device %s", index, plan.devices, wantDevice)
								}
								if len(plan.allocation.Devices.Results) != 1 || plan.allocation.Devices.Results[0].Device != wantDevice {
									t.Errorf("plan[%d] allocation = %#v, want device %s", index, plan.allocation, wantDevice)
								}
							}
							if plans[1].source != planPending || plans[0].source != planNew || plans[2].source != planNew {
								t.Fatalf("plan sources = %#v, want new/pending/new", plans)
							}
						})
					}
				})
				node := framework.NewNodeInfo()
				node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
				plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: enabled, EnableDRAConsumableCapacity: true}}
				status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
				want := fwk.Success
				if enabled {
					want = fwk.Error
				}
				if status.Code() != want {
					t.Errorf("legacy allocator Filter = %v, want %v", status, want)
				}
			})
		}
	})

	t.Run("status/deadline", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		claim, _, _ := filterAffinityFixture()
		cycle, _ := filterAffinityState([]*resourceapi.ResourceClaim{claim},
			filterAffinityLegacyAllocator{err: fmt.Errorf("allocation interrupted: %w", context.DeadlineExceeded)})
		node := framework.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
		plugin := &DynamicResources{}
		status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
		if status.Code() != fwk.Error || !strings.Contains(status.Message(), "timed out") {
			t.Errorf("deadline Filter = %v, want Error with timeout diagnostic", status)
		}
	})

	t.Run("missing-pending-reader", func(t *testing.T) {
		setSharingAffinityAccountingGates(t, true, true)
		_, ctx := ktesting.NewTestContext(t)
		claim, slice, class := filterAffinityFixture()
		allocator, err := structured.NewAllocator(ctx, structured.Features{SharingAffinity: true, ConsumableCapacity: true},
			structured.AllocatedState{}, filterAffinityClasses{class}, []*resourceapi.ResourceSlice{slice}, cel.NewCache(10, cel.Features{}))
		if err != nil {
			t.Fatalf("NewAllocator: %v", err)
		}
		cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
		state.informationsForClaim[0].allocation = &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{
				sharingAffinityTestResult("req", "device", "pending-share"),
			}},
		}
		tracker := newSharingAffinityAccountingTracker(t, false)
		plugin := &DynamicResources{
			fts:        feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true},
			draManager: filterAffinityLegacyManager{tracker: filterAffinityLegacyTracker{ResourceClaimTracker: tracker}},
		}
		node := framework.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
		status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
		if status.Code() != fwk.Error {
			t.Errorf("pending claim without affinity reader Filter = %v, want Error", status)
		}
		if !strings.Contains(status.Message(), "affinity") {
			t.Errorf("missing-reader error lacks affinity diagnostic: %q", status.Message())
		}
	})

	t.Run("status/runtime-fatal/cost-exhausted", func(t *testing.T) {
		setSharingAffinityAccountingGates(t, true, true)
		ctx, logs := sharingAffinityTestLogContext(t, 5)
		claim, slice, class := filterAffinityFixture()
		entry := sharingAffinityTestExtractor("subnet", `parameters.s.matches("a{1,3}") ? "x" : "x"`)
		if err := sharingaffinity.ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
			t.Fatalf("runtime-cost fixture must pass pool validation: %v", err)
		}
		slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
		pool := sharingaffinity.PoolExtractors{
			Key:        sharingaffinity.PoolKey{Driver: slice.Spec.Driver, Name: slice.Spec.Pool.Name, Generation: slice.Spec.Pool.Generation},
			Extractors: slice.Spec.SharingAffinity,
		}
		config := sharingAffinityTestConfig(sharingAffinityTestDriver, `{"s":"`+strings.Repeat("a", 300000)+`","private":"opaque-secret"}`)
		extractor := sharingaffinity.NewExtractor(10)
		claim.Spec.Devices.Config = nil
		// Calibrate through the public extractor so CEL cost-model changes don't
		// turn a runtime-cost test into a static-validation test.
		exhausted := false
		for range 32 {
			claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, config)
			_, err := extractor.Extract(ctx, pool, sharingaffinity.Request{Name: "req"}, claim.Spec.Devices.Config)
			if errors.Is(err, sharingaffinity.ErrCostExceeded) {
				exhausted = true
				break
			}
			if err != nil {
				t.Fatalf("calibrating runtime-cost fixture: %v", err)
			}
		}
		if !exhausted || len(claim.Spec.Devices.Config) < 2 {
			t.Fatal("runtime-cost fixture must exhaust a shared budget, but not a single-config budget")
		}
		allocator, err := structured.NewAllocator(ctx, structured.Features{SharingAffinity: true, ConsumableCapacity: true},
			structured.AllocatedState{}, filterAffinityClasses{class}, []*resourceapi.ResourceSlice{slice}, cel.NewCache(10, cel.Features{}))
		if err != nil {
			t.Fatalf("NewAllocator: %v", err)
		}
		cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
		node := framework.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}})
		plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
		status := plugin.Filter(ctx, cycle, &v1.Pod{}, node)
		if status.Code() != fwk.Error || !errors.Is(status.AsError(), sharingaffinity.ErrCostExceeded) {
			t.Errorf("over-budget Filter = %v, want Error wrapping ErrCostExceeded", status)
		}
		if len(state.nodeAllocations) != 0 {
			t.Error("over-budget Filter published an allocation")
		}
		if strings.Contains(status.Message(), "opaque-secret") || strings.Contains(logs.String(), "opaque-secret") {
			t.Error("over-budget Filter disclosed opaque parameters")
		}
	})

	t.Run("node-results/isolated", func(t *testing.T) {
		setSharingAffinityAccountingGates(t, true, true)
		_, ctx := ktesting.NewTestContext(t)
		claim, slice, class := filterAffinityFixture()
		before := claim.DeepCopy()
		allocator, err := structured.NewAllocator(ctx, structured.Features{SharingAffinity: true, ConsumableCapacity: true},
			structured.AllocatedState{}, filterAffinityClasses{class}, []*resourceapi.ResourceSlice{slice}, cel.NewCache(10, cel.Features{}))
		if err != nil {
			t.Fatalf("NewAllocator: %v", err)
		}
		cycle, state := filterAffinityState([]*resourceapi.ResourceClaim{claim}, allocator)
		plugin := &DynamicResources{fts: feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true}}
		var wg sync.WaitGroup
		for _, name := range []string{"one", "two"} {
			wg.Go(func() {
				node := framework.NewNodeInfo()
				node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
				if status := plugin.Filter(ctx, cycle, &v1.Pod{}, node); !status.IsSuccess() {
					t.Errorf("Filter(%s) = %v", name, status)
				}
			})
		}
		wg.Wait()
		one, two := state.nodeAllocations["one"], state.nodeAllocations["two"]
		if len(one.allocationResults) != 1 || len(two.allocationResults) != 1 {
			t.Fatalf("node allocation counts = %d, %d; want 1, 1", len(one.allocationResults), len(two.allocationResults))
		}
		original := two.allocationResults[0].DeepCopy()
		one.allocationResults[0].Devices.Results[0].Device = "mutated"
		one.allocationResults[0].Devices.Results[0].ShareID = ptr.To(types.UID("mutated"))
		if !reflect.DeepEqual(original, &two.allocationResults[0]) {
			t.Error("mutating one node's results changed the other node's results")
		}
		onePlan, oneOK := one.affinityByClaim[0]
		twoPlan, twoOK := two.affinityByClaim[0]
		if !oneOK || !twoOK || len(onePlan.devices) != 1 || len(twoPlan.devices) != 1 || onePlan.devices[0].Values == nil || twoPlan.devices[0].Values == nil {
			t.Fatalf("cached affinity plans = %#v, %#v; want one constrained device per node", one.affinityByClaim, two.affinityByClaim)
		}
		originalPlan := twoPlan.devices[0]
		originalPlan.Values = maps.Clone(originalPlan.Values)
		originalAllocation := twoPlan.allocation.DeepCopy()
		onePlan.devices[0].Values["subnet"] = "mutated"
		onePlan.devices[0].Request = "mutated"
		onePlan.allocation.Devices.Results[0].Device = "mutated"
		if !reflect.DeepEqual(originalPlan, two.affinityByClaim[0].devices[0]) || !reflect.DeepEqual(originalAllocation, two.affinityByClaim[0].allocation) {
			t.Error("mutating one node's cached affinity plan changed the other node's plan")
		}
		if onePlan.allocation == &one.allocationResults[0] {
			t.Error("cached affinity allocation aliases the node allocation results")
		}
		if !reflect.DeepEqual(before, claim) {
			t.Error("concurrent Filter mutated its input claim")
		}
	})
}
