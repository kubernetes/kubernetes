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

package dra

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	draapi "k8s.io/dynamic-resource-allocation/api"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

// Keep these tests serial: the apiserver and scheduler share process-global gates.
func TestSharingAffinityRestart(t *testing.T) {
	env := newSAEnv(t)
	env.run("retained-owners", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
		stop := env.startScheduler()
		firstManager := env.gate.handle.SharedDRAManager()
		a := saScheduleOwner(s, "a", subnet("subnet-a"), "nic")
		b := saScheduleOwner(s, "b", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))

		stop()
		saRetained(s, a, b)
		stop = env.startScheduler()
		defer stop()
		if firstManager == env.gate.handle.SharedDRAManager() {
			s.tCtx.Fatal("restart reused the first scheduler's DRA manager")
		}
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, a, b)
		c := saScheduleOwner(s, "c", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "3Gi", "subnet", "subnet-a"))
		d := saRestartClaim(s, "d", subnet("subnet-b"), "nic")
		pd := s.pod("pd", "", d)
		s.notScheduled(pd, v1.PodReasonUnschedulable, nil)
		s.staysPending(pd, d)
		s.release(a.pod, a.claim)
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, b, c)
		s.staysPending(pd, d)
		s.release(b.pod, b.claim)
		s.release(c.pod, c.claim)
		s.scheduled(pd)
		saOnDevice(s, d, "pool/nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", "subnet-b"))
	})
}

func TestSharingAffinityReconstructionConflict(t *testing.T) {
	env := newSAEnv(t)
	for _, first := range []string{"subnet-a"} {
		env.run("conflicting-seeded-owners/"+first+"-first", func(s *saScope) {
			s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
			second := "subnet-b"
			if first == second {
				second = "subnet-a"
			}
			// Fixture-seeded historical allocations: a correct scheduler cannot
			// create conflicting owners in the first place.
			a := saSeedOwner(s, "a", "pool", "nic", subnet(first))
			b := saSeedOwner(s, "b", "pool", "nic", subnet(second))
			stop := env.startScheduler()
			defer stop()
			saWaitDevices(env, s, saUnknown("pool", "nic", "2Gi"))
			saRetained(s, a, b)
			c := saRestartClaim(s, "c", subnet(second), "nic")
			pc := s.pod("pc", "", c)
			s.notScheduled(pc, v1.PodReasonUnschedulable, nil)
			s.staysPending(pc, c)
			s.release(a.pod, a.claim)
			saWaitDevices(env, s, saUnknown("pool", "nic", "1Gi"))
			saRetained(s, b)
			s.staysPending(pc, c)
			s.release(b.pod, b.claim)
			s.scheduled(pc)
			saOnDevice(s, c, "pool/nic")
			saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", second))
		})
	}
}

func TestSharingAffinityLegacyMigration(t *testing.T) {
	env := newSAEnv(t)
	env.run("mixed-owners-and-unaffected-devices", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev1"), saRestartDevice("dev2"))
		s.slice("other", "other", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("other"))
		// Fixture-seeded legacy/historical allocations, all with bound consumers.
		l1 := saSeedOwner(s, "l1", "pool", "dev1", "")
		l2 := saSeedOwner(s, "l2", "pool", "dev1", "")
		r1 := saSeedOwner(s, "r1", "pool", "dev1", subnet("subnet-a"))
		r2 := saSeedOwner(s, "r2", "pool", "dev2", subnet("subnet-a"))
		stop := env.startScheduler()
		defer stop()
		saWaitDevices(env, s, saUnknown("pool", "dev1", "3Gi"),
			saLocked("pool", "dev2", "1Gi", "subnet", "subnet-a"), saClean("other", "other"))
		saRetained(s, l1, l2, r1, r2)
		healthy := saScheduleOwner(s, "healthy", subnet("subnet-a"), "dev2")
		other := saScheduleOwner(s, "other", subnet("subnet-b"), "other")
		saOnDevice(s, healthy.claim, "pool/dev2")
		saOnDevice(s, other.claim, "other/other")
		saWaitDevices(env, s, saUnknown("pool", "dev1", "3Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		next := saRestartClaim(s, "next", subnet("subnet-a"), "dev1")
		pnext := s.pod("pnext", "", next)
		s.notScheduled(pnext, v1.PodReasonUnschedulable, nil)
		s.staysPending(pnext, next)
		s.release(l1.pod, l1.claim)
		saWaitDevices(env, s, saUnknown("pool", "dev1", "2Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, l2, r1, r2, healthy, other)
		s.staysPending(pnext, next)
		s.release(l2.pod, l2.claim)
		saWaitDevices(env, s, saUnknown("pool", "dev1", "1Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, r1, r2, healthy, other)
		// Even removing every legacy owner cannot clear a genuine fence
		// while the reconstructable owner remains on the device.
		s.staysPending(pnext, next)
		s.release(r1.pod, r1.claim)
		s.scheduled(pnext)
		saOnDevice(s, next, "pool/dev1")
		saWaitDevices(env, s, saLocked("pool", "dev1", "1Gi", "subnet", "subnet-a"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, r2, healthy, other)
	})
}

func TestSharingAffinityMetadataReconciliation(t *testing.T) {
	env := newSAEnv(t)
	stop := env.startScheduler()
	defer stop()
	for _, tc := range []struct {
		name       string
		key        string
		expression string
		newParams  string
		equal      bool
	}{
		{
			name: "equal-map-dual-path", key: "subnet",
			expression: `has(parameters.subnet) ? parameters.subnet : parameters.subnetID`,
			newParams:  nicParams(`"subnet":"subnet-a"`), equal: true,
		},
		{
			name: "renamed-key", key: "vpc-subnet", expression: saSubnetExpr,
			newParams: subnet("subnet-a"),
		},
		{
			name: "changed-value", key: "subnet", expression: `parameters.?vlan.orValue("")`,
			newParams: nicParams(`"subnetID":"subnet-a","vlan":"subnet-b"`),
		},
	} {
		env.run(tc.name, func(s *saScope) {
			slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
			oldParams := nicParams(`"subnetID":"subnet-a","vlan":"subnet-b"`)
			a := saScheduleOwner(s, "a", oldParams, "nic")
			b := saScheduleOwner(s, "b", oldParams, "nic")
			saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
			extractors := subnetExtractor(tc.expression)
			extractors[0].Name = tc.key
			want := saUnknown("pool", "nic", "2Gi")
			if tc.equal {
				want = saLocked("pool", "nic", "2Gi", "subnet", "subnet-a")
			}
			saRepublishAndProbe(env, s, slices, 2, extractors, tc.newParams, want)
			saRetained(s, a, b)
			c := saRestartClaim(s, "c", tc.newParams, "nic")
			pc := s.pod("pc", "", c)
			if tc.equal {
				s.scheduled(pc)
				saOnDevice(s, c, "pool/nic")
				saWaitDevices(env, s, saLocked("pool", "nic", "3Gi", "subnet", "subnet-a"))
				saRetained(s, a, b)
				return
			}
			s.notScheduled(pc, v1.PodReasonUnschedulable, nil)
			s.staysPending(pc, c)
			// Returning to the old map must not erase an established fence.
			saRepublishAndProbe(env, s, slices, 3, subnetExtractor(saSubnetExpr), oldParams, want)
			saRetained(s, a, b)
			s.staysPending(pc, c)
			saRepublishAndProbe(env, s, slices, 4, extractors, tc.newParams, want)
			s.release(a.pod, a.claim)
			saWaitDevices(env, s, saUnknown("pool", "nic", "1Gi"))
			saRetained(s, b)
			s.staysPending(pc, c)
			s.release(b.pod, b.claim)
			s.scheduled(pc)
			saOnDevice(s, c, "pool/nic")
			value := "subnet-a"
			if tc.name == "changed-value" {
				value = "subnet-b"
			}
			saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", tc.key, value))
		})
	}
}

func TestSharingAffinityGateReenable(t *testing.T) {
	env := newSAEnv(t)
	env.run("retained-reconstructable-and-legacy-owners", func(s *saScope) {
		slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev-r"), saRestartDevice("dev-l"))
		s.slice("fallback", "fallback", 1, 1, nil, saRestartDevice("fallback"))
		// Fixture-seeded legacy/historical allocations. A gate-off scheduler
		// cannot establish these owners in an affinity-declaring pool.
		r := saSeedOwner(s, "r", "pool", "dev-r", subnet("subnet-a"))
		l := saSeedOwner(s, "l", "pool", "dev-l", "")
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, false)
		stored, err := s.tCtx.Client().ResourceV1().ResourceSlices().Get(s.tCtx, slices[1].Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "read persisted affinity metadata with scheduler gate off")
		if !reflect.DeepEqual(stored.Spec.SharingAffinity, subnetExtractor(saSubnetExpr)) {
			s.tCtx.Fatalf("affinity metadata was not retained: %#v", stored.Spec.SharingAffinity)
		}
		stop := env.startScheduler()
		firstManager := env.gate.handle.SharedDRAManager()
		saWaitDevices(env, s, saWantDevice{pool: "pool", device: "dev-r", capacity: "1Gi"},
			saWantDevice{pool: "pool", device: "dev-l", capacity: "1Gi"})
		state, err := firstManager.ResourceClaims().GatherAllocatedState()
		s.tCtx.ExpectNoError(err, "gather gate-disabled accounting")
		if state.DeviceSharingAffinity.DeviceStates != nil {
			s.tCtx.Fatalf("gate-disabled scheduler populated affinity: %#v", state.DeviceSharingAffinity.DeviceStates)
		}
		fallback := saScheduleOwner(s, "fallback", subnet("subnet-b"), "")
		saOnDevice(s, fallback.claim, "fallback/fallback")
		blocked := saRestartClaim(s, "blocked", subnet("subnet-a"), "dev-r")
		pblocked := s.pod("pblocked", "", blocked)
		s.notScheduled(pblocked, v1.PodReasonUnschedulable, nil)
		s.staysPending(pblocked, blocked)
		saWaitDevices(env, s, saWantDevice{pool: "pool", device: "dev-r", capacity: "1Gi"},
			saWantDevice{pool: "pool", device: "dev-l", capacity: "1Gi"},
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback)
		// Restart without a pending placement, so the first post-restart
		// snapshot proves reconstruction of exactly the retained owners.
		s.release(pblocked, blocked)
		stop()
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, true)
		stop = env.startScheduler()
		defer stop()
		if firstManager == env.gate.handle.SharedDRAManager() {
			s.tCtx.Fatal("gate reenable reused the gate-disabled DRA manager")
		}
		saWaitDevices(env, s, saLocked("pool", "dev-r", "1Gi", "subnet", "subnet-a"),
			saUnknown("pool", "dev-l", "1Gi"),
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback)
		saRepublishAndProbe(env, s, slices, 2, subnetExtractor(saSubnetExpr), subnet("subnet-a"),
			saLocked("pool", "dev-r", "1Gi", "subnet", "subnet-a"), saUnknown("pool", "dev-l", "1Gi"))
		compatible := saScheduleOwner(s, "compatible", subnet("subnet-a"), "dev-r")
		saOnDevice(s, compatible.claim, "pool/dev-r")
		incompatible := saRestartClaim(s, "incompatible", subnet("subnet-b"), "dev-r")
		pincompatible := s.pod("pincompatible", "", incompatible)
		s.notScheduled(pincompatible, v1.PodReasonUnschedulable, nil)
		s.staysPending(pincompatible, incompatible)
		legacy := saRestartClaim(s, "legacy", subnet("subnet-a"), "dev-l")
		plegacy := s.pod("plegacy", "", legacy)
		s.notScheduled(plegacy, v1.PodReasonUnschedulable, nil)
		s.staysPending(plegacy, legacy)
		saWaitDevices(env, s, saLocked("pool", "dev-r", "2Gi", "subnet", "subnet-a"),
			saUnknown("pool", "dev-l", "1Gi"),
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback, compatible)
	})
}

func saRestartDevice(name string) saDevice {
	d := shared(name, "16Gi")
	d.attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"slot": {StringValue: ptr.To(name)}}
	return d
}

func saNonMarkerSelector(driver string) string {
	return fmt.Sprintf(`!("marker" in device.attributes[%q])`, driver)
}

func saMarkerSelector(driver, marker string) string {
	return fmt.Sprintf(`"marker" in device.attributes[%q] && device.attributes[%q].marker == %q`, driver, driver, marker)
}

func saRestartClaim(s *saScope, name, params, slot string) *resourceapi.ResourceClaim {
	s.tCtx.Helper()
	expression := saNonMarkerSelector(s.driver)
	if slot != "" {
		expression += fmt.Sprintf(` && "slot" in device.attributes[%q] && device.attributes[%q].slot == %q`, s.driver, s.driver, slot)
	}
	return saSelectedClaim(s, name, params, expression)
}

func saSelectedClaim(s *saScope, name, params, expression string) *resourceapi.ResourceClaim {
	s.tCtx.Helper()
	var configs []saConfig
	if params != "" {
		configs = []saConfig{{requests: []string{"nic"}, params: params}}
	}
	return s.claim(name, []saRequest{{
		name: "nic", memory: "1Gi",
		selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{Expression: expression}}},
	}}, configs...)
}

type saOwner struct {
	pod   *v1.Pod
	claim *resourceapi.ResourceClaim
}

func saScheduleOwner(s *saScope, name, params, slot string) saOwner {
	s.tCtx.Helper()
	claim := saRestartClaim(s, name, params, slot)
	pod := s.scheduled(s.pod("p"+name, "", claim))
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, claim.Name, metav1.GetOptions{})
	s.tCtx.ExpectNoError(err, "capture scheduled claim")
	if claim.Status.Allocation == nil {
		s.tCtx.Fatalf("scheduled pod %s has no claim allocation", pod.Name)
	}
	if slot != "" {
		results := claim.Status.Allocation.Devices.Results
		if len(results) != 1 || results[0].Device != slot || results[0].ShareID == nil {
			s.tCtx.Fatalf("claim %s: expected shared allocation on %s, got %+v", claim.Name, slot, results)
		}
	}
	return saOwner{pod: pod, claim: claim}
}

// saSeedOwner is only for fixture-seeded legacy/historical allocations which
// cannot be produced by a correctly enforcing scheduler.
func saSeedOwner(s *saScope, name, pool, device, params string) saOwner {
	s.tCtx.Helper()
	claim := saRestartClaim(s, name, params, device)
	pod := st.MakePod().Name("p" + name).Container("c").Obj()
	pod.Spec.NodeName = saNode
	pod = createPod(s.tCtx, s.ns, "", pod, claim)
	claim.Status.Allocation = &resourceapi.AllocationResult{
		NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchFields: []v1.NodeSelectorRequirement{{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{saNode}}},
		}}},
		Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
			Request: "nic", Driver: s.driver, Pool: pool, Device: device,
			ShareID:          ptr.To(uuid.NewUUID()),
			ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{saMemory: resource.MustParse("1Gi")},
		}}},
	}
	for _, config := range claim.Spec.Devices.Config {
		claim.Status.Allocation.Devices.Config = append(claim.Status.Allocation.Devices.Config, resourceapi.DeviceAllocationConfiguration{
			Source: resourceapi.AllocationConfigSourceClaim, Requests: config.Requests, DeviceConfiguration: config.DeviceConfiguration,
		})
	}
	claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{{Resource: "pods", Name: pod.Name, UID: pod.UID}}
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).UpdateStatus(s.tCtx, claim, metav1.UpdateOptions{})
	s.tCtx.ExpectNoError(err, "fixture-seed historical allocation and bound consumer")
	return saOwner{pod: pod, claim: claim}
}

func saRetained(s *saScope, owners ...saOwner) {
	s.tCtx.Helper()
	for _, owner := range owners {
		pod, err := s.tCtx.Client().CoreV1().Pods(s.ns).Get(s.tCtx, owner.pod.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get retained bound pod")
		if pod.UID != owner.pod.UID || pod.Spec.NodeName != saNode || pod.DeletionTimestamp != nil {
			s.tCtx.Fatalf("retained pod %s was replaced, unbound, or deleted: %+v", pod.Name, pod)
		}
		claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, owner.claim.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get retained allocated claim")
		if claim.UID != owner.claim.UID || claim.ResourceVersion != owner.claim.ResourceVersion ||
			claim.DeletionTimestamp != nil || !reflect.DeepEqual(claim.Status.Allocation, owner.claim.Status.Allocation) ||
			!reflect.DeepEqual(claim.Status.ReservedFor, owner.claim.Status.ReservedFor) {
			s.tCtx.Fatalf("retained claim %s changed: before=%+v, after=%+v", claim.Name, owner.claim, claim)
		}
	}
}

func saOnDevice(s *saScope, claim *resourceapi.ResourceClaim, want string) {
	s.tCtx.Helper()
	if got := s.device(claim); got != want {
		s.tCtx.Fatalf("claim %s allocated on %s, want %s", claim.Name, got, want)
	}
}

type saWantDevice struct {
	pool, device, capacity string
	affinity               structured.DeviceAffinityState
}

func saLocked(pool, device, capacity, key, value string) saWantDevice {
	return saWantDevice{pool: pool, device: device, capacity: capacity,
		affinity: structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{key: value}}}
}

func saUnknown(pool, device, capacity string) saWantDevice {
	return saWantDevice{pool: pool, device: device, capacity: capacity,
		affinity: structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}}
}

func saClean(pool, device string) saWantDevice {
	return saWantDevice{pool: pool, device: device}
}

// saCheckDevices compares all requested devices in one immutable snapshot.
func saCheckDevices(s *saScope, state *schedulerapi.AllocatedState, wants ...saWantDevice) error {
	if state == nil {
		return fmt.Errorf("allocated state is nil")
	}
	for _, want := range wants {
		id := structured.MakeDeviceID(s.driver, want.pool, want.device)
		if state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: s.driver, Pool: want.pool}) {
			return fmt.Errorf("pool %s remains stale", want.pool)
		}
		got, found := state.DeviceSharingAffinity.DeviceStates[id]
		if !reflect.DeepEqual(got, want.affinity) || (want.affinity.Status == structured.AffinityStatusClean && found) {
			return fmt.Errorf("device %s: affinity=%+v (present=%t), want %+v", id, got, found, want.affinity)
		}
		if state.AllocatedDevices.Has(id) {
			return fmt.Errorf("device %s unexpectedly has a dedicated allocation", id)
		}
		capacity := state.AggregatedCapacity[id][draapi.MakeFullyQualifiedName(saMemory, s.driver)]
		if want.capacity == "" {
			if _, found := state.AggregatedCapacity[id]; found || state.AllocatedSharedDeviceIDs.Has(id) {
				return fmt.Errorf("device %s has not drained: capacity=%v shared=%t", id, capacity, state.AllocatedSharedDeviceIDs.Has(id))
			}
		} else if capacity == nil || capacity.Cmp(resource.MustParse(want.capacity)) != 0 || !state.AllocatedSharedDeviceIDs.Has(id) {
			return fmt.Errorf("device %s: capacity=%v shared=%t, want %s shared", id, capacity, state.AllocatedSharedDeviceIDs.Has(id), want.capacity)
		}
	}
	return nil
}

func saWaitDevices(env *saEnv, s *saScope, wants ...saWantDevice) {
	s.tCtx.Helper()
	s.tCtx.Eventually(func() error {
		state, err := env.gate.handle.SharedDRAManager().ResourceClaims().GatherAllocatedState()
		if err != nil {
			return err
		}
		return saCheckDevices(s, state, wants...)
	}).WithTimeout(schedulingTimeout).Should(gomega.Succeed(), "wait for reconstructed device state")
}

// Binding a generation-specific marker proves both metadata publication and
// pool reconciliation. A bare !StalePools check can still see the old generation.
func saRepublishAndProbe(env *saEnv, s *saScope, slices saSlices, generation int64,
	extractors []resourceapi.SharingAffinityExtractor, params string, wants ...saWantDevice) {
	s.tCtx.Helper()
	marker := fmt.Sprintf("m%d", generation)
	device := fmt.Sprintf("dev-marker-%d", generation)
	s.updateSlice(slices, func(slice *resourceapi.ResourceSlice) {
		slice.Spec.Pool.Generation = generation
		if len(slice.Spec.SharingAffinity) != 0 {
			slice.Spec.SharingAffinity = extractors
			return
		}
		// Ordinary devices deliberately precede the marker: a missing marker
		// attribute must be false, not a CEL error aborting the allocation.
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
			Name: device, AllowMultipleAllocations: ptr.To(true),
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(s.driver + "/marker"): {StringValue: ptr.To(marker)},
			},
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{saMemory: {Value: resource.MustParse("16Gi")}},
		})
	})
	claim := saSelectedClaim(s, "probe-"+marker, params, saMarkerSelector(s.driver, marker))
	pod := s.scheduled(s.pod("probe-"+marker, "", claim))
	saOnDevice(s, claim, slices[0].Spec.Pool.Name+"/"+device)
	state, err := env.gate.handle.SharedDRAManager().ResourceClaims().GatherAllocatedState()
	s.tCtx.ExpectNoError(err, "gather state after generation-specific marker bound")
	s.tCtx.ExpectNoError(saCheckDevices(s, state, wants...), "reconciled owner state after marker bound")
	s.release(pod, claim)
	saWaitDevices(env, s, append(wants, saClean(slices[0].Spec.Pool.Name, device))...)
}

func TestSharingAffinityMarkerSelectorRejectsOrdinaryDevice(t *testing.T) {
	const driver = "sharing-affinity.example.com"
	for _, selector := range []struct {
		name       string
		expression string
		want       []bool
	}{
		{name: "probe", expression: saMarkerSelector(driver, "m2"), want: []bool{false, true, false}},
		{name: "workload", expression: saNonMarkerSelector(driver), want: []bool{true, false, false}},
	} {
		t.Run(selector.name, func(t *testing.T) {
			compiled := dracel.GetCompiler(dracel.Features{EnableConsumableCapacity: true}).CompileCELExpression(selector.expression, dracel.Options{})
			if compiled.Error != nil {
				t.Fatalf("compile marker selector: %v", compiled.Error)
			}
			for i, marker := range []string{"", "m2", "m1"} {
				attributes := map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}
				if marker != "" {
					attributes[resourceapi.QualifiedName(driver+"/marker")] = resourceapi.DeviceAttribute{StringValue: ptr.To(marker)}
				}
				got, _, err := compiled.DeviceMatches(context.Background(), dracel.Device{
					Driver: driver, AllowMultipleAllocations: ptr.To(true), Attributes: attributes,
				})
				if err != nil || got != selector.want[i] {
					t.Fatalf("marker %q: matches=%t err=%v, want %t without evaluation error", marker, got, err, selector.want[i])
				}
			}
		})
	}
}
