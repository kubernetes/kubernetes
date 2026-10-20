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

package scheduler

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

// sharingAffinityInFlightSnapshot captures sharers and tokens per in-flight
// claim, read under the tracker's in-flight lock.
func sharingAffinityInFlightSnapshot(h *sharingAffinityScheduleHarness) map[types.UID]string {
	tracker := sharingAffinityClaimTracker(h.dra).Elem()
	mu := forceField(tracker, "inFlightMutex").Addr().Interface().(*sync.RWMutex)
	mu.RLock()
	defer mu.RUnlock()
	got := map[types.UID]string{}
	m := forceField(tracker, "inFlightAllocations")
	for _, k := range m.MapKeys() {
		e := m.MapIndex(k)
		tokens := sets.List(forceField(copyValue(e), "tokens").Interface().(sets.Set[types.UID]))
		got[types.UID(k.String())] = fmt.Sprintf("sharers=%d tokens=%v", forceField(copyValue(e), "sharers").Int(), tokens)
	}
	return got
}

func copyValue(v reflect.Value) reflect.Value {
	c := reflect.New(v.Type()).Elem()
	c.Set(v)
	return c
}

// sharingAffinityInjectToken attaches a reservation token to an existing
// in-flight entry, standing in for a concurrent tokenized Reserve.
func sharingAffinityInjectToken(h *sharingAffinityScheduleHarness, uid types.UID, token types.UID) {
	tracker := sharingAffinityClaimTracker(h.dra).Elem()
	mu := forceField(tracker, "inFlightMutex").Addr().Interface().(*sync.RWMutex)
	mu.Lock()
	defer mu.Unlock()
	m := forceField(tracker, "inFlightAllocations")
	key := reflect.ValueOf(uid)
	e := copyValue(m.MapIndex(key))
	forceSetField(forceField(e, "tokens"), sets.New[types.UID](token))
	m.SetMapIndex(key, e)
}

func sharingAffinityCapacitySnapshot(t *testing.T, h *sharingAffinityScheduleHarness, pool string) string {
	t.Helper()
	var state *structured.AllocatedState
	var err error
	for i := 0; i < 10; i++ {
		if state, err = h.dra.ResourceClaims().GatherAllocatedState(); err == nil {
			break
		}
	}
	if err != nil {
		t.Fatalf("GatherAllocatedState: %v", err)
	}
	id := structured.MakeDeviceID("sharing.example.com", pool, "device")
	return fmt.Sprintf("%v shared=%v", state.AggregatedCapacity[id], state.AllocatedSharedDeviceIDs.Has(id))
}

// Composite and topology-aware PodGroups (and the basic shape for parity), with
// DRAWorkloadResourceClaims off and on, park after a real group cycle that is
// excluded by a pool held stale in pending reconciliation. Populated owner
// ReservedFor, in-flight sharers, tokens and consumed capacity are unchanged
// across failure, waiter registration and activation; only a worker-only
// reconciliation commit (no ResourceSlice, claim or Pod event) activates the
// parked group, after which every member binds.
func TestSharingAffinityPodGroupMatrixParkedReconciliationOnlyRecovery(t *testing.T) {
	for _, kind := range []string{"basic", "composite", "topology"} {
		for _, workloadClaims := range []bool{false, true} {
			name := fmt.Sprintf("%s-workload-claim-gate-%v", kind, map[bool]string{false: "off", true: "on"}[workloadClaims])
			t.Run(name, func(t *testing.T) {
				metrics.Register()
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:                 true,
					features.TopologyAwareWorkloadScheduling: kind != "basic",
					features.CompositePodGroup:               kind == "composite",
					features.DRAWorkloadResourceClaims:       workloadClaims,
				})
				const schedulerName = "sharing-affinity-pgm"
				const pool, node, topoKey = "pool-pgm", "node-pgm", "example.com/rack"
				pgw := st.MakePodGroup().Name("pg").Namespace("default").UID("pg").MinCount(2)
				objects := []runtime.Object{}
				switch kind {
				case "composite":
					pgw = pgw.ParentCompositePodGroup("cpg").WorkloadRef("wl", "pg")
					objects = append(objects, st.MakeCompositePodGroup().Name("cpg").Namespace("default").UID("cpg").
						MinGroupCount(1).BasicPolicy().WorkloadRef("wl", "cpg").Obj())
				case "topology":
					pgw = pgw.TopologyKey(topoKey)
				}
				pg := pgw.Obj()
				objects = append(objects, pg)
				var members []*v1.Pod
				for _, n := range []string{"member-a", "member-b"} {
					claim, pod := sharingAffinityGapClaim(n, schedulerName)
					pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To(pg.Name)}
					members = append(members, pod)
					objects = append(objects, claim)
				}
				inflight, _ := sharingAffinityGapClaim("inflight", "unused")
				slice := sharingAffinityGapSlice(pool, node, true)
				slice.Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"bandwidth": {Value: resource.MustParse("10Gi"), RequestPolicy: &resourceapi.CapacityRequestPolicy{Default: ptr.To(resource.MustParse("1Gi"))}},
				}
				objects = append(objects, inflight, slice,
					&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}})

				h := newSharingAffinityScheduleHarnessWithBackoff(t, objects, time.Hour, schedulerName)
				nodeObj := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: node, Labels: map[string]string{topoKey: "rack-1"}}}
				if _, err := h.client.CoreV1().Nodes().Create(h.ctx, nodeObj, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
				h.poll(t, "node cached", func() bool {
					if err := h.scheduler.Cache.UpdateSnapshot(h.logger, h.snapshot); err != nil {
						return false
					}
					nodes, err := h.snapshot.List()
					return err == nil && len(nodes) > 0
				})

				// Populated owner: running sharer in ReservedFor plus consumed capacity.
				owner := sharingAffinityGapOwner(pool)
				owner.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{{Resource: "pods", Name: "running-sharer", UID: "running-sharer"}}
				owner.Status.Allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"bandwidth": resource.MustParse("1Gi")}
				if _, err := h.client.ResourceV1().ResourceClaims(owner.Namespace).Create(h.ctx, owner.DeepCopy(), metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
				releaseWorker, releaseObserved, observedDone := holdObservedAndHandOffToWorker(t, h, owner, pool)

				// Populated in-flight state: two sharers plus a reservation token.
				pending := inflight.DeepCopy()
				pending.Status.Allocation = &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
					Request: "req", Driver: "sharing.example.com", Pool: pool, Device: "device", ShareID: ptr.To(types.UID("inflight-share")),
					ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"bandwidth": resource.MustParse("2Gi")},
				}}}}
				for i := 0; i < 2; i++ {
					if err := h.dra.ResourceClaims().SignalClaimPendingAllocation(inflight.UID, pending); err != nil {
						t.Fatalf("signal in-flight: %v", err)
					}
				}
				sharingAffinityInjectToken(h, inflight.UID, "token-1")
				wantInFlight := map[types.UID]string{inflight.UID: "sharers=2 tokens=[token-1]"}
				if got := sharingAffinityInFlightSnapshot(h); !reflect.DeepEqual(got, wantInFlight) {
					t.Fatalf("seeded in-flight = %v, want %v", got, wantInFlight)
				}
				capBefore := sharingAffinityCapacitySnapshot(t, h, pool)
				// Owner 1Gi plus in-flight 2Gi must both be counted.
				if want := "map[sharing.example.com/bandwidth:3Gi] shared=true"; capBefore != want {
					t.Fatalf("populated capacity = %s, want %s", capBefore, want)
				}
				assertPopulated := func(stage string) {
					t.Helper()
					if got := sharingAffinityInFlightSnapshot(h); !reflect.DeepEqual(got, wantInFlight) {
						t.Fatalf("%s: in-flight = %v, want %v", stage, got, wantInFlight)
					}
					if got := sharingAffinityCapacitySnapshot(t, h, pool); got != capBefore {
						t.Fatalf("%s: consumed capacity = %s, want %s", stage, got, capBefore)
					}
					gotOwner, err := h.client.ResourceV1().ResourceClaims(owner.Namespace).Get(h.ctx, owner.Name, metav1.GetOptions{})
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(gotOwner.Status.ReservedFor, owner.Status.ReservedFor) ||
						!reflect.DeepEqual(gotOwner.Status.Allocation, owner.Status.Allocation) {
						t.Fatalf("%s: owner status changed: %#v", stage, gotOwner.Status)
					}
				}

				var mu sync.Mutex
				var writes []string
				record := func(action clienttesting.Action) (bool, runtime.Object, error) {
					switch action.GetVerb() {
					case "get", "list", "watch":
					default:
						mu.Lock()
						writes = append(writes, action.GetVerb()+"/"+action.GetResource().Resource+"/"+action.GetSubresource())
						mu.Unlock()
					}
					return false, nil, nil
				}
				h.client.PrependReactor("*", "resourceclaims", record)
				h.client.PrependReactor("*", "resourceslices", record)
				pgPostFilterBefore := sharingAffinityExtensionPointCount(t, metrics.PodGroupPostFilter, schedulerName)
				postFilterBefore := sharingAffinityExtensionPointCount(t, metrics.PostFilter, schedulerName)

				h.createPods(t, members...)
				h.poll(t, "pod group active", func() bool { return len(h.queue.PodsInActiveQ()) == len(members) })
				h.scheduler.ScheduleOne(h.ctx)
				h.poll(t, "pod group parked", func() bool {
					return len(h.queue.PodsInActiveQ()) == 0 &&
						len(h.queue.UnschedulablePods())+len(h.queue.PodsInBackoffQ()) == len(members)
				})
				h.assertNoBinding(t)
				if got := sharingAffinityExtensionPointCount(t, metrics.PodGroupPostFilter, schedulerName) - pgPostFilterBefore; got != 1 {
					t.Fatalf("PodGroupPostFilter ran %d times, want 1", got)
				}
				if got := sharingAffinityExtensionPointCount(t, metrics.PostFilter, schedulerName) - postFilterBefore; got != 0 {
					t.Fatalf("per-Pod PostFilter ran %d times in a PodGroup cycle", got)
				}
				want := map[types.UID][]string{}
				for _, pod := range members {
					want[pod.UID] = []string{pool}
					if n := h.activator.count(pod.UID); n != 0 {
						t.Fatalf("stale pool activated %s %d times before reconciliation", pod.Name, n)
					}
				}
				assertSharingAffinityWaiters(t, h, schedulerName, want)
				assertPopulated("after failure and waiter registration")

				// Parked for an hour of backoff; only the worker commit can
				// bring the group back, with no API object events.
				releaseWorker()
				h.poll(t, "worker-only commit activates every member", func() bool {
					for _, pod := range members {
						if h.activator.count(pod.UID) != 1 {
							return false
						}
					}
					return len(h.queue.PodsInActiveQ()) == len(members)
				})
				select {
				case <-observedDone:
					t.Fatal("observed computation committed; recovery was not reconciliation-only")
				default:
				}
				mu.Lock()
				if len(writes) != 0 {
					t.Fatalf("recovery wrote API objects: %v", writes)
				}
				mu.Unlock()
				assertSharingAffinityWaiters(t, h, schedulerName, nil)
				assertPopulated("after activation")

				releaseObserved()
				sharingAffinityAwait(t, observedDone, "superseded observed commit")
				h.scheduler.ScheduleOne(h.ctx)
				got := map[string]string{}
				for len(got) < len(members) {
					select {
					case b := <-h.bound:
						got[b.Name] = b.Target.Name
					case <-time.After(5 * time.Second):
						t.Fatalf("timed out waiting for group bindings, got %v", got)
					}
				}
				for _, pod := range members {
					if got[pod.Name] != node {
						t.Fatalf("member %s bound to %q, want %s", pod.Name, got[pod.Name], node)
					}
					if n := h.activator.count(pod.UID); n != 1 {
						t.Fatalf("member %s activated %d times, want 1", pod.Name, n)
					}
				}
				if got := sharingAffinityInFlightSnapshot(h)[inflight.UID]; got != wantInFlight[inflight.UID] {
					t.Fatalf("after binding: unrelated in-flight entry = %q, want %q", got, wantInFlight[inflight.UID])
				}
			})
		}
	}
}
