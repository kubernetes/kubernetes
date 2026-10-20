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
	"reflect"
	"sort"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/ptr"
)

func sharingAffinityExtensionPointCount(t *testing.T, point, profileName string) uint64 {
	t.Helper()
	var total uint64
	for _, code := range []string{"Success", "Unschedulable", "UnschedulableAndUnresolvable", "Error"} {
		n, err := testutil.GetHistogramMetricCount(metrics.FrameworkExtensionPointDuration.WithLabelValues(point, code, profileName))
		if err != nil {
			t.Fatalf("read %s metric: %v", point, err)
		}
		total += n
	}
	return total
}

// A populated PodGroup whose members are all excluded by an incomplete
// sharing-affinity pool runs a real group cycle: per-Pod PostFilter is skipped,
// DynamicResources PodGroupPostFilter registers every member without touching
// the populated owner claim, only pool reconciliation activates the group, and
// the next group cycle places all members.
func TestSharingAffinityPodGroupCycleReconciliationOnlyRecovery(t *testing.T) {
	for _, workloadClaims := range []bool{false, true} {
		name := "basic-workload-claim-gate-off"
		if workloadClaims {
			name = "basic-workload-claim-gate-on"
		}
		t.Run(name, func(t *testing.T) {
			metrics.Register()
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:           true,
				features.DRAWorkloadResourceClaims: workloadClaims,
			})
			const schedulerName = "sharing-affinity-pg"
			const pool, node = "pool-pg", "node-pg"
			pg := &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pg", UID: "pg"},
				Spec: schedulingv1beta1.PodGroupSpec{SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
					Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 2},
				}},
			}
			var members []*v1.Pod
			objects := []runtime.Object{pg}
			for _, n := range []string{"member-a", "member-b"} {
				claim, pod := sharingAffinityGapClaim(n, schedulerName)
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To(pg.Name)}
				members = append(members, pod)
				objects = append(objects, claim)
			}
			// Populated pre-existing ownership: an in-use sharer with
			// ReservedFor, a share token and consumed capacity.
			owner := sharingAffinityGapOwner(pool)
			owner.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{{Resource: "pods", Name: "running-sharer", UID: "running-sharer"}}
			objects = append(objects, owner,
				sharingAffinityGapSlice(pool, node, false),
				&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}})

			h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
			h.addNode(t, node)
			var claimWrites []string
			h.client.PrependReactor("*", "resourceclaims", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetVerb() != "get" && action.GetVerb() != "list" && action.GetVerb() != "watch" {
					claimWrites = append(claimWrites, action.GetVerb()+"/"+action.GetSubresource())
				}
				return false, nil, nil
			})
			postFilterBefore := sharingAffinityExtensionPointCount(t, metrics.PostFilter, schedulerName)
			pgPostFilterBefore := sharingAffinityExtensionPointCount(t, metrics.PodGroupPostFilter, schedulerName)

			h.createPods(t, members...)
			h.poll(t, "pod group active", func() bool { return len(h.queue.PodsInActiveQ()) == len(members) })
			// With zero backoff the harness queue requeues a failed group
			// straight back to activeQ; run a second incomplete-pool cycle to
			// prove failure is stable and registration is idempotent.
			for i := 0; i < 2; i++ {
				h.scheduler.ScheduleOne(h.ctx)
				h.poll(t, "pod group requeued after failed cycle", func() bool { return len(h.queue.PodsInActiveQ()) == len(members) })
				h.assertNoBinding(t)
			}

			if got := sharingAffinityExtensionPointCount(t, metrics.PostFilter, schedulerName) - postFilterBefore; got != 0 {
				t.Fatalf("per-Pod PostFilter ran %d times in a PodGroup cycle", got)
			}
			if got := sharingAffinityExtensionPointCount(t, metrics.PodGroupPostFilter, schedulerName) - pgPostFilterBefore; got != 2 {
				t.Fatalf("PodGroupPostFilter ran %d times, want 2", got)
			}
			want := map[types.UID][]string{}
			for _, pod := range members {
				want[pod.UID] = []string{pool}
			}
			assertSharingAffinityWaiters(t, h, schedulerName, want)
			if len(claimWrites) != 0 {
				t.Fatalf("failed affinity-excluded group cycle wrote claims: %v", claimWrites)
			}
			gotOwner, err := h.client.ResourceV1().ResourceClaims(owner.Namespace).Get(h.ctx, owner.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(gotOwner.Status, owner.Status) {
				t.Fatalf("owner status changed by group failure:\n got %#v\nwant %#v", gotOwner.Status, owner.Status)
			}
			for _, pod := range members {
				if got := h.activator.count(pod.UID); got != 0 {
					t.Fatalf("incomplete pool activated %s %d times", pod.Name, got)
				}
			}

			h.completePool(t, pool, node)
			h.poll(t, "reconciliation alone activates every member", func() bool {
				for _, pod := range members {
					if h.activator.count(pod.UID) != 1 {
						return false
					}
				}
				return len(h.queue.PodsInActiveQ()) == len(members)
			})
			assertSharingAffinityWaiters(t, h, schedulerName, nil)
			h.scheduler.ScheduleOne(h.ctx)
			got := map[string]string{}
			for len(got) < len(members) {
				select {
				case binding := <-h.bound:
					got[binding.Name] = binding.Target.Name
				case <-time.After(5 * time.Second):
					t.Fatalf("timed out waiting for group bindings, got %v", got)
				}
			}
			var names []string
			for _, pod := range members {
				names = append(names, pod.Name)
				if got[pod.Name] != node {
					t.Fatalf("member %s bound to %q, want %s; all %v", pod.Name, got[pod.Name], node, got)
				}
				if n := h.activator.count(pod.UID); n != 1 {
					t.Fatalf("member %s activated %d times, want exactly 1", pod.Name, n)
				}
			}
			sort.Strings(names)
			gotOwner, err = h.client.ResourceV1().ResourceClaims(owner.Namespace).Get(h.ctx, owner.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(gotOwner.Status.ReservedFor, owner.Status.ReservedFor) ||
				!reflect.DeepEqual(gotOwner.Status.Allocation, owner.Status.Allocation) {
				t.Fatalf("owner ReservedFor/allocation not preserved across recovery for %v: %#v", names, gotOwner.Status)
			}
		})
	}
}
