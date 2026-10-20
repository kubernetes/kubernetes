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
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/utils/ptr"
)

// Publication may finish after the queue consumed its slice event. The excluded
// pool must activate its waiter when reconciliation finishes, without another event.
func TestAffinityLostWakeupRetry(t *testing.T) {
	slices := poolEventsSlices()
	pending := sharingAffinityAccountingClaim("readiness", "blue")
	pending.UID, pending.ResourceVersion = "", ""
	pending.Status.Allocation = nil
	pending.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
	pending.Spec.Devices.Requests[0].Exactly.Count = 1
	pending.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
		Requests: map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("5")},
	}
	allocated := sharingAffinityAccountingClaim("allocated", "blue")
	allocated.UID, allocated.ResourceVersion = "", ""
	tCtx, tc := poolEventsSetup(t, slices, pending, allocated)
	tracker := tc.draManager.resourceClaimTracker

	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	var once sync.Once
	tracker.publishMutex.Lock()
	tracker.publishBeforeCommitHook = func(string, string) {
		once.Do(func() {
			close(entered)
			<-release
		})
	}
	tracker.publishMutex.Unlock()

	updated := slices[0].DeepCopy()
	updated.Spec.SharingAffinity[0].Expression = `has(parameters.network) ? parameters.network : parameters.subnet`
	// The tracker dispatches handlers serially, so a delivery observer would
	// queue behind the blocked publisher; wait on the hook instead.
	client := tc.client.ResourceV1().ResourceSlices()
	current, err := client.Get(tCtx, updated.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ResourceSlice: %v", err)
	}
	updated.ResourceVersion = current.ResourceVersion
	if _, err := client.Update(tCtx, updated, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("update ResourceSlice: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("metadata publication for the update did not start")
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "lost-wakeup", Namespace: "default", UID: "lost-wakeup"},
		Spec: v1.PodSpec{ResourceClaims: []v1.PodResourceClaim{{
			Name: "resource", ResourceClaimName: ptr.To("readiness"),
		}}},
	}
	waiter, activator, _ := affinityPluginWaiter(t, tCtx, tc, pod)
	cycle := framework.NewCycleState()
	if _, status := tc.p.PreFilter(tCtx, cycle, pod, tc.nodeInfos); !status.IsSuccess() {
		t.Fatalf("PreFilter during delayed publication = %v, want pool-scoped exclusion", status)
	}
	status := tc.p.Filter(tCtx, cycle, pod, tc.nodeInfos[0])
	if status.Code() != fwk.Unschedulable {
		t.Fatalf("Filter during delayed publication = %v, want Unschedulable", status)
	}
	if want := updated.Spec.Driver + "/" + updated.Spec.Pool.Name; !strings.Contains(status.Message(), want) {
		t.Errorf("Filter status %q does not identify affected pool %q", status.Message(), want)
	}
	if msg := status.Message(); strings.Contains(msg, "parameters.subnet") {
		t.Errorf("status leaks extractor content: %q", msg)
	}
	if _, status := waiter.PostFilter(tCtx, cycle, pod, nil); status.Code() != fwk.Unschedulable {
		t.Fatalf("PostFilter = %v, want Unschedulable", status)
	}
	if activator.count() != 0 {
		t.Fatal("held publication prematurely activated its waiter")
	}

	// Pods whose claims are all allocated never consult the registry.
	allocatedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "all-allocated", Namespace: "default", UID: "all-allocated"},
		Spec: v1.PodSpec{ResourceClaims: []v1.PodResourceClaim{{
			Name: "resource", ResourceClaimName: ptr.To(allocated.Name),
		}}},
	}
	if _, status := tc.p.PreFilter(tCtx, framework.NewCycleState(), allocatedPod, tc.nodeInfos); !status.IsSuccess() && !status.IsSkip() {
		t.Fatalf("PreFilter for all-allocated pod during delayed publication = %v, want success", status)
	}

	unblock()
	// Acquiring the publisher lock waits for the blocked commit to finish.
	tracker.publishMutex.Lock()
	tracker.publishBeforeCommitHook = nil
	tracker.publishMutex.Unlock()

	affinityWaitForActivation(t, tCtx, activator, pod)
	cycle = framework.NewCycleState()
	_, status = tc.p.PreFilter(tCtx, cycle, pod, tc.nodeInfos)
	if !status.IsSuccess() {
		t.Fatalf("PreFilter after publication without new event: %v", status)
	}
	if status := tc.p.Filter(tCtx, cycle, pod, tc.nodeInfos[0]); !status.IsSuccess() {
		t.Fatalf("Filter after equal-map reconciliation without new event: %v", status)
	}
}
