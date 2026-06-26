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

package kubelet

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/utils/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

// deferAdmitHandler is a controllable admit handler used to drive deferred
// admission tests. When admit is true it admits; otherwise it returns the
// configured defer/reason/message.
type deferAdmitHandler struct {
	admit       bool
	deferResult bool
	reason      string
	message     string
}

func (h *deferAdmitHandler) Admit(_ context.Context, attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if h.admit {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	return lifecycle.PodAdmitResult{
		Admit:   false,
		Defer:   h.deferResult,
		Reason:  h.reason,
		Message: h.message,
	}
}

// newDeferredAdmissionKubelet returns a test kubelet whose admission is driven
// by the returned controllable handler. The node is configured so benign pods
// pass the default predicate handlers, leaving admission decisions to the
// returned handler.
func newDeferredAdmissionKubelet(t *testing.T) (*TestKubelet, *deferAdmitHandler) {
	t.Helper()
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kl := testKubelet.kubelet
	kl.nodeLister = testNodeLister{nodes: []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{Name: string(kl.nodeName)},
			Status: v1.NodeStatus{
				Allocatable: v1.ResourceList{
					v1.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	h := &deferAdmitHandler{reason: "DeviceNotReady", message: "device not ready"}
	kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{h})
	return testKubelet, h
}

func newDeferTestPod(uid string) *v1.Pod {
	return podWithUIDNameNsSpec(types.UID(uid), "pod-"+uid, "ns", v1.PodSpec{
		NodeName:   string(testKubeletHostname),
		Containers: []v1.Container{{Name: "c1"}},
	})
}

// recordDispatches installs a syncPodFn on the fake pod workers that records the
// update type seen per pod UID, so tests can assert how a pod was dispatched.
func recordDispatches(kl *Kubelet) map[types.UID]kubetypes.SyncPodType {
	dispatched := map[types.UID]kubetypes.SyncPodType{}
	kl.podWorkers.(*fakePodWorkers).syncPodFn = func(_ context.Context, updateType kubetypes.SyncPodType, pod, _ *v1.Pod, _ *kubecontainer.PodStatus) (bool, func(), error) {
		if pod != nil {
			dispatched[pod.UID] = updateType
		}
		return false, nil, nil
	}
	return dispatched
}

func stepClock(t *testing.T, kl *Kubelet, d time.Duration) {
	t.Helper()
	fc, ok := kl.clock.(*testingclock.FakeClock)
	require.True(t, ok, "kubelet clock must be a fake clock for deferred admission tests")
	fc.Step(d)
}

// A pod whose admission is deferred is tracked and kept Pending (not rejected).
func TestDeferredAdmissionTracksDeferredPod(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-1")

	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})

	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "deferred pod should be tracked")
	require.NotContains(t, dispatched, pod.UID, "deferred pod must not be dispatched to the pod workers")
	if status, found := kl.statusManager.GetPodStatus(pod.UID); found {
		require.NotEqual(t, v1.PodFailed, status.Phase, "deferred pod must not be failed")
	}
}

// A previously deferred pod is dispatched with SyncPodCreate once admission succeeds.
func TestDeferredAdmissionRetrySucceeds(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-2")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	// Device becomes available.
	h.admit = true
	stepClock(t, kl, 10*time.Second)
	kl.retryDeferredAdmissions(tCtx)

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "admitted pod should be removed from the deferred map")
	require.Equal(t, kubetypes.SyncPodCreate, dispatched[pod.UID], "admitted pod should be dispatched with SyncPodCreate")
	if status, found := kl.statusManager.GetPodStatus(pod.UID); found {
		require.NotEqual(t, v1.PodFailed, status.Phase, "admitted pod must not be failed")
	}
}

// A deferred pod still within the timeout keeps being deferred.
func TestDeferredAdmissionStillDeferredWithinTimeout(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-3")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	stepClock(t, kl, deferredAdmissionTimeout/2)
	kl.retryDeferredAdmissions(tCtx)

	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "pod within the timeout should still be deferred")
	if status, found := kl.statusManager.GetPodStatus(pod.UID); found {
		require.NotEqual(t, v1.PodFailed, status.Phase, "pod within timeout must not be failed")
	}
}

// A deferred pod past the timeout is permanently rejected, preserving the
// original (non-synthetic) rejection reason.
func TestDeferredAdmissionTimesOut(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-4")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	// Advance past the deferral timeout; device is still unavailable.
	stepClock(t, kl, deferredAdmissionTimeout+time.Second)
	kl.retryDeferredAdmissions(tCtx)

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "timed-out pod should be removed from the deferred map")
	checkPodStatus(t, kl, pod, v1.PodFailed)
	status, _ := kl.statusManager.GetPodStatus(pod.UID)
	require.Equal(t, "DeviceNotReady", status.Reason, "rejection reason must not be relabeled as a timeout")
}

// A deferred pod that becomes termination-requested is dropped, not admitted.
func TestDeferredAdmissionTerminationRequestedNotAdmitted(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-5")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	// The pod is now requested for termination, and admission would otherwise succeed.
	kl.podWorkers.(*fakePodWorkers).terminationRequested = map[types.UID]bool{pod.UID: true}
	h.admit = true
	kl.retryDeferredAdmissions(tCtx)

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "terminating pod should be dropped from the deferred map")
	require.NotContains(t, dispatched, pod.UID, "terminating pod must not be admitted/dispatched")
	if status, found := kl.statusManager.GetPodStatus(pod.UID); found {
		require.NotEqual(t, v1.PodFailed, status.Phase, "terminating deferred pod must not be rejected")
	}
}

// A pod removed while deferred is cleaned up from the deferred map.
func TestDeferredAdmissionClearedOnRemove(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-6")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	kl.HandlePodRemoves(tCtx, []*v1.Pod{pod})

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "removed pod should be cleared from the deferred map")
}

// removeOrphanedDeferredAdmissions prunes deferred entries for pods no longer present.
func TestDeferredAdmissionOrphanCleanup(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("defer-7")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	// The remaining set does not contain the deferred pod.
	kl.allocationManager.RemoveOrphanedDeferredAdmissions(sets.New[types.UID]("some-other-pod"))

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "orphaned deferred pod should be pruned")
}

// HandlePodSyncs skips dispatch for deferred pods.
func TestDeferredAdmissionSkippedByHandlePodSyncs(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("sync-skip-1")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	kl.HandlePodSyncs(tCtx, []*v1.Pod{pod})

	require.NotContains(t, dispatched, pod.UID, "HandlePodSyncs must not dispatch a deferred pod")
}

// HandlePodUpdates skips dispatch for deferred pods.
func TestDeferredAdmissionSkippedByHandlePodUpdates(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("update-skip-1")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	kl.HandlePodUpdates(tCtx, []*v1.Pod{pod})

	require.NotContains(t, dispatched, pod.UID, "HandlePodUpdates must not dispatch a deferred pod")
}

// HandlePodReconcile skips dispatch for deferred pods.
func TestDeferredAdmissionSkippedByHandlePodReconcile(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	dispatched := recordDispatches(kl)

	h.admit = false
	h.deferResult = true
	pod := newDeferTestPod("reconcile-skip-1")
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.True(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID))

	kl.HandlePodReconcile(tCtx, []*v1.Pod{pod})

	require.NotContains(t, dispatched, pod.UID, "HandlePodReconcile must not dispatch a deferred pod")
}

// A non-deferrable rejection in HandlePodAdditions fails the pod immediately and
// does not track it as deferred.
func TestDeferredAdmissionNonDeferrableRejection(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet, h := newDeferredAdmissionKubelet(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	h.admit = false
	h.deferResult = false
	h.reason = "OutOfMemory"
	h.message = "node out of memory"
	pod := newDeferTestPod("reject-1")

	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})

	require.False(t, kl.allocationManager.IsPodAdmissionDeferred(pod.UID), "non-deferrable rejection must not be tracked")
	checkPodStatus(t, kl, pod, v1.PodFailed)
}
