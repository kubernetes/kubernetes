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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

const sharingAffinityPQDriver = "sharing.example.com"

func sharingAffinityPoolValue(manager fwk.SharedDRAManager, pool string) reflect.Value {
	queue := forceField(sharingAffinityClaimTracker(manager).Elem(), "allocatedDevices").Elem()
	add, _ := forceField(queue, "reconcile").Type().MethodByName("Add")
	value := reflect.New(add.Type.In(0)).Elem()
	forceSetField(value.FieldByName("driver"), sharingAffinityPQDriver)
	forceSetField(value.FieldByName("pool"), pool)
	return value
}

// enqueueSharingAffinityPool adds a pool to the reconciler queue the same way
// metadata publication does, without delivering any informer event.
func enqueueSharingAffinityPool(manager fwk.SharedDRAManager, pool string) {
	allocated := forceField(sharingAffinityClaimTracker(manager).Elem(), "allocatedDevices").Elem()
	forceField(allocated, "reconcile").MethodByName("Add").Call([]reflect.Value{sharingAffinityPoolValue(manager, pool)})
}

// holdSharingAffinityWorker blocks only the reconciler worker for one pool.
func holdSharingAffinityWorker(t *testing.T, manager fwk.SharedDRAManager, pool string) (entered <-chan struct{}, release func()) {
	t.Helper()
	enteredCh, releaseCh := make(chan struct{}), make(chan struct{})
	var enterOnce, releaseOnce sync.Once
	release = func() { releaseOnce.Do(func() { close(releaseCh) }) }
	t.Cleanup(release)
	field := forceField(sharingAffinityClaimTracker(manager).Elem(), "reconcileAfterComputeHook")
	forceSetFieldValue(field, reflect.MakeFunc(field.Type(), func(args []reflect.Value) []reflect.Value {
		if args[0].FieldByName("pool").String() == pool {
			enterOnce.Do(func() { close(enteredCh) })
			<-releaseCh
		}
		return nil
	}))
	return enteredCh, release
}

// holdObservedAndHandOffToWorker adds an owner whose observed computation stays
// held, then lets the worker take over the pending record and hold it too.
// Releasing only the worker therefore proves a worker-only commit.
func holdObservedAndHandOffToWorker(t *testing.T, h *sharingAffinityScheduleHarness, owner *resourceapi.ResourceClaim, pool string) (releaseWorker, releaseObserved func(), observedDone <-chan struct{}) {
	t.Helper()
	obsEntered, obsRelease := make(chan struct{}), make(chan struct{})
	var obsEnterOnce, obsReleaseOnce sync.Once
	releaseObserved = func() { obsReleaseOnce.Do(func() { close(obsRelease) }) }
	t.Cleanup(releaseObserved)
	setSharingAffinityObservedAfterCompute(h.dra, func() {
		obsEnterOnce.Do(func() { close(obsEntered); <-obsRelease })
	})
	wkEntered, releaseWorker := holdSharingAffinityWorker(t, h.dra, pool)
	done := make(chan struct{})
	go func() {
		defer close(done)
		assumecache.AddTestObject(sharingAffinityClaimCache(t, h.dra), owner)
	}()
	sharingAffinityAwait(t, obsEntered, "observed owner computation held")
	enqueueSharingAffinityPool(h.dra, pool)
	sharingAffinityAwait(t, wkEntered, "worker takeover computation held")
	return releaseWorker, releaseObserved, done
}

func sharingAffinityAwait(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// sharingAffinityWaiterIndex returns the pools each Pod waits for in one
// profile's DynamicResources plugin.
func sharingAffinityWaiterIndex(t *testing.T, h *sharingAffinityScheduleHarness, profileName string) map[types.UID][]string {
	t.Helper()
	plugins := forceField(reflect.ValueOf(h.scheduler.Profiles[profileName]).Elem(), "pluginsMap")
	plugin := plugins.MapIndex(reflect.ValueOf(dynamicresources.Name)).Elem().Elem()
	waiters := forceField(plugin, "affinityWaiters").Elem()
	mu := forceField(waiters, "mu").Addr().Interface().(*sync.Mutex)
	mu.Lock()
	defer mu.Unlock()
	got := map[types.UID][]string{}
	byPod := forceField(waiters, "byPod")
	for _, uid := range byPod.MapKeys() {
		var pools []string
		for _, pool := range byPod.MapIndex(uid).MapKeys() {
			pools = append(pools, pool.FieldByName("pool").String())
		}
		sort.Strings(pools)
		got[types.UID(uid.String())] = pools
	}
	return got
}

func assertSharingAffinityWaiters(t *testing.T, h *sharingAffinityScheduleHarness, profileName string, want map[types.UID][]string) {
	t.Helper()
	got := sharingAffinityWaiterIndex(t, h, profileName)
	if want == nil {
		want = map[types.UID][]string{}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("profile %s waiter index = %v, want %v", profileName, got, want)
	}
}

func sharingAffinitySelectingClass(name, pool string) *resourceapi.DeviceClass {
	return &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.DeviceClassSpec{Selectors: []resourceapi.DeviceSelector{{
			CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["sharing.example.com"].pool == "` + pool + `"`},
		}}},
	}
}

func sharingAffinityAttributedSlice(pool, node string, complete bool) *resourceapi.ResourceSlice {
	slice := sharingAffinityGapSlice(pool, node, complete)
	slice.Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"pool": {StringValue: ptr.To(pool)},
	}
	return slice
}

func (h *sharingAffinityScheduleHarness) createPods(t *testing.T, pods ...*v1.Pod) {
	t.Helper()
	for _, pod := range pods {
		if _, err := h.client.CoreV1().Pods(pod.Namespace).Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
}

// scheduleActive runs one cycle per active Pod and returns bindings made.
func (h *sharingAffinityScheduleHarness) scheduleActive(t *testing.T, n int, wantBindings int) map[string]string {
	t.Helper()
	h.poll(t, "pods active", func() bool { return len(h.queue.PodsInActiveQ()) == n })
	for i := 0; i < n; i++ {
		h.scheduler.ScheduleOne(h.ctx)
	}
	got := map[string]string{}
	for len(got) < wantBindings {
		select {
		case binding := <-h.bound:
			got[binding.Name] = binding.Target.Name
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %d bindings, got %v", wantBindings, got)
		}
	}
	return got
}

func (h *sharingAffinityScheduleHarness) assertNoBinding(t *testing.T) {
	t.Helper()
	select {
	case binding := <-h.bound:
		t.Fatalf("unexpected binding %s -> %s", binding.Name, binding.Target.Name)
	default:
	}
}

// P recovers only through the reconciler worker while Q stays incomplete. The
// P-fitting Pod is activated once and placed; the Q-only Pod is retried once,
// re-registers only on Q, stays parked over a quiet window and is activated
// exactly once more when Q resolves.
func TestSharingAffinityPriorityQueueReconcilerOnlyPQRecovery(t *testing.T) {
	const schedulerName = "sharing-affinity"
	claimP, podP := sharingAffinityGapClaim("consumer-p", schedulerName)
	claimQ, podQ := sharingAffinityGapClaim("consumer-q-only", schedulerName)
	claimQ.Spec.Devices.Requests[0].Exactly.DeviceClassName = "network-q"
	objects := []runtime.Object{
		claimP, claimQ,
		sharingAffinityGapOwner("pool-q"),
		sharingAffinityAttributedSlice("pool-p", "node-p", true),
		sharingAffinityAttributedSlice("pool-q", "node-q", false),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
		sharingAffinitySelectingClass("network-q", "pool-q"),
	}
	h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
	h.addNode(t, "node-p")
	h.addNode(t, "node-q")

	releaseWorker, releaseObserved, observedDone := holdObservedAndHandOffToWorker(t, h, sharingAffinityGapOwner("pool-p"), "pool-p")
	h.createPods(t, podP, podQ)
	if got := h.scheduleActive(t, 2, 0); len(got) != 0 {
		t.Fatalf("pods bound while P and Q are excluded: %v", got)
	}
	h.poll(t, "both pods parked", func() bool { return len(h.queue.UnschedulablePods()) == 2 })
	assertSharingAffinityWaiters(t, h, schedulerName, map[types.UID][]string{
		podP.UID: {"pool-p", "pool-q"},
		podQ.UID: {"pool-p", "pool-q"},
	})
	if h.activator.count(podP.UID)+h.activator.count(podQ.UID) != 0 {
		t.Fatal("excluded pools activated waiting pods before reconciliation")
	}

	// Only the worker commits P; the observed computation is still held and
	// blocks claim event delivery, so activation must come from the worker.
	releaseWorker()
	h.poll(t, "worker-only P commit activates both waiters", func() bool { return len(h.queue.PodsInActiveQ()) == 2 })
	select {
	case <-observedDone:
		t.Fatal("observed computation committed before release; P recovery was not worker-only")
	default:
	}
	if p, q := h.activator.count(podP.UID), h.activator.count(podQ.UID); p != 1 || q != 1 {
		t.Fatalf("worker-only P activations: P=%d Q=%d, want 1 and 1", p, q)
	}
	// The superseded observed commit must not add activations. Release it
	// before binding: PreBind needs claim events to flow.
	releaseObserved()
	sharingAffinityAwait(t, observedDone, "superseded observed commit")
	got := h.scheduleActive(t, 2, 1)
	if got[podP.Name] != "node-p" || len(got) != 1 {
		t.Fatalf("bindings after P recovery = %v, want only %s -> node-p", got, podP.Name)
	}
	h.poll(t, "Q-only pod parked again", func() bool {
		pods := h.queue.UnschedulablePods()
		return len(pods) == 1 && pods[0].UID == podQ.UID
	})
	assertSharingAffinityWaiters(t, h, schedulerName, map[types.UID][]string{podQ.UID: {"pool-q"}})
	if p, q := h.activator.count(podP.UID), h.activator.count(podQ.UID); p != 1 || q != 1 {
		t.Fatalf("P resolution activations: P=%d Q=%d, want 1 and 1", p, q)
	}

	// Quiet window: the Q-only Pod must not loop before Q resolves.
	time.Sleep(300 * time.Millisecond)
	if q := h.activator.count(podQ.UID); q != 1 {
		t.Fatalf("Q-only pod activated %d times before Q resolved, want 1", q)
	}
	if len(h.queue.PodsInActiveQ()) != 0 || len(h.queue.UnschedulablePods()) != 1 {
		t.Fatalf("Q-only pod looped: active=%d unschedulable=%d", len(h.queue.PodsInActiveQ()), len(h.queue.UnschedulablePods()))
	}
	h.assertNoBinding(t)

	h.completePool(t, "pool-q", "node-q")
	got = h.scheduleActive(t, 1, 1)
	if got[podQ.Name] != "node-q" {
		t.Fatalf("Q-only pod bindings = %v, want %s -> node-q", got, podQ.Name)
	}
	if p, q := h.activator.count(podP.UID), h.activator.count(podQ.UID); p != 1 || q != 2 {
		t.Fatalf("final activations: P=%d Q=%d, want 1 and 2", p, q)
	}
	assertSharingAffinityWaiters(t, h, schedulerName, nil)
}

// Both real profiles recover from a worker-only reconciliation commit. After one
// profile is cancelled, the other stays subscribed and is still activated.
func TestSharingAffinityTwoProfilesReconcilerOnlyAndIndependentCancellation(t *testing.T) {
	const pool = "pool-profiles"
	claimA, podA := sharingAffinityGapClaim("consumer-a1", "sharing-a")
	claimB, podB := sharingAffinityGapClaim("consumer-b1", "sharing-b")
	claimA2, podA2 := sharingAffinityGapClaim("consumer-a2", "sharing-a")
	claimB2, podB2 := sharingAffinityGapClaim("consumer-b2", "sharing-b")
	objects := []runtime.Object{
		claimA, claimB, claimA2, claimB2,
		sharingAffinityGapSlice(pool, "node-profiles", true),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, "sharing-a", "sharing-b")
	h.addNode(t, "node-profiles")

	owner := sharingAffinityGapOwner(pool)
	releaseWorker, releaseObserved, observedDone := holdObservedAndHandOffToWorker(t, h, owner, pool)
	h.createPods(t, podA, podB)
	h.scheduleActive(t, 2, 0)
	h.poll(t, "both profiles parked", func() bool { return len(h.queue.UnschedulablePods()) == 2 })
	assertSharingAffinityWaiters(t, h, "sharing-a", map[types.UID][]string{podA.UID: {pool}})
	assertSharingAffinityWaiters(t, h, "sharing-b", map[types.UID][]string{podB.UID: {pool}})

	releaseWorker()
	h.poll(t, "worker-only commit activates both profiles", func() bool { return len(h.queue.PodsInActiveQ()) == 2 })
	select {
	case <-observedDone:
		t.Fatal("observed computation committed before release; activation was not worker-only")
	default:
	}
	for _, pod := range []*v1.Pod{podA, podB} {
		if n := h.activator.count(pod.UID); n != 1 {
			t.Fatalf("pod %s activated %d times, want exactly one", pod.Name, n)
		}
	}
	releaseObserved()
	sharingAffinityAwait(t, observedDone, "superseded observed commit")
	got := h.scheduleActive(t, 2, 2)
	if got[podA.Name] != "node-profiles" || got[podB.Name] != "node-profiles" {
		t.Fatalf("bindings = %v, want both profiles on node-profiles", got)
	}
	for _, pod := range []*v1.Pod{podA, podB} {
		if n := h.activator.count(pod.UID); n != 1 {
			t.Fatalf("pod %s activated %d times after placement, want exactly one", pod.Name, n)
		}
	}

	h.profileCancel["sharing-a"]()
	h.poll(t, "cancelled profile waiters closed", func() bool {
		waiters := forceField(forceField(reflect.ValueOf(h.scheduler.Profiles["sharing-a"]).Elem(), "pluginsMap").
			MapIndex(reflect.ValueOf(dynamicresources.Name)).Elem().Elem(), "affinityWaiters").Elem()
		mu := forceField(waiters, "mu").Addr().Interface().(*sync.Mutex)
		mu.Lock()
		defer mu.Unlock()
		return waiters.FieldByName("closed").Bool()
	})

	// A second owner with a new UID re-fences the pool via reconciliation only.
	owner2 := sharingAffinityGapOwner(pool)
	owner2.Name, owner2.UID = "owner-"+pool+"-2", types.UID("owner-"+pool+"-2")
	releaseWorker2, releaseObserved2, observedDone2 := holdObservedAndHandOffToWorker(t, h, owner2, pool)
	h.createPods(t, podA2, podB2)
	h.scheduleActive(t, 2, 0)
	h.poll(t, "second wave parked", func() bool { return len(h.queue.UnschedulablePods()) == 2 })
	assertSharingAffinityWaiters(t, h, "sharing-a", nil)
	assertSharingAffinityWaiters(t, h, "sharing-b", map[types.UID][]string{podB2.UID: {pool}})

	releaseWorker2()
	h.poll(t, "live profile activated", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	select {
	case <-observedDone2:
		t.Fatal("second observed computation committed before release")
	default:
	}
	releaseObserved2()
	sharingAffinityAwait(t, observedDone2, "second superseded observed commit")
	got = h.scheduleActive(t, 1, 1)
	if got[podB2.Name] != "node-profiles" || len(got) != 1 {
		t.Fatalf("bindings after cancellation = %v, want only %s", got, podB2.Name)
	}
	if a, b := h.activator.count(podA2.UID), h.activator.count(podB2.UID); a != 0 || b != 1 {
		t.Fatalf("post-cancellation activations: cancelled=%d live=%d, want 0 and 1", a, b)
	}
	pods := h.queue.UnschedulablePods()
	if len(pods) != 1 || pods[0].UID != podA2.UID {
		t.Fatalf("cancelled profile pod should stay parked, unschedulable=%v", pods)
	}
	assertSharingAffinityWaiters(t, h, "sharing-b", nil)
}
