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
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"
	"unsafe"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/utils/ptr"
)

type sharingAffinityMultiActivator struct {
	fwk.PodActivator
	mu    sync.Mutex
	calls map[types.UID]int
}

func (a *sharingAffinityMultiActivator) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	a.mu.Lock()
	if a.calls == nil {
		a.calls = make(map[types.UID]int)
	}
	for _, pod := range pods {
		a.calls[pod.UID]++
	}
	a.mu.Unlock()
	a.PodActivator.Activate(logger, pods)
}

func (a *sharingAffinityMultiActivator) count(uid types.UID) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.calls[uid]
}

type sharingAffinityScheduleHarness struct {
	ctx       context.Context
	cancel    context.CancelFunc
	logger    klog.Logger
	client    *clientsetfake.Clientset
	factory   informers.SharedInformerFactory
	scheduler *Scheduler
	queue     internalqueue.SchedulingQueue
	snapshot  *internalcache.Snapshot
	bound     chan *v1.Binding
	activator *sharingAffinityMultiActivator
	dra       fwk.SharedDRAManager
	// profileCancel cancels a single profile's plugin context so tests can
	// prove that one profile's shutdown leaves the others subscribed.
	profileCancel map[string]context.CancelFunc
}

func newSharingAffinityScheduleHarness(t *testing.T, objects []runtime.Object, schedulerNames ...string) *sharingAffinityScheduleHarness {
	t.Helper()
	return newSharingAffinityScheduleHarnessWithBackoff(t, objects, 0, schedulerNames...)
}

// newSharingAffinityScheduleHarnessWithBackoff lets a failed entity park in
// backoffQ, so only an explicit activation can return it before the timer.
func newSharingAffinityScheduleHarnessWithBackoff(t *testing.T, objects []runtime.Object, backoff time.Duration, schedulerNames ...string) *sharingAffinityScheduleHarness {
	t.Helper()
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity:  true,
		features.DRASharingAffinity:     true,
		features.SchedulerAsyncAPICalls: false,
	})
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	client := clientsetfake.NewClientset(objects...)
	factory := informers.NewSharedInformerFactory(client, 0)
	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	t.Cleanup(broadcaster.Shutdown)
	var profilesConfig []schedulerconfig.KubeSchedulerProfile
	for _, schedulerName := range schedulerNames {
		profilesConfig = append(profilesConfig, schedulerconfig.KubeSchedulerProfile{
			SchedulerName: schedulerName,
			Plugins: &schedulerconfig.Plugins{
				QueueSort:  schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: queuesort.Name}}},
				MultiPoint: schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: dynamicresources.Name}}},
				Bind:       schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: defaultbinder.Name}}},
			},
			PluginConfig: []schedulerconfig.PluginConfig{{
				Name: dynamicresources.Name, Args: &schedulerconfig.DynamicResourcesArgs{},
			}},
		})
	}
	components, err := NewFrameworkComponents(ctx, client, factory, WithParallelism(1), WithProfiles(profilesConfig...))
	if err != nil {
		t.Fatal(err)
	}
	snapshot := internalcache.NewEmptySnapshot()
	activator := &sharingAffinityMultiActivator{}
	profiles := profile.Map{}
	profileCancel := map[string]context.CancelFunc{}
	for _, cfg := range profilesConfig {
		profileCtx, profileCancelFunc := context.WithCancel(ctx)
		single := *components
		single.options.profiles = []schedulerconfig.KubeSchedulerProfile{cfg}
		m, err := NewFrameworkMap(profileCtx, &single, func(name string) events.EventRecorderLogger {
			return broadcaster.NewRecorder(scheme.Scheme, name)
		}, snapshot, frameworkruntime.WithPodActivator(activator))
		if err != nil {
			profileCancelFunc()
			t.Fatal(err)
		}
		t.Cleanup(profileCancelFunc)
		profileCancel[cfg.SchedulerName] = profileCancelFunc
		for name, fw := range m {
			profiles[name] = fw
		}
	}
	t.Cleanup(func() {
		if err := profiles.Close(); err != nil {
			t.Errorf("close profiles: %v", err)
		}
	})
	hintsByProfile := internalqueue.QueueingHintMapPerProfile{}
	preEnqueue := map[string]map[string]fwk.PreEnqueuePlugin{}
	for name, profile := range profiles {
		hints, err := buildQueueingHintMap(ctx, profile.EnqueueExtensions())
		if err != nil {
			t.Fatal(err)
		}
		hintsByProfile[name] = hints
		preEnqueue[name] = map[string]fwk.PreEnqueuePlugin{}
		for _, plugin := range profile.PreEnqueuePlugins() {
			preEnqueue[name][plugin.Name()] = plugin
		}
	}
	queue := internalqueue.NewSchedulingQueue(profiles[schedulerNames[0]].QueueSortFunc(), factory,
		internalqueue.WithMetricsRecorder(components.metricsRecorder),
		internalqueue.WithPodLister(factory.Core().V1().Pods().Lister()),
		internalqueue.WithPreEnqueuePluginMap(preEnqueue),
		internalqueue.WithQueueingHintMapPerProfile(hintsByProfile),
		internalqueue.WithPodInitialBackoffDuration(backoff),
		internalqueue.WithPodMaxBackoffDuration(backoff))
	activator.PodActivator = queue
	for _, profile := range profiles {
		profile.SetPodNominator(queue)
	}
	sched := &Scheduler{
		Cache: components.cache, client: client, NextEntity: queue.Pop,
		SchedulingQueue: queue, Profiles: profiles, nodeInfoSnapshot: snapshot,
		StopEverything: ctx.Done(), logger: logger,
	}
	initTestAlgorithm(t, sched)
	sched.applyDefaultHandlers()
	if err := addAllEventHandlers(sched, factory, nil, components.resourceClaimCache, components.resourceSliceTracker,
		components.draManager, unionedGVKs(hintsByProfile)); err != nil {
		t.Fatal(err)
	}
	h := &sharingAffinityScheduleHarness{
		ctx: ctx, cancel: cancel, logger: logger, client: client, factory: factory,
		scheduler: sched, queue: queue, snapshot: snapshot, bound: make(chan *v1.Binding, 10), activator: activator,
		dra: components.draManager, profileCancel: profileCancel,
	}
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "binding" {
			return false, nil, nil
		}

		binding := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
		pods := v1.SchemeGroupVersion.WithResource("pods")
		obj, err := client.Tracker().Get(pods, binding.Namespace, binding.Name)
		if err != nil {
			return true, nil, err
		}
		scheduled := obj.(*v1.Pod).DeepCopy()
		scheduled.Spec.NodeName = binding.Target.Name
		scheduled.ResourceVersion = "2"
		if err := client.Tracker().Update(pods, scheduled, binding.Namespace); err != nil {
			return true, nil, err
		}
		h.bound <- binding.DeepCopy()
		return true, binding, nil
	})
	factory.Start(ctx.Done())
	t.Cleanup(func() {
		cancel()
		queue.Close()
		factory.Shutdown()
	})
	for informer, synced := range factory.WaitForCacheSync(ctx.Done()) {
		if !synced {
			t.Fatalf("informer %v did not sync", informer)
		}
	}
	if err := sched.WaitForHandlersSync(ctx); err != nil {
		t.Fatal(err)
	}
	queue.Run(logger)
	return h
}

func (h *sharingAffinityScheduleHarness) poll(t *testing.T, description string, condition func() bool) {
	t.Helper()
	if err := wait.PollUntilContextTimeout(h.ctx, 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return condition(), nil
	}); err != nil {
		t.Fatalf("%s: %v", description, err)
	}
}

// Queue presence alone can come from a ResourceSlice event that precedes
// owner reconciliation. Observe the genuine waiter callbacks before scheduling
// can bind a pod and remove its waiter.
func (h *sharingAffinityScheduleHarness) waitActivated(t *testing.T, description string, pods ...*v1.Pod) {
	t.Helper()
	h.poll(t, description, func() bool {
		for _, pod := range pods {
			if h.activator.count(pod.UID) < 1 {
				return false
			}
		}
		return true
	})
}

func (h *sharingAffinityScheduleHarness) addNode(t *testing.T, name string) {
	t.Helper()
	if _, err := h.client.CoreV1().Nodes().Create(h.ctx, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	h.poll(t, "node cached", func() bool {
		if err := h.scheduler.Cache.UpdateSnapshot(h.logger, h.snapshot); err != nil {
			return false
		}
		nodes, err := h.snapshot.List()
		return err == nil && len(nodes) > 0
	})
}

func forceSetField(field reflect.Value, value any) {
	reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(reflect.ValueOf(value))
}

func forceSetFieldValue(field, value reflect.Value) {
	reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(value)
}

func forceField(value reflect.Value, name string) reflect.Value {
	field := value.FieldByName(name)
	return reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem()
}

func sharingAffinityClaimTracker(manager fwk.SharedDRAManager) reflect.Value {
	return forceField(reflect.ValueOf(manager).Elem(), "resourceClaimTracker")
}

func sharingAffinityClaimCache(t *testing.T, manager fwk.SharedDRAManager) *assumecache.AssumeCache {
	t.Helper()
	cache, ok := forceField(sharingAffinityClaimTracker(manager).Elem(), "cache").Interface().(*assumecache.AssumeCache)
	if !ok {
		t.Fatalf("claim cache has unexpected type %T", forceField(sharingAffinityClaimTracker(manager).Elem(), "cache").Interface())
	}
	return cache
}

func setSharingAffinityObservedAfterCompute(manager fwk.SharedDRAManager, hook func()) {
	tracker := sharingAffinityClaimTracker(manager).Elem()
	allocated := forceField(tracker, "allocatedDevices").Elem()
	forceSetField(allocated.FieldByName("observedAfterComputeHook"), hook)
}

func setSharingAffinityPublishBeforeCommit(manager fwk.SharedDRAManager, hook func(string, string)) {
	tracker := sharingAffinityClaimTracker(manager).Elem()
	forceSetField(tracker.FieldByName("publishBeforeCommitHook"), hook)
}

func setSharingAffinityReserveAfterList(manager fwk.SharedDRAManager, hook func()) {
	tracker := sharingAffinityClaimTracker(manager).Elem()
	forceSetField(tracker.FieldByName("reserveAfterListHook"), hook)
}

func setSharingAffinityReconcileAfterCompute(manager fwk.SharedDRAManager, hook func()) {
	tracker := sharingAffinityClaimTracker(manager).Elem()
	field := tracker.FieldByName("reconcileAfterComputeHook")
	if hook == nil {
		forceSetFieldValue(field, reflect.Zero(field.Type()))
		return
	}
	forceSetFieldValue(field, reflect.MakeFunc(field.Type(), func([]reflect.Value) []reflect.Value {
		hook()
		return nil
	}))
}

func sharingAffinityGapClaim(name, schedulerName string) (*resourceapi.ResourceClaim, *v1.Pod) {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name + "-claim", Namespace: "default", UID: types.UID(name + "-claim"), ResourceVersion: "1"},
		Spec: resourceapi.ResourceClaimSpec{Devices: resourceapi.DeviceClaim{
			Requests: []resourceapi.DeviceRequest{{
				Name: "req",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: "network", AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				},
			}},
			Config: []resourceapi.DeviceClaimConfiguration{{
				Requests: []string{"req"},
				DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver:     "sharing.example.com",
					Parameters: runtime.RawExtension{Raw: []byte(`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"blue"}`)},
				}},
			}},
		}},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default", UID: types.UID(name), ResourceVersion: "1"},
		Spec: v1.PodSpec{
			SchedulerName: schedulerName,
			Containers:    []v1.Container{{Name: "container", Image: "image"}},
			ResourceClaims: []v1.PodResourceClaim{{
				Name: "resource", ResourceClaimName: ptr.To(claim.Name),
			}},
		},
	}
	return claim, pod
}

func sharingAffinityGapSlice(pool, node string, complete bool) *resourceapi.ResourceSlice {
	count := int64(1)
	if !complete {
		count = 2
	}
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: pool + "-devices", UID: types.UID(pool + "-devices"), ResourceVersion: "1"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: "sharing.example.com", NodeName: ptr.To(node),
			Pool: resourceapi.ResourcePool{Name: pool, Generation: 1, ResourceSliceCount: count},
			SharingAffinity: []resourceapi.SharingAffinityExtractor{{
				Name: "subnet", APIGroup: "config.example.com", Kind: "Network", Expression: `parameters.subnet`,
			}},
			Devices: []resourceapi.Device{{Name: "device", AllowMultipleAllocations: ptr.To(true)}},
		},
	}
}

func sharingAffinityGapOwner(pool string) *resourceapi.ResourceClaim {
	owner, _ := sharingAffinityGapClaim("owner-"+pool, "unused")
	owner.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
			Request: "req", Driver: "sharing.example.com", Pool: pool, Device: "device", ShareID: ptr.To(types.UID("owner-" + pool + "-share")),
		}}},
	}
	return owner
}

func (h *sharingAffinityScheduleHarness) completePool(t *testing.T, pool, node string) {
	t.Helper()
	missing := sharingAffinityGapSlice(pool, node, false)
	missing.Name = pool + "-completion"
	missing.UID = types.UID(missing.Name)
	missing.Spec.Devices = nil
	missing.Spec.SharingAffinity = nil
	if _, err := h.client.ResourceV1().ResourceSlices().Create(h.ctx, missing, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
}

func (h *sharingAffinityScheduleHarness) scheduleUntilBound(t *testing.T, pod *v1.Pod, node string) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	check := func(binding *v1.Binding) {
		t.Helper()
		if binding.Name != pod.Name || binding.Target.Name != node {
			t.Fatalf("binding = %s -> %s, want %s -> %s", binding.Name, binding.Target.Name, pod.Name, node)
		}
	}
	for {
		select {
		case binding := <-h.bound:
			check(binding)
			return
		case <-deadline:
			t.Fatalf("pod %s did not bind to %s", pod.Name, node)
		case <-tick.C:
			if len(h.queue.PodsInActiveQ()) == 1 {
				h.scheduler.ScheduleOne(h.ctx)
			}
		}
	}
}

func TestSharingAffinityTwoProfilesShareManagerActivateAndSchedule(t *testing.T) {
	claimA, podA := sharingAffinityGapClaim("consumer-a", "sharing-a")
	claimB, podB := sharingAffinityGapClaim("consumer-b", "sharing-b")
	objects := []runtime.Object{
		claimA, claimB,
		sharingAffinityGapOwner("pool-shared"),
		sharingAffinityGapSlice("pool-shared", "node-shared", false),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, "sharing-a", "sharing-b")
	h.addNode(t, "node-shared")
	for _, pod := range []*v1.Pod{podA, podB} {
		if _, err := h.client.CoreV1().Pods("default").Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	h.poll(t, "both pods active", func() bool { return len(h.queue.PodsInActiveQ()) == 2 })
	h.scheduler.ScheduleOne(h.ctx)
	h.scheduler.ScheduleOne(h.ctx)
	h.poll(t, "both pods parked as unschedulable", func() bool { return len(h.queue.UnschedulablePods()) == 2 })

	h.completePool(t, "pool-shared", "node-shared")
	h.waitActivated(t, "reconciliation activated both profiles' pods", podA, podB)
	h.poll(t, "both profiles activated through shared manager", func() bool {
		return len(h.queue.PodsInActiveQ()) == 2
	})
	got := map[string]string{}
	deadline := time.After(5 * time.Second)
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for len(got) < 2 {
		select {
		case binding := <-h.bound:
			got[binding.Name] = binding.Target.Name
		case <-deadline:
			t.Fatalf("timed out waiting for both bindings, got %v (active=%d unschedulable=%d backoff=%d)", got, len(h.queue.PodsInActiveQ()), len(h.queue.UnschedulablePods()), len(h.queue.PodsInBackoffQ()))
		case <-tick.C:
			// ScheduleOne returns before PreBind hands the claim to the
			// observed cache. Wait for that pod to bind or requeue before
			// driving the other profile, whose pool is pending during the
			// handoff. A parked retry still counts, so it is not ignored.
			active := len(h.queue.PodsInActiveQ())
			queued := active + len(h.queue.UnschedulablePods()) + len(h.queue.PodsInBackoffQ())
			if active > 0 && queued+len(got) == 2 {
				h.scheduler.ScheduleOne(h.ctx)
			}
		}
	}
	for _, pod := range []*v1.Pod{podA, podB} {
		if got[pod.Name] != "node-shared" {
			t.Fatalf("pod %s bound to %q, want node-shared; all bindings %v", pod.Name, got[pod.Name], got)
		}
		if got := h.activator.count(pod.UID); got > 1 {
			t.Fatalf("pod %s activated %d times, want at most one", pod.Name, got)
		}
	}
}

func TestSharingAffinityPriorityQueueQRetryDoesNotLoop(t *testing.T) {
	const schedulerName = "sharing-affinity"
	claim, pod := sharingAffinityGapClaim("consumer-q", schedulerName)
	objects := []runtime.Object{
		claim,
		sharingAffinityGapOwner("pool-q"),
		sharingAffinityGapSlice("pool-q", "node-q", false),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
	h.addNode(t, "node-q")
	if _, err := h.client.CoreV1().Pods("default").Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	h.poll(t, "pod active", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	h.scheduler.ScheduleOne(h.ctx)
	h.poll(t, "pod unschedulable", func() bool { return len(h.queue.UnschedulablePods()) == 1 })
	for i := 0; i < 3; i++ {
		h.queue.Activate(h.logger, map[string]*v1.Pod{fmt.Sprintf("default/%s", pod.Name): pod})
		h.poll(t, "manual retry active", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
		h.scheduler.ScheduleOne(h.ctx)
		h.poll(t, "manual retry parked once", func() bool { return len(h.queue.UnschedulablePods()) == 1 })
	}
	if h.activator.count(pod.UID) != 0 {
		t.Fatalf("unresolved pool activated pod %d times", h.activator.count(pod.UID))
	}
	select {
	case binding := <-h.bound:
		t.Fatalf("unresolved pool bound pod to %s", binding.Target.Name)
	default:
	}
}

func TestSharingAffinityPriorityQueueReconciliationOnlyActivation(t *testing.T) {
	const schedulerName = "sharing-affinity"
	claim, pod := sharingAffinityGapClaim("consumer-reconcile", schedulerName)
	objects := []runtime.Object{
		claim,
		sharingAffinityGapSlice("pool-reconcile", "node-reconcile", true),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
	h.addNode(t, "node-reconcile")

	entered, release := make(chan struct{}), make(chan struct{})
	var enterOnce, releaseOnce sync.Once
	setSharingAffinityObservedAfterCompute(h.dra, func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	})
	setSharingAffinityReconcileAfterCompute(h.dra, func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	})
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
	})
	owner := sharingAffinityGapOwner("pool-reconcile")
	go assumecache.AddTestObject(sharingAffinityClaimCache(t, h.dra), owner)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("owner reconciliation did not enter held computation")
	}
	if _, err := h.client.CoreV1().Pods("default").Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	h.poll(t, "pod active", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	h.scheduler.ScheduleOne(h.ctx)
	h.poll(t, "pod registered while owner reconciliation held", func() bool {
		return len(h.queue.UnschedulablePods())+len(h.queue.PodsInActiveQ()) == 1
	})
	releaseOnce.Do(func() { close(release) })
	h.poll(t, "owner reconciliation alone activates pod", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	h.scheduler.ScheduleOne(h.ctx)
	select {
	case binding := <-h.bound:
		if binding.Name != pod.Name || binding.Target.Name != "node-reconcile" {
			t.Fatalf("binding = %s -> %s, want %s -> node-reconcile", binding.Name, binding.Target.Name, pod.Name)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("pod did not bind after reconciliation-only activation")
	}
}

func TestSharingAffinityHeldListerAheadPublicationBoundedAttempts(t *testing.T) {
	const schedulerName = "sharing-affinity"
	claim, pod := sharingAffinityGapClaim("consumer-lister-ahead", schedulerName)
	objects := []runtime.Object{
		claim,
		sharingAffinityGapOwner("pool-ahead"),
		sharingAffinityGapSlice("pool-ahead", "node-ahead", true),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
	h.addNode(t, "node-ahead")
	h.poll(t, "initial owner settled", func() bool {
		state, err := h.dra.ResourceClaims().GatherAllocatedState()
		return err == nil && !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: "sharing.example.com", Pool: "pool-ahead"})
	})

	entered, release := make(chan struct{}), make(chan struct{})
	var enterOnce, releaseOnce sync.Once
	setSharingAffinityPublishBeforeCommit(h.dra, func(driver, pool string) {
		if driver == "sharing.example.com" && pool == "pool-ahead" {
			enterOnce.Do(func() { close(entered) })
			<-release
		}
	})
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
	})
	updated := sharingAffinityGapSlice("pool-ahead", "node-ahead", true)
	updated.ResourceVersion = "2"
	updated.Spec.Pool.Generation = 2
	updated.Spec.SharingAffinity[0].Expression = `"blue"`
	if _, err := h.client.ResourceV1().ResourceSlices().Update(h.ctx, updated, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("metadata publication was not held")
	}
	if _, err := h.client.CoreV1().Pods("default").Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	const attempts = 3
	for i := 0; i < attempts; i++ {
		h.poll(t, fmt.Sprintf("attempt %d active", i), func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
		h.scheduler.ScheduleOne(h.ctx)
		h.poll(t, fmt.Sprintf("attempt %d parked", i), func() bool { return len(h.queue.UnschedulablePods()) == 1 })
		// The final attempt stays parked so only the release can activate it.
		if i < attempts-1 {
			h.queue.Activate(h.logger, map[string]*v1.Pod{"default/" + pod.Name: pod})
		}
	}
	// Quiet window: the held publication must neither activate nor loop.
	time.Sleep(300 * time.Millisecond)
	if got := h.activator.count(pod.UID); got != 0 {
		t.Fatalf("held lister-ahead publication activated pod %d times", got)
	}
	if len(h.queue.PodsInActiveQ()) != 0 || len(h.queue.UnschedulablePods()) != 1 {
		t.Fatalf("pod left parked state before release: active=%d unschedulable=%d", len(h.queue.PodsInActiveQ()), len(h.queue.UnschedulablePods()))
	}
	releaseOnce.Do(func() { close(release) })
	h.waitActivated(t, "publication release reconciliation activates pod", pod)
	h.poll(t, "publication release settled", func() bool {
		state, err := h.dra.ResourceClaims().GatherAllocatedState()
		return err == nil && !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: "sharing.example.com", Pool: "pool-ahead"})
	})
	h.poll(t, "publication release activates pod", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	if got := h.activator.count(pod.UID); got != 1 {
		t.Fatalf("publication release activated pod %d times, want exactly one", got)
	}
	h.scheduleUntilBound(t, pod, "node-ahead")
	if got := h.activator.count(pod.UID); got != 1 {
		t.Fatalf("pod activated %d times after placement, want exactly one", got)
	}
}

func TestSharingAffinityScheduleOneReserveErrorBackoffPlacement(t *testing.T) {
	const schedulerName = "sharing-affinity"
	claim, pod := sharingAffinityGapClaim("consumer-reserve", schedulerName)
	owner := sharingAffinityGapOwner("pool-reserve")
	objects := []runtime.Object{
		claim, owner,
		sharingAffinityGapSlice("pool-reserve", "node-reserve", true),
		&resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}},
	}
	h := newSharingAffinityScheduleHarness(t, objects, schedulerName)
	h.addNode(t, "node-reserve")
	h.poll(t, "initial reserve owner settled", func() bool {
		state, err := h.dra.ResourceClaims().GatherAllocatedState()
		return err == nil && !state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: "sharing.example.com", Pool: "pool-reserve"})
	})

	entered, release := make(chan struct{}), make(chan struct{})
	var enterOnce, releaseOnce, updateOnce sync.Once
	setSharingAffinityObservedAfterCompute(h.dra, func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	})
	setSharingAffinityReconcileAfterCompute(h.dra, func() {
		enterOnce.Do(func() { close(entered) })
		<-release
	})
	setSharingAffinityReserveAfterList(h.dra, func() {
		updateOnce.Do(func() {
			updated := owner.DeepCopy()
			updated.ResourceVersion = "99"
			updated.Status.Allocation.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"units": resource.MustParse("7")}
			go assumecache.AddTestObject(sharingAffinityClaimCache(t, h.dra), updated)
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Error("owner update did not enter held computation")
			}
		})
	})
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
	})
	if _, err := h.client.CoreV1().Pods("default").Create(h.ctx, pod, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	h.poll(t, "pod active", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	h.scheduler.ScheduleOne(h.ctx)
	select {
	case binding := <-h.bound:
		t.Fatalf("Reserve Error attempt unexpectedly bound pod to %s", binding.Target.Name)
	default:
	}
	if len(h.queue.UnschedulablePods()) != 0 {
		t.Fatalf("Reserve Error parked pod as unschedulable: %v", h.queue.UnschedulablePods())
	}
	releaseOnce.Do(func() { close(release) })
	h.poll(t, "reserve-error retry active after backoff", func() bool { return len(h.queue.PodsInActiveQ()) == 1 })
	h.scheduleUntilBound(t, pod, "node-reserve")
}
