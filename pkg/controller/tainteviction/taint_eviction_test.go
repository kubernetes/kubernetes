/*
Copyright 2017 The Kubernetes Authors.

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

package tainteviction

import (
	"context"
	"fmt"
	goruntime "runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/tainteviction/metrics"
	"k8s.io/kubernetes/pkg/controller/testutil"
	testingclock "k8s.io/utils/clock/testing"
)

var timeForControllerToProgressForSanityCheck = 20 * time.Millisecond

func getPodsAssignedToNode(ctx context.Context, c *fake.Clientset) GetPodsByNodeNameFunc {
	return func(nodeName string) ([]*corev1.Pod, error) {
		selector := fields.SelectorFromSet(fields.Set{"spec.nodeName": nodeName})
		pods, err := c.CoreV1().Pods(corev1.NamespaceAll).List(ctx, metav1.ListOptions{
			FieldSelector: selector.String(),
			LabelSelector: labels.Everything().String(),
		})
		if err != nil {
			return []*corev1.Pod{}, fmt.Errorf("failed to get Pods assigned to node %v", nodeName)
		}
		rPods := make([]*corev1.Pod, len(pods.Items))
		for i := range pods.Items {
			rPods[i] = &pods.Items[i]
		}
		return rPods, nil
	}
}

func createNoExecuteTaint(index int) corev1.Taint {
	now := metav1.Now()
	return corev1.Taint{
		Key:       "testTaint" + fmt.Sprintf("%v", index),
		Value:     "test" + fmt.Sprintf("%v", index),
		Effect:    corev1.TaintEffectNoExecute,
		TimeAdded: &now,
	}
}

func addToleration(pod *corev1.Pod, index int, duration int64) *corev1.Pod {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if duration < 0 {
		pod.Spec.Tolerations = []corev1.Toleration{{Key: "testTaint" + fmt.Sprintf("%v", index), Value: "test" + fmt.Sprintf("%v", index), Effect: corev1.TaintEffectNoExecute}}

	} else {
		pod.Spec.Tolerations = []corev1.Toleration{{Key: "testTaint" + fmt.Sprintf("%v", index), Value: "test" + fmt.Sprintf("%v", index), Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &duration}}
	}
	return pod
}

func addTaintsToNode(node *corev1.Node, key, value string, indices []int) *corev1.Node {
	taints := []corev1.Taint{}
	for _, index := range indices {
		taints = append(taints, createNoExecuteTaint(index))
	}
	node.Spec.Taints = taints
	return node
}

var alwaysReady = func() bool { return true }

type blockingPodLister struct {
	corelisters.PodLister
	captured chan struct{}
	release  chan struct{}
}

func (l *blockingPodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	return &blockingPodNamespaceLister{
		PodNamespaceLister: l.PodLister.Pods(namespace),
		captured:           l.captured,
		release:            l.release,
	}
}

type blockingPodNamespaceLister struct {
	corelisters.PodNamespaceLister
	captured chan struct{}
	release  chan struct{}
}

func (l *blockingPodNamespaceLister) Get(name string) (*corev1.Pod, error) {
	pod, err := l.PodNamespaceLister.Get(name)
	close(l.captured)
	<-l.release
	return pod, err
}

func setupNewController(ctx context.Context, fakeClientSet *fake.Clientset) (*Controller, cache.Indexer, cache.Indexer) {
	informerFactory := informers.NewSharedInformerFactory(fakeClientSet, 0)
	podIndexer := informerFactory.Core().V1().Pods().Informer().GetIndexer()
	nodeIndexer := informerFactory.Core().V1().Nodes().Informer().GetIndexer()
	mgr, _ := New(ctx, fakeClientSet, informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Nodes(), "taint-eviction-controller")
	mgr.podListerSynced = alwaysReady
	mgr.nodeListerSynced = alwaysReady
	mgr.getPodsAssignedToNode = getPodsAssignedToNode(ctx, fakeClientSet)
	return mgr, podIndexer, nodeIndexer
}

func useFakePodEvictionQueue(controller *Controller, fakeClock *testingclock.FakeClock) {
	useFakePodEvictionQueueWithBackoff(controller, fakeClock, time.Second, time.Second)
}

func useFakePodEvictionQueueWithBackoff(controller *Controller, fakeClock *testingclock.FakeClock, baseDelay, maxDelay time.Duration) {
	controller.podEvictionQueue.ShutDown()
	controller.podEvictionQueue = workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.NewTypedItemExponentialFailureRateLimiter[podEvictionItem](baseDelay, maxDelay),
		workqueue.TypedRateLimitingQueueConfig[podEvictionItem]{
			Name:  "test_noexec_taint_pod_eviction",
			Clock: fakeClock,
		},
	)
	controller.taintEvictionQueue.clock = fakeClock
}

func currentPodEvictionRetry(controller *Controller, podNamespacedName types.NamespacedName) (podEvictionItem, bool) {
	controller.podEvictionLock.Lock()
	defer controller.podEvictionLock.Unlock()
	state, ok := controller.podEvictionTokens[podNamespacedName.String()]
	return state.item, ok
}

func waitForPodEvictionRetry(controller *Controller, podRef NamespacedObject) (podEvictionItem, error) {
	var item podEvictionItem
	err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		var ok bool
		item, ok = currentPodEvictionRetry(controller, podRef.NamespacedName)
		return ok && item.podRef == podRef, nil
	})
	return item, err
}

func waitForTimedWorkerAfterRetry(controller *Controller, fakeClock *testingclock.FakeClock, podNamespacedName types.NamespacedName, delay time.Duration) error {
	var lastErr error
	for range 3 {
		fakeClock.Step(time.Second)
		if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 200*time.Millisecond, true, func(context.Context) (bool, error) {
			worker := controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String())
			return worker != nil && worker.FireAt.Sub(worker.CreatedAt) == delay, nil
		}); err == nil {
			return nil
		} else {
			lastErr = err
		}
	}
	return lastErr
}

type timestampedPod struct {
	names     []string
	timestamp time.Duration
}

type durationSlice []timestampedPod

func (a durationSlice) Len() int           { return len(a) }
func (a durationSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a durationSlice) Less(i, j int) bool { return a[i].timestamp < a[j].timestamp }

func TestFilterNoExecuteTaints(t *testing.T) {
	taints := []corev1.Taint{
		{
			Key:    "one",
			Value:  "one",
			Effect: corev1.TaintEffectNoExecute,
		},
		{
			Key:    "two",
			Value:  "two",
			Effect: corev1.TaintEffectNoSchedule,
		},
	}
	taints = getNoExecuteTaints(taints)
	if len(taints) != 1 || taints[0].Key != "one" {
		t.Errorf("Filtering doesn't work. Got %v", taints)
	}
}

func TestCreatePod(t *testing.T) {
	testCases := []struct {
		description  string
		pod          *corev1.Pod
		taintedNodes map[string][]corev1.Taint
		expectPatch  bool
		expectDelete bool
	}{
		{
			description:  "not scheduled - ignore",
			pod:          testutil.NewPod("pod1", ""),
			taintedNodes: map[string][]corev1.Taint{},
			expectDelete: false,
		},
		{
			description:  "scheduled on untainted Node",
			pod:          testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]corev1.Taint{},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node",
			pod:         testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "schedule on tainted Node with finite toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node with zero-second toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 1, 0),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "schedule on tainted Node with infinite toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description: "schedule on tainted Node with infinite invalid toleration",
			pod:         addToleration(testutil.NewPod("pod1", "node1"), 2, -1),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:  true,
			expectDelete: true,
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewSimpleClientset(&corev1.PodList{Items: []corev1.Pod{*item.pod}})
			controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
			controller.recorder = testutil.NewFakeRecorder()
			controller.taintedNodes = item.taintedNodes

			wg.Go(func() {
				controller.Run(ctx)
			})

			if err := podIndexer.Add(item.pod); err != nil {
				t.Fatalf("Failed to add pod to indexer: %v", err)
			}
			controller.PodUpdated(nil, item.pod)

			verifyPodActions(t, item.description, fakeClientset, item.expectPatch, item.expectDelete)

			cancel()
		})
	}
}

func TestPodEvictionDeletionFailureRetriesDurably(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	// The old implementation gave up after five immediate attempts.
	deniedAttempts := int32(6)
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= deniedAttempts {
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() >= 1, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for first delete attempt: %v", err)
	}

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	if _, err := waitForPodEvictionRetry(controller, podRef); err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}

	for deleteAttempts.Load() <= deniedAttempts {
		previous := deleteAttempts.Load()
		fakeClock.Step(time.Second)
		if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
			return deleteAttempts.Load() > previous, nil
		}); err != nil {
			t.Fatalf("Timed out waiting for durable retry after %d attempts: %v", previous, err)
		}
	}

	if got, want := deleteAttempts.Load(), deniedAttempts+1; got != want {
		t.Fatalf("Unexpected delete attempts: got %d, want %d", got, want)
	}
}

func TestPodEvictionDurableRetryKeepsRateLimiterState(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteAction := action.(clienttesting.DeleteAction)
		deleteAttempts.Add(1)
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	recorder := testutil.NewFakeRecorder()
	controller.recorder = recorder
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueueWithBackoff(controller, fakeClock, time.Second, 10*time.Second)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	item, err := waitForPodEvictionRetry(controller, podRef)
	if err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}
	if got, want := deleteAttempts.Load(), int32(retries); got != want {
		t.Fatalf("Initial timed worker should perform the legacy burst: got %d delete attempts, want %d", got, want)
	}
	if got := controller.podEvictionQueue.NumRequeues(item); got != 1 {
		t.Fatalf("Unexpected initial NumRequeues: got %d, want 1", got)
	}

	for i, step := range []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second} {
		fakeClock.Step(step)
		wantAttempts := int32(retries + i + 1)
		if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
			return deleteAttempts.Load() == wantAttempts, nil
		}); err != nil {
			t.Fatalf("Timed out waiting for durable retry %d after %s: %v", i+1, step, err)
		}

		current, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
		if !ok {
			t.Fatalf("Durable retry state disappeared after retry %d", i+1)
		}
		if current != item {
			t.Fatalf("Durable retry item changed after retry %d: got %#v, want %#v", i+1, current, item)
		}
		if got, want := controller.podEvictionQueue.NumRequeues(item), i+2; got != want {
			t.Fatalf("Unexpected NumRequeues after retry %d: got %d, want %d", i+1, got, want)
		}
	}

	recorder.Lock()
	defer recorder.Unlock()
	deletionEvents := 0
	for _, event := range recorder.Events {
		if event.Message == "Marking for deletion Pod default/pod1" {
			deletionEvents++
		}
	}
	if deletionEvents != 1 {
		t.Fatalf("Unexpected deletion event count after initial burst and durable retries: got %d, want 1", deletionEvents)
	}
}

func TestPodEvictionRetryRejectsOlderProducer(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	controller := &Controller{
		podEvictionTokens: make(map[string]podEvictionState),
		podEvictionQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.NewTypedItemExponentialFailureRateLimiter[podEvictionItem](time.Second, 10*time.Second),
			workqueue.TypedRateLimitingQueueConfig[podEvictionItem]{
				Name:  "test_noexec_taint_pod_eviction",
				Clock: fakeClock,
			},
		),
	}

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"}, UID: "pod1-uid"}
	olderCreatedAt := fakeClock.Now()
	olderFireAt := olderCreatedAt.Add(10 * time.Second)
	newerCreatedAt := olderCreatedAt.Add(time.Second)
	newerFireAt := olderCreatedAt.Add(5 * time.Second)

	newerItem, added := controller.addPodEvictionRetry(podRef, newerCreatedAt, newerFireAt, "node1", "")
	if !added {
		t.Fatalf("Expected newer retry producer to be accepted")
	}
	olderItem, added := controller.addPodEvictionRetry(podRef, olderCreatedAt, olderFireAt, "node1", "")
	if added {
		t.Fatalf("Expected older retry producer to be rejected")
	}

	current, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
	if !ok {
		t.Fatalf("Expected durable retry state to remain")
	}
	if current != newerItem {
		t.Fatalf("Older producer replaced newer retry state: got %#v, want %#v", current, newerItem)
	}
	if controller.podEvictionRetryMatches(olderItem) {
		t.Fatalf("Older retry item unexpectedly matched current retry state")
	}
	if !controller.podEvictionRetryMatches(newerItem) {
		t.Fatalf("Newer retry item did not match current retry state")
	}
	if got := controller.podEvictionQueue.NumRequeues(newerItem); got != 1 {
		t.Fatalf("Unexpected newer item NumRequeues: got %d, want 1", got)
	}
	if got := controller.podEvictionQueue.NumRequeues(olderItem); got != 0 {
		t.Fatalf("Older item should not have been rate limited: got NumRequeues %d, want 0", got)
	}
}

func TestPodEvictionDurableRetryWithZeroSecondTolerationDeletesDirectly(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteAction := action.(clienttesting.DeleteAction)
		deleteAttempts.Add(1)
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueueWithBackoff(controller, fakeClock, time.Second, 10*time.Second)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	item, err := waitForPodEvictionRetry(controller, podRef)
	if err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}
	if got, want := deleteAttempts.Load(), int32(retries); got != want {
		t.Fatalf("Initial timed worker should perform the legacy burst: got %d delete attempts, want %d", got, want)
	}

	zero := int64(0)
	zeroToleratingPod := pod.DeepCopy()
	zeroToleratingPod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint1",
		Value:             "test1",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &zero,
	}}
	if _, err := fakeClientset.CoreV1().Pods(zeroToleratingPod.Namespace).Update(ctx, zeroToleratingPod, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update pod in fake client: %v", err)
	}
	if err := podIndexer.Update(zeroToleratingPod); err != nil {
		t.Fatalf("Failed to update pod in indexer: %v", err)
	}

	fakeClock.Step(time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == int32(retries+1), nil
	}); err != nil {
		t.Fatalf("Timed out waiting for one durable retry delete attempt: %v", err)
	}

	current, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
	if !ok {
		t.Fatalf("Durable retry state disappeared after zero-second toleration retry")
	}
	if current != item {
		t.Fatalf("Durable retry item changed after zero-second toleration retry: got %#v, want %#v", current, item)
	}
	if got := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String()); got != nil {
		t.Fatalf("Unexpected timed worker handoff for zero-second toleration retry: %#v", got)
	}
	if got, want := controller.podEvictionQueue.NumRequeues(item), 2; got != want {
		t.Fatalf("Unexpected NumRequeues after zero-second toleration retry: got %d, want %d", got, want)
	}

	fakeClock.Step(2 * time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == int32(retries+2), nil
	}); err != nil {
		t.Fatalf("Timed out waiting for second durable retry delete attempt: %v", err)
	}
	if got := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String()); got != nil {
		t.Fatalf("Unexpected timed worker after second zero-second toleration retry: %#v", got)
	}
	if current, ok := currentPodEvictionRetry(controller, podRef.NamespacedName); !ok || current != item {
		t.Fatalf("Durable retry item changed or disappeared after second zero-second toleration retry: got %#v, ok=%v, want %#v", current, ok, item)
	}
	if got, want := controller.podEvictionQueue.NumRequeues(item), 3; got != want {
		t.Fatalf("Unexpected NumRequeues after second zero-second toleration retry: got %d, want %d", got, want)
	}
}

func TestPodEvictionRetryDoesNotDeleteReplacementPod(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= retries {
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	if _, err := waitForPodEvictionRetry(controller, podRef); err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}

	replacement := pod.DeepCopy()
	replacement.UID = "pod1-replacement-uid"
	if _, err := fakeClientset.CoreV1().Pods(replacement.Namespace).Update(ctx, replacement, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to replace pod in fake client: %v", err)
	}
	if err := podIndexer.Update(replacement); err != nil {
		t.Fatalf("Failed to replace pod in indexer: %v", err)
	}

	fakeClock.Step(time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		_, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
		return !ok, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for stale retry to be forgotten: %v", err)
	}

	if got := deleteAttempts.Load(); got != retries {
		t.Fatalf("Unexpected delete retry for replacement pod: got %d delete attempts, want %d", got, retries)
	}
}

func TestPodEvictionRetryCancelledWhenTaintRemoved(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= retries {
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, nodeIndexer := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == retries, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for initial delete attempts: %v", err)
	}

	if err := nodeIndexer.Add(testutil.NewNode("node1")); err != nil {
		t.Fatalf("Failed to add node to indexer: %v", err)
	}
	controller.handleNodeUpdate(ctx, nodeUpdateItem{"node1"})

	fakeClock.Step(time.Second)
	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		_, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
		return !ok, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for retry cancellation: %v", err)
	}

	if got := deleteAttempts.Load(); got != retries {
		t.Fatalf("Unexpected delete retry after taint removal: got %d delete attempts, want %d", got, retries)
	}
}

func TestPodEvictionRetrySchedulesNewDeadlineForFiniteToleration(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= retries {
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == retries, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for initial delete attempts: %v", err)
	}

	tolerationSeconds := int64(5)
	toleratingPod := pod.DeepCopy()
	toleratingPod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint1",
		Value:             "test1",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	if _, err := fakeClientset.CoreV1().Pods(toleratingPod.Namespace).Update(ctx, toleratingPod, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update pod in fake client: %v", err)
	}
	if err := podIndexer.Update(toleratingPod); err != nil {
		t.Fatalf("Failed to update pod in indexer: %v", err)
	}
	controller.handlePodUpdate(ctx, podUpdateItem{
		podName:      toleratingPod.Name,
		podNamespace: toleratingPod.Namespace,
		nodeName:     toleratingPod.Spec.NodeName,
	})

	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	if err := waitForTimedWorkerAfterRetry(controller, fakeClock, podNamespacedName, time.Duration(tolerationSeconds)*time.Second); err != nil {
		t.Fatalf("Timed out waiting for new finite toleration deadline: %v", err)
	}

	fakeClock.Step(time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 200*time.Millisecond, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == retries, nil
	}); err != nil {
		t.Fatalf("Pod was deleted before the new toleration deadline: %v", err)
	}

	fakeClock.Step(time.Duration(tolerationSeconds) * time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() > retries, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for delete after new toleration deadline: %v", err)
	}
}

func TestPodEvictionRetrySchedulesNewDeadlineForChangedTaint(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= retries {
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, nodeIndexer := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	controller.PodUpdated(nil, pod)

	podRef := NamespacedObject{NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, UID: pod.UID}
	if _, err := waitForPodEvictionRetry(controller, podRef); err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}

	tolerationSeconds := int64(5)
	toleratingPod := pod.DeepCopy()
	toleratingPod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint2",
		Value:             "test2",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	if _, err := fakeClientset.CoreV1().Pods(toleratingPod.Namespace).Update(ctx, toleratingPod, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update pod in fake client: %v", err)
	}
	if err := podIndexer.Update(toleratingPod); err != nil {
		t.Fatalf("Failed to update pod in indexer: %v", err)
	}

	newNode := addTaintsToNode(testutil.NewNode("node1"), "testTaint2", "taint2", []int{2})
	if err := nodeIndexer.Add(newNode); err != nil {
		t.Fatalf("Failed to add node to indexer: %v", err)
	}
	controller.handleNodeUpdate(ctx, nodeUpdateItem{nodeName: newNode.Name})

	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	if err := waitForTimedWorkerAfterRetry(controller, fakeClock, podNamespacedName, time.Duration(tolerationSeconds)*time.Second); err != nil {
		t.Fatalf("Timed out waiting for new taint deadline: %v", err)
	}

	fakeClock.Step(time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 200*time.Millisecond, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == retries, nil
	}); err != nil {
		t.Fatalf("Pod was deleted before the new taint deadline: %v", err)
	}

	fakeClock.Step(time.Duration(tolerationSeconds) * time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() > retries, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for delete after new taint deadline: %v", err)
	}
}

// TestPodEvictionRetryWithExpiredTolerationDeletesDirectly verifies that a durable
// retry occurring after a finite TolerationSeconds window has already expired does
// NOT restart that toleration window. The retry must delete the pod immediately
// rather than scheduling a fresh timed worker for another N seconds.
//
// Regression test for: https://github.com/kubernetes/kubernetes/issues/140639
func TestPodEvictionRetryWithExpiredTolerationDeletesDirectly(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Pod starts with TolerationSeconds: 5 for the matching taint so the initial
	// eviction goes through the timed-worker path (podEvictionLater), not the
	// immediate path (podEvictionNow).
	tolerationSeconds := int64(5)
	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	pod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint1",
		Value:             "test1",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	// All delete attempts fail so the durable retry queue is exercised.
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteAction := action.(clienttesting.DeleteAction)
		deleteAttempts.Add(1)
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	// Use 1s/10s backoff so each retry step can be driven deterministically
	// by advancing the fake clock by exactly one backoff period.
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueueWithBackoff(controller, fakeClock, time.Second, 10*time.Second)

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	// Trigger the initial eviction decision. Because the pod has TolerationSeconds: 5,
	// getPodEvictionDecision returns podEvictionLater and a timed worker is created.
	controller.PodUpdated(nil, pod)

	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}

	// Wait for the timed worker to appear in the queue with the 5s window.
	var fireAt time.Time
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		w := controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String())
		if w == nil || w.FireAt.Sub(w.CreatedAt) != time.Duration(tolerationSeconds)*time.Second {
			return false, nil
		}
		fireAt = w.FireAt
		return true, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for initial timed worker with 5s window: %v", err)
	}

	// No delete attempts should have happened yet — the pod is still in its
	// toleration window.
	if got := deleteAttempts.Load(); got != 0 {
		t.Fatalf("Delete should not have been attempted before toleration window expires: got %d attempts", got)
	}

	// Advance the fake clock past the 5-second toleration window to fire the
	// timed worker.
	fakeClock.SetTime(fireAt)

	// Wait for the initial burst of delete attempts to be exhausted.
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == int32(retries), nil
	}); err != nil {
		t.Fatalf("Timed out waiting for initial burst of %d delete attempts: %v", retries, err)
	}

	// Wait for the durable retry to be registered in podEvictionTokens. The
	// item carries the original createdAt/fireAt so the retry path knows when
	// the toleration window actually expired.
	podRef := NamespacedObject{NamespacedName: podNamespacedName, UID: pod.UID}
	item, err := waitForPodEvictionRetry(controller, podRef)
	if err != nil {
		t.Fatalf("Timed out waiting for durable retry handoff: %v", err)
	}

	// The timed worker slot must be empty at this point — the worker that fired
	// at T+5s has already completed and removed itself.
	if w := controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()); w != nil {
		t.Fatalf("Timed worker still present after initial burst: %#v", w)
	}

	// Advance the fake clock by the first backoff period (1s) to release the
	// rate-limited retry item from podEvictionQueue.
	fakeClock.Step(time.Second)

	// The retry must delete directly (attempt count goes to retries+1) without
	// creating a new 5-second timed worker. Under the bug, the retry would call
	// AddWork with a future triggerTime and no new delete attempt would happen
	// within the 200ms window below; the test would time out here.
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 200*time.Millisecond, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == int32(retries+1), nil
	}); err != nil {
		t.Fatalf("Durable retry did not delete directly after expired toleration window: %v", err)
	}

	// Confirm no new timed worker was created. Under the bug a fresh 5-second
	// worker would have been created here.
	if w := controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()); w != nil {
		t.Fatalf("Retry incorrectly created a new timed worker, granting a fresh grace period: FireAt=%v CreatedAt=%v", w.FireAt, w.CreatedAt)
	}

	// The retry item must still be active (the delete failed again).
	if current, ok := currentPodEvictionRetry(controller, podNamespacedName); !ok || current != item {
		t.Fatalf("Durable retry item changed or disappeared: ok=%v got=%#v want=%#v", ok, current, item)
	}

	// Verify the second retry also deletes directly (2s backoff step).
	fakeClock.Step(2 * time.Second)
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, 200*time.Millisecond, true, func(context.Context) (bool, error) {
		return deleteAttempts.Load() == int32(retries+2), nil
	}); err != nil {
		t.Fatalf("Second durable retry did not delete directly: %v", err)
	}
	if w := controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()); w != nil {
		t.Fatalf("Second retry created an unexpected timed worker: %#v", w)
	}
}

func TestDeletePod(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fakeClientset := fake.NewSimpleClientset()
	controller, _, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	wg.Go(func() {
		controller.Run(ctx)
	})
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	controller.PodUpdated(testutil.NewPod("pod1", "node1"), nil)
	// wait a bit to see if nothing will panic
	time.Sleep(timeForControllerToProgressForSanityCheck)
}

// TestAddConditionAndDeletePodNotFoundIsSuccess verifies that a NotFound error
// returned by the Delete API call is treated as a clean success. This covers
// the race window between our Get and Delete calls where another actor may
// concurrently delete the pod. The controller must not enter the durable retry
// queue when this happens.
func TestAddConditionAndDeletePodNotFoundIsSuccess(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)

	var deleteCalls atomic.Int32
	// Simulate concurrent deletion: the Get succeeds (pod exists in fake store)
	// but the Delete call returns NotFound as if another actor deleted it first.
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteCalls.Add(1)
		return true, nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, action.(clienttesting.DeleteAction).GetName())
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}

	wg.Go(func() {
		controller.Run(ctx)
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}

	// Trigger eviction. Pod has no tolerations so it will be deleted immediately.
	controller.PodUpdated(nil, pod)

	// Wait for the delete reactor to be called.
	if err := wait.PollUntilContextTimeout(context.Background(), 10*time.Millisecond, time.Second, true, func(context.Context) (bool, error) {
		return deleteCalls.Load() > 0, nil
	}); err != nil {
		t.Fatalf("Timed out waiting for Delete call: %v", err)
	}

	// Give the controller a moment to process the result.
	time.Sleep(timeForControllerToProgressForSanityCheck)

	// The NotFound result must be treated as success: no retry token should
	// have been registered in podEvictionTokens.
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	if _, ok := currentPodEvictionRetry(controller, podNamespacedName); ok {
		t.Errorf("NotFound on Delete incorrectly registered a durable retry token")
	}
}

func TestDeletePodHandlerStatusPatchError(t *testing.T) {
	testCases := []struct {
		name           string
		patchError     error
		wantError      bool
		wantPatchCalls int32
		wantRetry      bool
	}{
		{
			name:           "not found",
			patchError:     apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, "pod1"),
			wantPatchCalls: 1,
		},
		{
			name:           "other error",
			patchError:     apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "pod1", fmt.Errorf("status patch denied")),
			wantError:      true,
			wantPatchCalls: retries,
			wantRetry:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			pod := testutil.NewPod("pod1", "node1")
			pod.UID = "pod1-uid"
			fakeClientset := fake.NewSimpleClientset(pod)

			var patchCalls atomic.Int32
			fakeClientset.PrependReactor("patch", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				patchCalls.Add(1)
				return true, nil, tc.patchError
			})
			var deleteCalls atomic.Int32
			fakeClientset.PrependReactor("delete", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				deleteCalls.Add(1)
				return true, nil, nil
			})

			controller, _, _ := setupNewController(ctx, fakeClientset)
			controller.recorder = testutil.NewFakeRecorder()
			t.Cleanup(controller.podEvictionQueue.ShutDown)

			args := NewWorkArgsWithUID(pod.Name, pod.Namespace, pod.UID)
			args.CreatedAt = time.Now()
			err := controller.deletePodHandler()(ctx, args.CreatedAt, args)
			if (err != nil) != tc.wantError {
				t.Fatalf("deletePodHandler() error = %v, wantError = %v", err, tc.wantError)
			}
			if got := patchCalls.Load(); got != tc.wantPatchCalls {
				t.Errorf("PatchPodStatus call count = %d, want %d", got, tc.wantPatchCalls)
			}
			if got := deleteCalls.Load(); got != 0 {
				t.Errorf("Delete call count = %d, want 0", got)
			}

			podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
			_, gotRetry := currentPodEvictionRetry(controller, podNamespacedName)
			if gotRetry != tc.wantRetry {
				t.Errorf("durable retry registered = %v, want %v", gotRetry, tc.wantRetry)
			}
		})
	}
}

func TestUpdatePod(t *testing.T) {
	testCases := []struct {
		description               string
		prevPod                   *corev1.Pod
		awaitForScheduledEviction bool
		newPod                    *corev1.Pod
		taintedNodes              map[string][]corev1.Taint
		expectPatch               bool
		expectDelete              bool
		skipOnWindows             bool
	}{
		{
			description: "scheduling onto tainted Node",
			prevPod:     testutil.NewPod("pod1", ""),
			newPod:      testutil.NewPod("pod1", "node1"),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "scheduling onto tainted Node with toleration",
			prevPod:     addToleration(testutil.NewPod("pod1", ""), 1, -1),
			newPod:      addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectDelete: false,
		},
		{
			description:               "removing toleration",
			prevPod:                   addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			newPod:                    testutil.NewPod("pod1", "node1"),
			awaitForScheduledEviction: true,
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description:               "lengthening toleration shouldn't work",
			prevPod:                   addToleration(testutil.NewPod("pod1", "node1"), 1, 1),
			newPod:                    addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			awaitForScheduledEviction: true,
			taintedNodes: map[string][]corev1.Taint{
				"node1": {createNoExecuteTaint(1)},
			},
			expectPatch:   true,
			expectDelete:  true,
			skipOnWindows: true,
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			if item.skipOnWindows && goruntime.GOOS == "windows" {
				// TODO: remove skip once the flaking test has been fixed.
				t.Skip("Skip flaking test on Windows.")
			}

			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewSimpleClientset(&corev1.PodList{Items: []corev1.Pod{*item.prevPod}})
			controller, podIndexer, _ := setupNewController(context.TODO(), fakeClientset)
			controller.recorder = testutil.NewFakeRecorder()
			controller.taintedNodes = item.taintedNodes

			wg.Go(func() {
				controller.Run(ctx)
			})

			podIndexer.Add(item.prevPod)
			controller.PodUpdated(nil, item.prevPod)

			if item.awaitForScheduledEviction {
				nsName := types.NamespacedName{Namespace: item.prevPod.Namespace, Name: item.prevPod.Name}
				err := wait.PollImmediate(time.Millisecond*10, time.Second, func() (bool, error) {
					scheduledEviction := controller.taintEvictionQueue.GetWorkerUnsafe(nsName.String())
					return scheduledEviction != nil, nil
				})
				if err != nil {
					t.Fatalf("Failed to await for scheduled eviction: %q", err)
				}
			}

			podIndexer.Update(item.newPod)
			controller.PodUpdated(item.prevPod, item.newPod)

			verifyPodActions(t, item.description, fakeClientset, item.expectPatch, item.expectDelete)
			cancel()
		})
	}
}

func TestCreateNode(t *testing.T) {
	testCases := []struct {
		description  string
		pods         []corev1.Pod
		node         *corev1.Node
		expectPatch  bool
		expectDelete bool
	}{
		{
			description: "Creating Node matching already assigned Pod",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			node:         testutil.NewNode("node1"),
			expectPatch:  false,
			expectDelete: false,
		},
		{
			description: "Creating tainted Node matching already assigned Pod",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			node:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "Creating tainted Node matching already assigned tolerating Pod",
			pods: []corev1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, -1),
			},
			node:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  false,
			expectDelete: false,
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewClientset(&corev1.PodList{Items: item.pods})
			controller, _, nodeIndexer := setupNewController(ctx, fakeClientset)
			if err := nodeIndexer.Add(item.node); err != nil {
				t.Fatalf("Failed to add node %q: %v", item.node.GetName(), err)
			}
			controller.recorder = testutil.NewFakeRecorder()

			wg.Go(func() {
				controller.Run(ctx)
			})

			controller.NodeUpdated(nil, item.node)

			verifyPodActions(t, item.description, fakeClientset, item.expectPatch, item.expectDelete)

			cancel()
		})
	}
}

func TestDeleteNode(t *testing.T) {
	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fakeClientset := fake.NewSimpleClientset()
	controller, _, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}

	wg.Go(func() {
		controller.Run(ctx)
	})

	controller.NodeUpdated(testutil.NewNode("node1"), nil)

	// await until controller.taintedNodes is empty
	err := wait.PollImmediate(10*time.Millisecond, time.Second, func() (bool, error) {
		controller.taintedNodesLock.Lock()
		defer controller.taintedNodesLock.Unlock()
		_, ok := controller.taintedNodes["node1"]
		return !ok, nil
	})
	if err != nil {
		t.Errorf("Failed to await for processing node deleted: %q", err)
	}
}

func TestUpdateNode(t *testing.T) {
	testCases := []struct {
		description     string
		pods            []corev1.Pod
		oldNode         *corev1.Node
		newNode         *corev1.Node
		expectPatch     bool
		expectDelete    bool
		additionalSleep time.Duration
	}{
		{
			description: "Added taint, expect node patched and deleted",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "Added tolerated taint",
			pods: []corev1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectDelete: false,
		},
		{
			description: "Only one added taint tolerated",
			pods: []corev1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 100),
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "Taint removed",
			pods: []corev1.Pod{
				*addToleration(testutil.NewPod("pod1", "node1"), 1, 1),
			},
			oldNode:         addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			newNode:         testutil.NewNode("node1"),
			expectDelete:    false,
			additionalSleep: 1500 * time.Millisecond,
		},
		{
			description: "Pod with multiple tolerations are evicted when first one runs out",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      "pod1",
					},
					Spec: corev1.PodSpec{
						NodeName: "node1",
						Tolerations: []corev1.Toleration{
							{Key: "testTaint1", Value: "test1", Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &[]int64{1}[0]},
							{Key: "testTaint2", Value: "test2", Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &[]int64{100}[0]},
						},
					},
					Status: corev1.PodStatus{
						Conditions: []corev1.PodCondition{
							{
								Type:   corev1.PodReady,
								Status: corev1.ConditionTrue,
							},
						},
					},
				},
			},
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectPatch:  true,
			expectDelete: true,
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewSimpleClientset(&corev1.PodList{Items: item.pods})
			controller, _, nodeIndexer := setupNewController(ctx, fakeClientset)
			nodeIndexer.Add(item.newNode)
			controller.recorder = testutil.NewFakeRecorder()

			wg.Go(func() {
				controller.Run(ctx)
			})

			controller.NodeUpdated(item.oldNode, item.newNode)

			if item.additionalSleep > 0 {
				time.Sleep(item.additionalSleep)
			}

			verifyPodActions(t, item.description, fakeClientset, item.expectPatch, item.expectDelete)
		})
	}
}

func TestUpdateNodeWithMultipleTaints(t *testing.T) {
	taint1 := createNoExecuteTaint(1)
	taint2 := createNoExecuteTaint(2)

	minute := int64(60)
	pod := testutil.NewPod("pod1", "node1")
	pod.Spec.Tolerations = []corev1.Toleration{
		{Key: taint1.Key, Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoExecute},
		{Key: taint2.Key, Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoExecute, TolerationSeconds: &minute},
	}
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}

	untaintedNode := testutil.NewNode("node1")

	doubleTaintedNode := testutil.NewNode("node1")
	doubleTaintedNode.Spec.Taints = []corev1.Taint{taint1, taint2}

	singleTaintedNode := testutil.NewNode("node1")
	singleTaintedNode.Spec.Taints = []corev1.Taint{taint1}

	var wg sync.WaitGroup
	defer wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fakeClientset := fake.NewSimpleClientset(pod)
	controller, _, nodeIndexer := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()

	wg.Go(func() {
		controller.Run(ctx)
	})

	// no taint
	nodeIndexer.Add(untaintedNode)
	controller.handleNodeUpdate(ctx, nodeUpdateItem{"node1"})
	// verify pod is not queued for deletion
	if controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()) != nil {
		t.Fatalf("pod queued for deletion with no taints")
	}

	// no taint -> infinitely tolerated taint
	nodeIndexer.Update(singleTaintedNode)
	controller.handleNodeUpdate(ctx, nodeUpdateItem{"node1"})
	// verify pod is not queued for deletion
	if controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()) != nil {
		t.Fatalf("pod queued for deletion with permanently tolerated taint")
	}

	// infinitely tolerated taint -> temporarily tolerated taint
	nodeIndexer.Update(doubleTaintedNode)
	controller.handleNodeUpdate(ctx, nodeUpdateItem{"node1"})
	// verify pod is queued for deletion
	if controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()) == nil {
		t.Fatalf("pod not queued for deletion after addition of temporarily tolerated taint")
	}

	// temporarily tolerated taint -> infinitely tolerated taint
	nodeIndexer.Update(singleTaintedNode)
	controller.handleNodeUpdate(ctx, nodeUpdateItem{"node1"})
	// verify pod is not queued for deletion
	if controller.taintEvictionQueue.GetWorkerUnsafe(podNamespacedName.String()) != nil {
		t.Fatalf("pod queued for deletion after removal of temporarily tolerated taint")
	}

	// verify pod is not deleted
	for _, action := range fakeClientset.Actions() {
		if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
			t.Error("Unexpected deletion")
		}
	}
}

func TestUpdateNodeWithMultiplePods(t *testing.T) {
	testCases := []struct {
		description         string
		pods                []corev1.Pod
		oldNode             *corev1.Node
		newNode             *corev1.Node
		expectedDeleteTimes durationSlice
	}{
		{
			description: "Pods with different toleration times are evicted appropriately",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
				*addToleration(testutil.NewPod("pod2", "node1"), 1, 1),
				*addToleration(testutil.NewPod("pod3", "node1"), 1, -1),
			},
			oldNode: testutil.NewNode("node1"),
			newNode: addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectedDeleteTimes: durationSlice{
				{[]string{"pod1"}, 0},
				{[]string{"pod2"}, time.Second},
			},
		},
		{
			description: "Evict all pods not matching all taints instantly",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
				*addToleration(testutil.NewPod("pod2", "node1"), 1, 1),
				*addToleration(testutil.NewPod("pod3", "node1"), 1, -1),
			},
			oldNode: testutil.NewNode("node1"),
			newNode: addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1, 2}),
			expectedDeleteTimes: durationSlice{
				{[]string{"pod1", "pod2", "pod3"}, 0},
			},
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			t.Logf("Starting testcase %q", item.description)

			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewSimpleClientset(&corev1.PodList{Items: item.pods})
			sort.Sort(item.expectedDeleteTimes)
			controller, _, nodeIndexer := setupNewController(ctx, fakeClientset)
			nodeIndexer.Add(item.newNode)
			controller.recorder = testutil.NewFakeRecorder()

			wg.Go(func() {
				controller.Run(ctx)
			})

			controller.NodeUpdated(item.oldNode, item.newNode)

			startedAt := time.Now()
			for i := range item.expectedDeleteTimes {
				if i == 0 || item.expectedDeleteTimes[i-1].timestamp != item.expectedDeleteTimes[i].timestamp {
					// compute a grace duration to give controller time to process updates. Choose big
					// enough intervals in the test cases above to avoid flakes.
					var increment time.Duration
					if i == len(item.expectedDeleteTimes)-1 || item.expectedDeleteTimes[i+1].timestamp == item.expectedDeleteTimes[i].timestamp {
						increment = 500 * time.Millisecond
					} else {
						increment = ((item.expectedDeleteTimes[i+1].timestamp - item.expectedDeleteTimes[i].timestamp) / time.Duration(2))
					}

					sleepTime := item.expectedDeleteTimes[i].timestamp - time.Since(startedAt) + increment
					if sleepTime < 0 {
						sleepTime = 0
					}
					t.Logf("Sleeping for %v", sleepTime)
					time.Sleep(sleepTime)
				}

				for delay, podName := range item.expectedDeleteTimes[i].names {
					deleted := false
					for _, action := range fakeClientset.Actions() {
						deleteAction, ok := action.(clienttesting.DeleteActionImpl)
						if !ok {
							t.Logf("Found not-delete action with verb %v. Ignoring.", action.GetVerb())
							continue
						}
						if deleteAction.GetResource().Resource != "pods" {
							continue
						}
						if podName == deleteAction.GetName() {
							deleted = true
						}
					}
					if !deleted {
						t.Errorf("Failed to deleted pod %v after %v", podName, delay)
					}
				}
				for _, action := range fakeClientset.Actions() {
					deleteAction, ok := action.(clienttesting.DeleteActionImpl)
					if !ok {
						t.Logf("Found not-delete action with verb %v. Ignoring.", action.GetVerb())
						continue
					}
					if deleteAction.GetResource().Resource != "pods" {
						continue
					}
					deletedPodName := deleteAction.GetName()
					expected := false
					for _, podName := range item.expectedDeleteTimes[i].names {
						if podName == deletedPodName {
							expected = true
						}
					}
					if !expected {
						t.Errorf("Pod %v was deleted even though it shouldn't have", deletedPodName)
					}
				}
				fakeClientset.ClearActions()
			}
		})
	}
}

func TestGetMinTolerationTime(t *testing.T) {
	one := int64(1)
	two := int64(2)
	oneSec := 1 * time.Second

	tests := []struct {
		tolerations []corev1.Toleration
		expected    time.Duration
	}{
		{
			tolerations: []corev1.Toleration{},
			expected:    0,
		},
		{
			tolerations: []corev1.Toleration{
				{
					TolerationSeconds: nil,
				},
			},
			expected: -1,
		},
		{
			tolerations: []corev1.Toleration{
				{
					TolerationSeconds: &one,
				},
				{
					TolerationSeconds: &two,
				},
			},
			expected: oneSec,
		},

		{
			tolerations: []corev1.Toleration{
				{
					TolerationSeconds: &one,
				},
				{
					TolerationSeconds: nil,
				},
			},
			expected: oneSec,
		},
		{
			tolerations: []corev1.Toleration{
				{
					TolerationSeconds: nil,
				},
				{
					TolerationSeconds: &one,
				},
			},
			expected: oneSec,
		},
	}

	for _, test := range tests {
		got := getMinTolerationTime(test.tolerations)
		if got != test.expected {
			t.Errorf("Incorrect min toleration time: got %v, expected %v", got, test.expected)
		}
	}
}

// TestEventualConsistency verifies if getPodsAssignedToNode returns incomplete data
// (e.g. due to watch latency), it will reconcile the remaining pods eventually.
// This scenario is partially covered by TestUpdatePods, but given this is an important
// property of TaintManager, it's better to have explicit test for this.
func TestEventualConsistency(t *testing.T) {
	testCases := []struct {
		description  string
		pods         []corev1.Pod
		prevPod      *corev1.Pod
		newPod       *corev1.Pod
		oldNode      *corev1.Node
		newNode      *corev1.Node
		expectPatch  bool
		expectDelete bool
	}{
		{
			description: "existing pod2 scheduled onto tainted Node",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			prevPod:      testutil.NewPod("pod2", ""),
			newPod:       testutil.NewPod("pod2", "node1"),
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "existing pod2 with taint toleration scheduled onto tainted Node",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			prevPod:      addToleration(testutil.NewPod("pod2", ""), 1, 100),
			newPod:       addToleration(testutil.NewPod("pod2", "node1"), 1, 100),
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "new pod2 created on tainted Node",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			prevPod:      nil,
			newPod:       testutil.NewPod("pod2", "node1"),
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
		{
			description: "new pod2 with tait toleration created on tainted Node",
			pods: []corev1.Pod{
				*testutil.NewPod("pod1", "node1"),
			},
			prevPod:      nil,
			newPod:       addToleration(testutil.NewPod("pod2", "node1"), 1, 100),
			oldNode:      testutil.NewNode("node1"),
			newNode:      addTaintsToNode(testutil.NewNode("node1"), "testTaint1", "taint1", []int{1}),
			expectPatch:  true,
			expectDelete: true,
		},
	}

	for _, item := range testCases {
		t.Run(item.description, func(t *testing.T) {
			var wg sync.WaitGroup
			defer wg.Wait()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			fakeClientset := fake.NewSimpleClientset(&corev1.PodList{Items: item.pods})
			controller, podIndexer, nodeIndexer := setupNewController(ctx, fakeClientset)
			nodeIndexer.Add(item.newNode)
			controller.recorder = testutil.NewFakeRecorder()

			wg.Go(func() {
				controller.Run(ctx)
			})

			if item.prevPod != nil {
				podIndexer.Add(item.prevPod)
				controller.PodUpdated(nil, item.prevPod)
			}

			// First we simulate NodeUpdate that should delete 'pod1'. It doesn't know about 'pod2' yet.
			controller.NodeUpdated(item.oldNode, item.newNode)

			verifyPodActions(t, item.description, fakeClientset, item.expectPatch, item.expectDelete)
			fakeClientset.ClearActions()

			// And now the delayed update of 'pod2' comes to the TaintManager. We should delete it as well.
			podIndexer.Update(item.newPod)
			controller.PodUpdated(item.prevPod, item.newPod)
			// wait a bit
			time.Sleep(timeForControllerToProgressForSanityCheck)
		})
	}
}

func verifyPodActions(t *testing.T, description string, fakeClientset *fake.Clientset, expectPatch, expectDelete bool) {
	t.Helper()
	podPatched := false
	podDeleted := false
	// use Poll instead of PollImmediate to give some processing time to the controller that the expected
	// actions are likely to be already sent
	err := wait.Poll(10*time.Millisecond, 5*time.Second, func() (bool, error) {
		for _, action := range fakeClientset.Actions() {
			if action.GetVerb() == "patch" && action.GetResource().Resource == "pods" {
				podPatched = true
			}
			if action.GetVerb() == "delete" && action.GetResource().Resource == "pods" {
				podDeleted = true
			}
		}
		return podPatched == expectPatch && podDeleted == expectDelete, nil
	})
	if err != nil {
		t.Errorf("Failed waiting for the expected actions: %q", err)
	}
	if podPatched != expectPatch {
		t.Errorf("[%v]Unexpected test result. Expected patch %v, got %v", description, expectPatch, podPatched)
	}
	if podDeleted != expectDelete {
		t.Errorf("[%v]Unexpected test result. Expected delete %v, got %v", description, expectDelete, podDeleted)
	}
}

// TestPodDeletionEvent Verify that the output events are as expected
func TestPodDeletionEvent(t *testing.T) {
	f := func(path cmp.Path) bool {
		switch path.String() {
		// These fields change at runtime, so ignore it
		case "LastTimestamp", "FirstTimestamp", "ObjectMeta.Name":
			return true
		}
		return false
	}

	t.Run("emitPodDeletionEvent", func(t *testing.T) {
		controller := &Controller{}
		recorder := testutil.NewFakeRecorder()
		controller.recorder = recorder
		controller.emitPodDeletionEvent(types.NamespacedName{
			Name:      "test",
			Namespace: "test",
		})
		want := []*corev1.Event{
			{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test",
				},
				InvolvedObject: corev1.ObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Namespace:  "test",
					Name:       "test",
				},
				Reason:  "TaintManagerEviction",
				Type:    "Normal",
				Count:   1,
				Message: "Marking for deletion Pod test/test",
				Source:  corev1.EventSource{Component: "nodeControllerTest"},
			},
		}
		if diff := cmp.Diff(want, recorder.Events, cmp.FilterPath(f, cmp.Ignore())); len(diff) > 0 {
			t.Errorf("emitPodDeletionEvent() returned data (-want,+got):\n%s", diff)
		}
	})

	t.Run("emitCancelPodDeletionEvent", func(t *testing.T) {
		controller := &Controller{}
		recorder := testutil.NewFakeRecorder()
		controller.recorder = recorder
		controller.emitCancelPodDeletionEvent(types.NamespacedName{
			Name:      "test",
			Namespace: "test",
		})
		want := []*corev1.Event{
			{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test",
				},
				InvolvedObject: corev1.ObjectReference{
					Kind:       "Pod",
					APIVersion: "v1",
					Namespace:  "test",
					Name:       "test",
				},
				Reason:  "TaintManagerEviction",
				Type:    "Normal",
				Count:   1,
				Message: "Cancelling deletion of Pod test/test",
				Source:  corev1.EventSource{Component: "nodeControllerTest"},
			},
		}
		if diff := cmp.Diff(want, recorder.Events, cmp.FilterPath(f, cmp.Ignore())); len(diff) > 0 {
			t.Errorf("emitPodDeletionEvent() returned data (-want,+got):\n%s", diff)
		}
	})
}

func TestPodDeletionsLatencyObservedInSeconds(t *testing.T) {
	metrics.Register()

	// PodDeletionsLatency is a plain Histogram, which has no Reset(), and other
	// tests in this package observe it, so compare against a baseline.
	histogram := metrics.PodDeletionsLatency.ObserverMetric
	countBefore, err := metricstestutil.GetHistogramMetricCount(histogram)
	if err != nil {
		t.Fatalf("failed to get latency count: %v", err)
	}
	sumBefore, err := metricstestutil.GetHistogramMetricValue(histogram)
	if err != nil {
		t.Fatalf("failed to get latency sum: %v", err)
	}

	pod := testutil.NewPod("pod1", "node1")
	fakeClientset := fake.NewSimpleClientset(pod)
	controller, _, _ := setupNewController(context.Background(), fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	t.Cleanup(controller.podEvictionQueue.ShutDown)

	// Regression test for https://github.com/kubernetes/kubernetes/issues/142617.
	// Before the fix, the latency was computed as time.Since(fireAt) * time.Second,
	// which overflows int64 above ~9.2s, so a 10s latency was recorded as negative.
	elapsed := 10 * time.Second
	handler := controller.deletePodHandler()
	// fireAt is elapsed in the past, so the handler should observe a latency of about elapsed.
	if err := handler(context.Background(), time.Now().Add(-elapsed), NewWorkArgs(pod.Name, pod.Namespace)); err != nil {
		t.Fatalf("deletePodHandler failed: %v", err)
	}

	countAfter, err := metricstestutil.GetHistogramMetricCount(histogram)
	if err != nil {
		t.Fatalf("failed to get latency count: %v", err)
	}
	sumAfter, err := metricstestutil.GetHistogramMetricValue(histogram)
	if err != nil {
		t.Fatalf("failed to get latency sum: %v", err)
	}
	if got := countAfter - countBefore; got != 1 {
		t.Fatalf("latency sample count increased by %d, want 1", got)
	}
	// Exactly one sample was added, so the difference of the sums is that sample's value.
	if observed := sumAfter - sumBefore; observed < elapsed.Seconds() || observed > elapsed.Seconds()+1 {
		t.Errorf("observed latency = %v seconds, want between %v and %v", observed, elapsed.Seconds(), elapsed.Seconds()+1)
	}
}

// TestPodEvictionRetryRaceWithTimedWorker verifies that if the durable retry queue
// executes while the original timed worker is still in the active workers map,
// the retry correctly deletes the pod without restarting the TolerationSeconds window.
//
// Regression test for race condition where processPodEvictionRetry checked
// decision.keepExisting BEFORE item.fireAt.After(item.createdAt).
func TestPodEvictionRetryRaceWithTimedWorker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tolerationSeconds := int64(5)
	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	pod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint1",
		Value:             "test1",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteAttempts.Add(1)
		return true, nil, nil // Delete succeeds
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	taint := createNoExecuteTaint(1)
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {taint},
	}

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}

	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}

	realNow := time.Now()
	createdAt := realNow.Add(-10 * time.Second)
	fireAt := realNow.Add(-5 * time.Second) // 5s toleration, already expired 5s ago
	controller.taintEvictionQueue.clock = testingclock.NewFakeClock(createdAt)
	t.Cleanup(controller.taintEvictionQueue.CancelAndWait)

	// 1. Keep the original timed worker visible, matching the handoff interval
	//    after its callback fires but before the wrapper removes it.
	controller.taintEvictionQueue.AddWork(ctx, newPodEvictionWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace, pod.UID, pod.Spec.NodeName, taintSetKey([]corev1.Taint{taint})), createdAt, fireAt)

	// 2. The retry item (created by the timed worker) is processed.
	item := podEvictionItem{
		podRef:    NamespacedObject{NamespacedName: podNamespacedName, UID: pod.UID},
		createdAt: createdAt,
		fireAt:    fireAt,
		nodeName:  pod.Spec.NodeName,
		taintSet:  taintSetKey([]corev1.Taint{taint}),
	}

	// This direct call represents the podEvictionWorker processing the retry asynchronously
	// while the timed worker is still in the map.
	err := controller.processPodEvictionRetry(ctx, item)
	if err != nil {
		t.Fatalf("processPodEvictionRetry failed: %v", err)
	}

	if got := deleteAttempts.Load(); got != 1 {
		t.Fatalf("Expected exactly 1 delete attempt, got %d. The retry was incorrectly discarded due to the visible timed worker.", got)
	}
}

// TestPodEvictionRetryHandoffSurvivesConcurrentUpdate verifies that a pod update
// cannot cancel the durable retry while the fired timed worker is completing.
func TestPodEvictionRetryHandoffSurvivesConcurrentUpdate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tolerationSeconds := int64(5)
	pod := testutil.NewPod("pod1", "node1")
	pod.UID = "pod1-uid"
	pod.Spec.Tolerations = []corev1.Toleration{{
		Key:               "testTaint1",
		Value:             "test1",
		Effect:            corev1.TaintEffectNoExecute,
		TolerationSeconds: &tolerationSeconds,
	}}
	fakeClientset := fake.NewSimpleClientset(pod)
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		deleteAction := action.(clienttesting.DeleteAction)
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
	})

	controller, _, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()

	fakeClock := testingclock.NewFakeClock(time.Now())
	handoffReady := make(chan struct{})
	releaseTimedWorker := make(chan struct{})
	var releaseOnce sync.Once
	deleteHandler := controller.deletePodHandler()
	controller.taintEvictionQueue = CreateWorkerQueue(func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		err := deleteHandler(ctx, fireAt, args)
		close(handoffReady)
		<-releaseTimedWorker
		return err
	})
	controller.taintEvictionQueue.clock = fakeClock
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(releaseTimedWorker) })
		controller.taintEvictionQueue.CancelAndWait()
	})

	podRef := NamespacedObject{
		NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name},
		UID:            pod.UID,
	}
	taintA := createNoExecuteTaint(1)
	createdAt := fakeClock.Now()
	fireAt := createdAt.Add(time.Duration(tolerationSeconds) * time.Second)
	controller.taintEvictionQueue.AddWork(ctx, newPodEvictionWorkArgs(pod.Name, pod.Namespace, pod.UID, pod.Spec.NodeName, taintSetKey([]corev1.Taint{taintA})), createdAt, fireAt)
	fakeClock.Step(time.Duration(tolerationSeconds) * time.Second)

	select {
	case <-handoffReady:
	case <-time.After(time.Second):
		t.Fatal("timed worker did not register its durable retry")
	}
	if _, ok := currentPodEvictionRetry(controller, podRef.NamespacedName); !ok {
		t.Fatal("durable retry was not registered before timed worker completion")
	}
	if worker := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String()); worker == nil {
		t.Fatal("timed worker was removed before the handoff interleaving")
	}

	controller.processPodOnNode(ctx, podRef, pod.Spec.NodeName, pod.Spec.Tolerations, []corev1.Taint{taintA}, fakeClock.Now())
	if _, ok := currentPodEvictionRetry(controller, podRef.NamespacedName); !ok {
		t.Error("concurrent pod update canceled the durable retry during timed-worker handoff")
	}

	releaseOnce.Do(func() { close(releaseTimedWorker) })
	controller.taintEvictionQueue.workerWG.Wait()
	if worker := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String()); worker != nil {
		t.Fatalf("completed timed worker was not removed: %#v", worker)
	}
	if _, ok := currentPodEvictionRetry(controller, podRef.NamespacedName); !ok {
		t.Error("pod requiring eviction has neither a timed worker nor a durable retry")
	}
}

func TestPodEvictionRetryReconcilesConcurrentPodUpdate(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pod := addToleration(testutil.NewPod("pod1", "node1"), 1, -1)
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(1)},
	}
	t.Cleanup(controller.podEvictionQueue.ShutDown)

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}

	captured := make(chan struct{})
	release := make(chan struct{})
	controller.podLister = &blockingPodLister{
		PodLister: controller.podLister,
		captured:  captured,
		release:   release,
	}

	podRef := NamespacedObject{
		NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name},
		UID:            pod.UID,
	}
	item, added := controller.addPodEvictionRetry(podRef, time.Now(), time.Now(), pod.Spec.NodeName, taintSetKey(controller.taintedNodes[pod.Spec.NodeName]))
	if !added {
		t.Fatal("Failed to register durable retry")
	}
	generation, matches := controller.podEvictionRetryGeneration(item)
	if !matches {
		t.Fatal("Durable retry was not current before processing")
	}

	retryDone := make(chan error, 1)
	go func() {
		retryDone <- controller.processPodEvictionRetry(ctx, item)
	}()
	<-captured

	// The retry has captured the indefinitely-tolerating pod. Before it can
	// relinquish ownership, process an update for the same UID that no longer
	// tolerates the taint.
	updatedPod := pod.DeepCopy()
	updatedPod.Spec.Tolerations = nil
	controller.processPodOnNode(ctx, podRef, updatedPod.Spec.NodeName, updatedPod.Spec.Tolerations, controller.taintedNodes[updatedPod.Spec.NodeName], time.Now())

	close(release)
	if err := <-retryDone; err != nil {
		t.Fatalf("processPodEvictionRetry failed: %v", err)
	}
	controller.forgetPodEvictionRetry(item, generation)

	_, retryExists := currentPodEvictionRetry(controller, podRef.NamespacedName)
	worker := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String())
	if !retryExists && worker == nil {
		t.Fatal("concurrent update relied on a retry that relinquished ownership")
	}
}

func TestExpiredPodEvictionRetryDoesNotApplyDeadlineToReplacementTaint(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tolerationSeconds := int64(5)
	pod := addToleration(testutil.NewPod("pod1", "node1"), 1, tolerationSeconds)
	pod.UID = "pod1-uid"
	fakeClientset := fake.NewSimpleClientset(pod)
	var deleteAttempts atomic.Int32
	burstDone := make(chan struct{})
	fakeClientset.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		attempt := deleteAttempts.Add(1)
		if attempt <= retries {
			if attempt == retries {
				close(burstDone)
			}
			deleteAction := action.(clienttesting.DeleteAction)
			return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, deleteAction.GetName(), fmt.Errorf("denied by test"))
		}
		return false, nil, nil
	})

	controller, podIndexer, _ := setupNewController(ctx, fakeClientset)
	controller.recorder = testutil.NewFakeRecorder()
	taintA := createNoExecuteTaint(1)
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {taintA},
	}
	fakeClock := testingclock.NewFakeClock(time.Now())
	useFakePodEvictionQueue(controller, fakeClock)
	t.Cleanup(func() {
		controller.podEvictionQueue.ShutDown()
		controller.taintEvictionQueue.CancelAndWait()
	})

	if err := podIndexer.Add(pod); err != nil {
		t.Fatalf("Failed to add pod to indexer: %v", err)
	}
	podRef := NamespacedObject{
		NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name},
		UID:            pod.UID,
	}
	createdAt := fakeClock.Now()
	fireAt := createdAt.Add(time.Duration(tolerationSeconds) * time.Second)
	controller.taintEvictionQueue.AddWork(ctx, newPodEvictionWorkArgs(pod.Name, pod.Namespace, pod.UID, pod.Spec.NodeName, taintSetKey([]corev1.Taint{taintA})), createdAt, fireAt)
	fakeClock.Step(time.Duration(tolerationSeconds) * time.Second)
	<-burstDone
	controller.taintEvictionQueue.workerWG.Wait()

	item, ok := currentPodEvictionRetry(controller, podRef.NamespacedName)
	if !ok {
		t.Fatal("Expired finite timer did not create a durable retry")
	}

	replacementPod := addToleration(pod.DeepCopy(), 2, tolerationSeconds)
	if _, err := fakeClientset.CoreV1().Pods(replacementPod.Namespace).Update(ctx, replacementPod, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update pod in fake client: %v", err)
	}
	if err := podIndexer.Update(replacementPod); err != nil {
		t.Fatalf("Failed to update pod in indexer: %v", err)
	}
	controller.taintedNodes = map[string][]corev1.Taint{
		"node1": {createNoExecuteTaint(2)},
	}

	if err := controller.processPodEvictionRetry(ctx, item); err != nil {
		t.Fatalf("processPodEvictionRetry failed: %v", err)
	}
	if got := deleteAttempts.Load(); got != retries {
		t.Fatalf("old deadline triggered deletion for replacement taint: got %d attempts, want %d", got, retries)
	}
	worker := controller.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String())
	if worker == nil {
		t.Fatal("replacement taint did not receive a fresh finite deadline")
	}
	if got, want := worker.FireAt.Sub(worker.CreatedAt), time.Duration(tolerationSeconds)*time.Second; got != want {
		t.Fatalf("replacement taint deadline = %v, want %v", got, want)
	}
}

func TestStaleUIDTimedWorkerCannotWinAddWorkRace(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	controller, _, _ := setupNewController(ctx, fake.NewSimpleClientset())
	controller.recorder = testutil.NewFakeRecorder()
	t.Cleanup(controller.podEvictionQueue.ShutDown)
	t.Cleanup(controller.taintEvictionQueue.CancelAndWait)

	now := time.Now()
	triggerTime := now.Add(time.Hour)
	newPodRef := NamespacedObject{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		UID:            "new-uid",
	}
	decision := controller.getPodEvictionDecision(
		klog.Background(),
		newPodRef,
		"node1",
		addToleration(testutil.NewPod("pod1", "node1"), 1, 3600).Spec.Tolerations,
		[]corev1.Taint{createNoExecuteTaint(1)},
		now,
	)
	if decision.kind != podEvictionLater {
		t.Fatalf("New pod decision = %v, want podEvictionLater", decision.kind)
	}

	// The old retry installs its worker after the new UID computed a decision
	// but before that decision is applied to the timed queue.
	controller.taintEvictionQueue.AddWork(ctx, NewWorkArgsWithUID("pod1", "default", "old-uid"), now, triggerTime)
	controller.taintEvictionQueue.UpdateWork(ctx, newPodEvictionWorkArgs("pod1", "default", newPodRef.UID, "node1", decision.taintSet), decision.startTime, decision.triggerTime)

	worker := controller.taintEvictionQueue.GetWorkerUnsafe(newPodRef.NamespacedName.String())
	if worker == nil {
		t.Fatal("Expected a finite timed worker")
	}
	if got := worker.WorkItem.Object.UID; got != newPodRef.UID {
		t.Fatalf("stale UID worker retained ownership: got UID %q, want %q", got, newPodRef.UID)
	}
}

func TestPodEvictionRetryRelinquishesObsoleteOwnership(t *testing.T) {
	testCases := []struct {
		name       string
		prepare    func(*corev1.Pod)
		podMissing bool
	}{
		{
			name:       "pod not found",
			podMissing: true,
		},
		{
			name: "pod terminating",
			prepare: func(pod *corev1.Pod) {
				now := metav1.Now()
				pod.DeletionTimestamp = &now
			},
		},
		{
			name: "node name removed",
			prepare: func(pod *corev1.Pod) {
				pod.Spec.NodeName = ""
			},
		},
		{
			name: "infinite toleration",
			prepare: func(pod *corev1.Pod) {
				addToleration(pod, 1, -1)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			pod := testutil.NewPod("pod1", "node1")
			pod.UID = "pod1-uid"
			if tc.prepare != nil {
				tc.prepare(pod)
			}
			controller, podIndexer, _ := setupNewController(ctx, fake.NewSimpleClientset(pod))
			controller.recorder = testutil.NewFakeRecorder()
			taints := []corev1.Taint{createNoExecuteTaint(1)}
			controller.taintedNodes = map[string][]corev1.Taint{"node1": taints}
			t.Cleanup(controller.podEvictionQueue.ShutDown)
			if !tc.podMissing {
				if err := podIndexer.Add(pod); err != nil {
					t.Fatalf("Failed to add pod to indexer: %v", err)
				}
			}

			podRef := NamespacedObject{
				NamespacedName: types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name},
				UID:            pod.UID,
			}
			item, added := controller.addPodEvictionRetry(podRef, time.Now(), time.Now(), "node1", taintSetKey(taints))
			if !added {
				t.Fatal("Failed to register durable retry")
			}
			generation, matches := controller.podEvictionRetryGeneration(item)
			if !matches {
				t.Fatal("Durable retry was not current before processing")
			}

			if err := controller.processPodEvictionRetry(ctx, item); err != nil {
				t.Fatalf("processPodEvictionRetry failed: %v", err)
			}
			if !controller.forgetPodEvictionRetry(item, generation) {
				t.Fatal("Obsolete retry did not relinquish ownership")
			}
			if _, ok := currentPodEvictionRetry(controller, podRef.NamespacedName); ok {
				t.Fatal("Obsolete durable retry remained registered")
			}
		})
	}
}
