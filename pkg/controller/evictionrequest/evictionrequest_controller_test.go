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

package evictionrequest

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2/ktesting"
	testingclock "k8s.io/utils/clock/testing"
)

const testControllerName = "evictionrequest-controller"

var alwaysReady = func() bool { return true }

// newTestController creates a controller for testing with the given objects.
func newTestController(t *testing.T, evictionRequests []*coordinationv1alpha1.EvictionRequest, pods []*v1.Pod, fakeClock *testingclock.FakeClock) (*EvictionRequestController, informers.SharedInformerFactory, *fake.Clientset) {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)

	// Build object list for fake client
	var objects []runtime.Object
	for _, er := range evictionRequests {
		objects = append(objects, er)
	}
	for _, pod := range pods {
		objects = append(objects, pod)
	}

	client := fake.NewClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	c, err := NewController(
		ctx,
		informerFactory.Coordination().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		client,
		testControllerName,
	)
	if err != nil {
		t.Fatalf("unexpected error creating controller: %v", err)
	}

	if fakeClock == nil {
		fakeClock = testingclock.NewFakeClock(time.Now())
	}
	c.clock = fakeClock
	// Recreate queues with the fake clock so AddAfter uses it
	c.queue = workqueue.NewTypedRateLimitingQueueWithConfig(
		workqueue.DefaultTypedControllerRateLimiter[string](),
		workqueue.TypedRateLimitingQueueConfig[string]{
			Name:  testControllerName,
			Clock: fakeClock,
		},
	)

	// Mark informers as synced
	c.evictionRequestListerSynced = alwaysReady
	c.podListerSynced = alwaysReady

	// Add objects to informer indexers
	for _, er := range evictionRequests {
		if err := informerFactory.Coordination().V1alpha1().EvictionRequests().Informer().GetIndexer().Add(er); err != nil {
			t.Fatalf("failed to add eviction request to indexer: %v", err)
		}
	}
	for _, pod := range pods {
		if err := informerFactory.Core().V1().Pods().Informer().GetIndexer().Add(pod); err != nil {
			t.Fatalf("failed to add pod to indexer: %v", err)
		}
	}

	return c, informerFactory, client
}

// newPod creates a test pod with the given name and UID.
func newPod(namespace, name, uid string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			UID:       types.UID(uid),
			Labels:    map[string]string{"app": "test"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "test", Image: "test"}},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

// newEvictionRequest creates a test EvictionRequest targeting the given pod.
func newEvictionRequest(namespace, podName, podUID string) *coordinationv1alpha1.EvictionRequest {
	return &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      podUID, // EvictionRequest name should match pod UID
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target: coordinationv1alpha1.EvictionTarget{
				Pod: &coordinationv1alpha1.LocalTargetReference{
					Name: podName,
					UID:  podUID,
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}
}

func hasCanceledCondition(er *coordinationv1alpha1.EvictionRequest) bool {
	return meta.IsStatusConditionTrue(er.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionCanceled))
}

func hasEvictedCondition(er *coordinationv1alpha1.EvictionRequest) bool {
	return meta.IsStatusConditionTrue(er.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted))
}

// getLastStatusPatch finds the last patch action targeting the status subresource
// and unmarshals it into an EvictionRequest. This is needed because the fake client
// doesn't implement SSA field ownership tracking, so fields absent from the apply
// config (e.g. cleared ActiveInterceptors) aren't removed from the stored object.
func getLastStatusPatch(t *testing.T, client *fake.Clientset) *coordinationv1alpha1.EvictionRequest {
	t.Helper()
	for i := len(client.Actions()) - 1; i >= 0; i-- {
		action := client.Actions()[i]
		if action.GetVerb() == "patch" && action.GetSubresource() == "status" {
			patchAction := action.(core.PatchAction)
			er := &coordinationv1alpha1.EvictionRequest{}
			if err := json.Unmarshal(patchAction.GetPatch(), er); err != nil {
				t.Fatalf("failed to unmarshal status patch: %v", err)
			}
			return er
		}
	}
	t.Fatal("no status patch action found")
	return nil
}

// errorPodLister is a PodLister that always returns an error.
type errorPodLister struct {
	err error
}

func (l *errorPodLister) List(_ labels.Selector) ([]*v1.Pod, error)  { return nil, l.err }
func (l *errorPodLister) Pods(string) corelisters.PodNamespaceLister { return l }
func (l *errorPodLister) Get(string) (*v1.Pod, error)                { return nil, l.err }

func TestNewController(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	c, err := NewController(
		ctx,
		informerFactory.Coordination().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		client,
		testControllerName,
	)
	if err != nil {
		t.Fatalf("unexpected error creating controller: %v", err)
	}

	if c == nil {
		t.Fatal("expected controller to be created")
	}
	if c.queue == nil {
		t.Error("expected queue to be initialized")
	}
	if c.labelSyncQueue == nil {
		t.Error("expected labelSyncQueue to be initialized")
	}
}

func TestSyncHandler_ValidationPodNotFound(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create EvictionRequest targeting a pod that doesn't exist
	er := newEvictionRequest("default", "missing-pod", "missing-uid")
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, nil, nil)

	// Sync
	err := c.syncHandler(ctx, "default/missing-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Canceled condition
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "missing-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasCanceledCondition(updated) {
		t.Error("expected Canceled condition to be set")
	}
}

func TestSyncHandler_PodListerTransientError(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	er := newEvictionRequest("default", "test-pod", "test-uid")
	c, _, _ := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, nil, nil)

	// Replace pod lister with one that returns a transient error
	c.podLister = &errorPodLister{err: fmt.Errorf("connection refused")}

	err := c.syncHandler(ctx, "default/test-uid")
	if err == nil {
		t.Fatal("expected error to be returned for transient pod lister failure")
	}
}

func TestSyncHandler_ValidationUIDMismatch(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create pod with one UID
	pod := newPod("default", "test-pod", "actual-uid")

	// Create EvictionRequest targeting pod but with wrong UID
	er := newEvictionRequest("default", "test-pod", "wrong-uid")
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/wrong-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Canceled condition
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "wrong-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasCanceledCondition(updated) {
		t.Error("expected Canceled condition to be set")
	}
}

func TestSyncHandler_ValidationWorkloadRef(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create pod with WorkloadRef set
	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.WorkloadRef = &v1.WorkloadReference{Name: "my-workload"}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Canceled condition
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasCanceledCondition(updated) {
		t.Error("expected Canceled condition to be set for WorkloadRef pod")
	}
}

func TestSyncHandler_ValidationEmptyRequesters(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Spec.Requesters = nil // Empty requesters

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Canceled condition
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasCanceledCondition(updated) {
		t.Error("expected Canceled condition to be set for empty requesters")
	}
}

func TestSyncHandler_PodTerminal_Evicted(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create pod in terminal phase
	pod := newPod("default", "test-pod", "test-uid")
	pod.Status.Phase = v1.PodSucceeded

	er := newEvictionRequest("default", "test-pod", "test-uid")
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Evicted condition
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasEvictedCondition(updated) {
		t.Error("expected Evicted condition to be set for terminal pod")
	}
}

func TestSyncHandler_PodDeleted_Evicted(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create EvictionRequest with TargetInterceptors already set (means validation passed before)
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = 1 // Simulate that we've already observed this request
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: ImperativeEvictionInterceptor},
	}

	// Pod doesn't exist (deleted)
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, nil, nil)

	// Sync
	err := c.syncHandler(ctx, "default/deleted-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify status was updated with Evicted condition (not Canceled)
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "deleted-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasEvictedCondition(updated) {
		t.Error("expected Evicted condition to be set when pod deleted after validation")
	}
	if hasCanceledCondition(updated) {
		t.Error("should not have Canceled condition when pod deleted after validation")
	}
}

func TestSyncHandler_PodNotFound_MovesActiveInterceptorToProcessed(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testInterceptor := "cleanup-interceptor.example.com"
	now := metav1.Now()

	// Create EvictionRequest with an active interceptor that has a heartbeat
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = 1
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: testInterceptor},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{testInterceptor}
	// Set interceptor status with heartbeat so it doesn't auto-advance
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:          testInterceptor,
			HeartbeatTime: &now,
		},
	}

	// Pod doesn't exist (deleted while interceptor was active)
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, nil, nil)

	// Sync
	err := c.syncHandler(ctx, "default/deleted-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the status patch content directly since the fake client doesn't
	// implement SSA field ownership (fields absent from apply config aren't cleared).
	updated := getLastStatusPatch(t, client)

	// Should be Evicted
	if !hasEvictedCondition(updated) {
		t.Error("expected Evicted condition to be set when pod deleted")
	}

	// Active interceptors should be cleared (not present in apply config)
	if len(updated.Status.ActiveInterceptors) != 0 {
		t.Errorf("expected active interceptors to be cleared, got %v", updated.Status.ActiveInterceptors)
	}

	// The active interceptor should be moved to processed
	if len(updated.Status.ProcessedInterceptors) != 1 {
		t.Fatalf("expected 1 processed interceptor, got %d", len(updated.Status.ProcessedInterceptors))
	}
	if updated.Status.ProcessedInterceptors[0] != testInterceptor {
		t.Errorf("expected %s in processed, got %s", testInterceptor, updated.Status.ProcessedInterceptors[0])
	}

	// The interceptor status SHOULD still exist with heartbeat but NO CompletionTime
	var interceptorStatus *coordinationv1alpha1.InterceptorStatus
	for i := range updated.Status.Interceptors {
		if updated.Status.Interceptors[i].Name == testInterceptor {
			interceptorStatus = &updated.Status.Interceptors[i]
			break
		}
	}
	if interceptorStatus == nil {
		t.Fatalf("expected interceptor status to exist")
	}
	if interceptorStatus.CompletionTime != nil {
		t.Error("expected interceptor to NOT have CompletionTime (it was interrupted)")
	}
	if interceptorStatus.HeartbeatTime == nil {
		t.Error("expected interceptor to still have HeartbeatTime")
	}
}

func TestSyncHandler_PodTerminal_DefersCompletionForActiveInterceptor(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testInterceptor := "cleanup-interceptor.example.com"

	now := time.Now()
	fakeClock := testingclock.NewFakeClock(now)

	// Pod is terminal (Succeeded) with DeletionTimestamp set 1s ago
	pod := newPod("default", "deleted-pod", "deleted-uid")
	deletionTime := metav1.NewTime(now.Add(-1 * time.Second))
	pod.DeletionTimestamp = &deletionTime
	pod.Status.Phase = v1.PodSucceeded

	// Create EvictionRequest with an active interceptor that hasn't completed
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = 1
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: testInterceptor},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{testInterceptor}
	heartbeatTime := metav1.NewTime(now.Add(-10 * time.Second))
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:          testInterceptor,
			HeartbeatTime: &heartbeatTime,
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// First sync: should defer completion because DeletionTimestamp is only 1s ago
	err := c.syncHandler(ctx, "default/deleted-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "deleted-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	// Should NOT be Evicted yet as we're giving the interceptor grace time to complete
	if hasEvictedCondition(updated) {
		t.Error("expected Evicted condition to NOT be set yet (grace period not elapsed)")
	}

	// Active interceptor should still be active
	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Errorf("expected 1 active interceptor during grace period, got %d", len(updated.Status.ActiveInterceptors))
	}

	// Advance clock past the grace period — this makes the delayed resync item ready
	fakeClock.Step(GracefulCompletionDelay)

	// Poll for the deferred resync to appear in the queue
	if err := wait.PollUntilContextTimeout(ctx, 1*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		return c.queue.Len() == 1, nil
	}); err != nil {
		t.Errorf("expected 1 item in queue for deferred resync, got %d", c.queue.Len())
	}

	// Second sync after grace period elapsed
	client.ClearActions()
	err = c.syncHandler(ctx, "default/deleted-uid")
	if err != nil {
		t.Fatalf("unexpected error on second sync: %v", err)
	}

	// Use getLastStatusPatch because the fake client doesn't implement SSA field
	// ownership (cleared ActiveInterceptors won't be reflected via Get).
	updated = getLastStatusPatch(t, client)

	// Now it should be Evicted
	if !hasEvictedCondition(updated) {
		t.Error("expected Evicted condition after grace period elapsed")
	}

	// Active should be cleared and the interceptor should be moved to processed
	if len(updated.Status.ActiveInterceptors) != 0 {
		t.Errorf("expected active interceptors to be cleared, got %v", updated.Status.ActiveInterceptors)
	}
	if len(updated.Status.ProcessedInterceptors) != 1 || updated.Status.ProcessedInterceptors[0] != testInterceptor {
		t.Errorf("expected %s in processed, got %s", testInterceptor, updated.Status.ProcessedInterceptors[0])
	}
}

func TestSyncHandler_InitializeTargetInterceptors(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: "custom-interceptor.example.com"},
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify TargetInterceptors initialized
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.TargetInterceptors) != 2 {
		t.Errorf("expected 2 target interceptors, got %d", len(updated.Status.TargetInterceptors))
	}

	if updated.Status.TargetInterceptors[0].Name != "custom-interceptor.example.com" {
		t.Errorf("expected first interceptor to be custom, got %s", updated.Status.TargetInterceptors[0].Name)
	}

	if updated.Status.TargetInterceptors[1].Name != ImperativeEvictionInterceptor {
		t.Errorf("expected last interceptor to be imperative, got %s", updated.Status.TargetInterceptors[1].Name)
	}

	// Verify interceptor statuses were initialized for ALL target interceptors
	if len(updated.Status.Interceptors) != 2 {
		t.Fatalf("expected 2 interceptor statuses, got %d", len(updated.Status.Interceptors))
	}

	// Active interceptor (first one) should have StartTime set
	customStatus := findInterceptorStatus(updated.Status.Interceptors, "custom-interceptor.example.com")
	if customStatus == nil {
		t.Fatal("expected interceptor status for custom interceptor")
	}
	if customStatus.StartTime == nil {
		t.Error("expected active interceptor to have start time initialized")
	}
	if customStatus.HeartbeatTime != nil {
		t.Error("expected active interceptor to not have heartbeat time set by controller")
	}

	// Non-active interceptor should have status entry but no StartTime
	imperativeStatus := findInterceptorStatus(updated.Status.Interceptors, ImperativeEvictionInterceptor)
	if imperativeStatus == nil {
		t.Fatal("expected interceptor status for imperative interceptor")
	}
	if imperativeStatus.StartTime != nil {
		t.Error("expected non-active interceptor to not have start time set")
	}
}

func TestSyncHandler_SelectFirstInterceptor(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	er := newEvictionRequest("default", "test-pod", "test-uid")

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify ActiveInterceptors set to first interceptor
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Errorf("expected 1 active interceptor, got %d", len(updated.Status.ActiveInterceptors))
	}

	// Since pod has no EvictionInterceptors, first should be imperative
	if updated.Status.ActiveInterceptors[0] != ImperativeEvictionInterceptor {
		t.Errorf("expected active interceptor to be imperative, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify interceptor status was initialized
	if len(updated.Status.Interceptors) != 1 {
		t.Fatalf("expected 1 interceptor status, got %d", len(updated.Status.Interceptors))
	}

	interceptorStatus := updated.Status.Interceptors[0]
	if interceptorStatus.Name != ImperativeEvictionInterceptor {
		t.Errorf("expected interceptor status name to be %s, got %s",
			ImperativeEvictionInterceptor, interceptorStatus.Name)
	}

	if interceptorStatus.StartTime == nil {
		t.Error("expected interceptor status to have start time initialized")
	}

	if interceptorStatus.HeartbeatTime != nil {
		t.Error("expected interceptor status to not have heartbeat time set")
	}
}

func TestSyncHandler_AdvanceOnCompletion(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testEvictionInterceptorClass := "first-interceptor.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: testEvictionInterceptorClass},
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	// Set up status as if first interceptor is active and completed
	now := metav1.Now()
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: testEvictionInterceptorClass},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{testEvictionInterceptorClass}
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:           testEvictionInterceptorClass,
			CompletionTime: &now, // Completed
			Message:        "done",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify advanced to next interceptor
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Fatalf("expected 1 active interceptor, got %d", len(updated.Status.ActiveInterceptors))
	}

	if updated.Status.ActiveInterceptors[0] != ImperativeEvictionInterceptor {
		t.Errorf("expected active interceptor to advance to imperative, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify first interceptor is in ProcessedInterceptors
	if len(updated.Status.ProcessedInterceptors) != 1 {
		t.Fatalf("expected 1 processed interceptor, got %d", len(updated.Status.ProcessedInterceptors))
	}
	if updated.Status.ProcessedInterceptors[0] != testEvictionInterceptorClass {
		t.Errorf("expected first interceptor in processed, got %s", updated.Status.ProcessedInterceptors[0])
	}

	// Verify newly active interceptor (imperative) has StartTime set
	imperativeStatus := findInterceptorStatus(updated.Status.Interceptors, ImperativeEvictionInterceptor)
	if imperativeStatus == nil {
		t.Fatal("expected interceptor status for newly active imperative interceptor")
	}
	if imperativeStatus.StartTime == nil {
		t.Error("expected newly active interceptor to have start time set")
	}
	if imperativeStatus.HeartbeatTime != nil {
		t.Error("expected newly active interceptor to not have heartbeat time set by controller")
	}

	// Verify completed interceptor status is preserved
	firstStatus := findInterceptorStatus(updated.Status.Interceptors, testEvictionInterceptorClass)
	if firstStatus == nil {
		t.Fatal("expected interceptor status for completed interceptor to be preserved")
	}
	if firstStatus.CompletionTime == nil {
		t.Error("expected completed interceptor to still have completion time")
	}
}

func TestSyncHandler_AdvanceOnTimeout(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	slowEvictionInterceptorClass := "slow-interceptor.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: slowEvictionInterceptorClass},
	}

	// Create fake clock first so we can set heartbeat time before creating controller
	fakeClock := testingclock.NewFakeClock(time.Now())
	heartbeatTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: slowEvictionInterceptorClass},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{slowEvictionInterceptorClass}
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:          slowEvictionInterceptorClass,
			HeartbeatTime: &heartbeatTime,
			Message:       "still working",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock past the timeout
	fakeClock.Step(2 * InterceptorHeartbeatTimeout)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify advanced to next interceptor due to timeout
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Fatalf("expected 1 active interceptor, got %d", len(updated.Status.ActiveInterceptors))
	}

	if updated.Status.ActiveInterceptors[0] != ImperativeEvictionInterceptor {
		t.Errorf("expected active interceptor to advance to imperative after timeout, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify slow interceptor is in ProcessedInterceptors
	if len(updated.Status.ProcessedInterceptors) != 1 {
		t.Fatalf("expected 1 processed interceptor, got %d", len(updated.Status.ProcessedInterceptors))
	}
}

func TestSyncHandler_AdvanceOnTimeoutWithoutHeartbeat(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	staleInterceptor := "stale-interceptor.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: staleInterceptor},
	}

	fakeClock := testingclock.NewFakeClock(time.Now())
	startTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: staleInterceptor},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{staleInterceptor}
	// Interceptor has StartTime but never heartbeated
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:      staleInterceptor,
			StartTime: &startTime,
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock past the timeout
	fakeClock.Step(2 * InterceptorHeartbeatTimeout)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify advanced to next interceptor due to StartTime-based timeout
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Fatalf("expected 1 active interceptor, got %d", len(updated.Status.ActiveInterceptors))
	}

	if updated.Status.ActiveInterceptors[0] != ImperativeEvictionInterceptor {
		t.Errorf("expected active interceptor to advance to imperative after StartTime timeout, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify stale interceptor is in ProcessedInterceptors
	if len(updated.Status.ProcessedInterceptors) != 1 {
		t.Fatalf("expected 1 processed interceptor, got %d", len(updated.Status.ProcessedInterceptors))
	}
}

func TestSyncHandler_NoAdvanceBeforeTimeout(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	activeEvictionInterceptorClass := "active-interceptor.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionInterceptors = []v1.EvictionInterceptor{
		{Name: activeEvictionInterceptorClass},
	}

	// Create fake clock first so we can set heartbeat time before creating controller
	fakeClock := testingclock.NewFakeClock(time.Now())
	heartbeatTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: activeEvictionInterceptorClass},
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{activeEvictionInterceptorClass}
	er.Status.ObservedGeneration = er.Generation
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:          activeEvictionInterceptorClass,
			HeartbeatTime: &heartbeatTime,
			Message:       "working",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock but NOT past the timeout
	fakeClock.Step(InterceptorHeartbeatTimeout / 4)

	// Clear actions
	client.ClearActions()

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify interceptor was NOT advanced (still the same)
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.ActiveInterceptors) != 1 {
		t.Fatalf("expected 1 active interceptor, got %d", len(updated.Status.ActiveInterceptors))
	}

	if updated.Status.ActiveInterceptors[0] != activeEvictionInterceptorClass {
		t.Errorf("expected active interceptor to remain unchanged before timeout, got %s", updated.Status.ActiveInterceptors[0])
	}

	// Verify ProcessedInterceptors is empty
	if len(updated.Status.ProcessedInterceptors) != 0 {
		t.Fatalf("expected no processed interceptors before timeout, got %d", len(updated.Status.ProcessedInterceptors))
	}
}

func TestSyncHandler_AllInterceptorsProcessed(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")

	er := newEvictionRequest("default", "test-pod", "test-uid")
	// Set up status where all interceptors are processed
	now := metav1.Now()
	er.Status.TargetInterceptors = []v1.EvictionInterceptor{
		{Name: ImperativeEvictionInterceptor},
	}
	er.Status.ActiveInterceptors = []string{ImperativeEvictionInterceptor}
	er.Status.Interceptors = []coordinationv1alpha1.InterceptorStatus{
		{
			Name:           ImperativeEvictionInterceptor,
			CompletionTime: &now,
			Message:        "done",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Use getLastStatusPatch because the fake client doesn't implement SSA field
	// ownership (cleared ActiveInterceptors won't be reflected via Get).
	updated := getLastStatusPatch(t, client)

	if len(updated.Status.ActiveInterceptors) != 0 {
		t.Errorf("expected no active interceptors when all processed, got %d", len(updated.Status.ActiveInterceptors))
	}

	// Verify imperative interceptor is in ProcessedInterceptors
	if len(updated.Status.ProcessedInterceptors) != 1 {
		t.Errorf("expected 1 processed interceptor, got %d", len(updated.Status.ProcessedInterceptors))
	}
}

func TestSyncHandler_ObservedGenerationSet(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Generation = 5

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify ObservedGeneration was set
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if updated.Status.ObservedGeneration != 5 {
		t.Errorf("expected ObservedGeneration to be 5, got %d", updated.Status.ObservedGeneration)
	}
}

func TestSyncLabelHandler_SyncsLabels(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	pod.Labels = map[string]string{
		"app":     "myapp",
		"version": "v1",
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Labels = map[string]string{
		"existing": "label",
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync labels
	err := c.syncLabelHandler(ctx, "default/test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify labels were synced
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	// Should have both existing and pod labels
	if updated.Labels["existing"] != "label" {
		t.Error("expected existing label to be preserved")
	}
	if updated.Labels["app"] != "myapp" {
		t.Error("expected pod label 'app' to be synced")
	}
	if updated.Labels["version"] != "v1" {
		t.Error("expected pod label 'version' to be synced")
	}
}

func TestSyncLabelHandler_PodLabelsOverwrite(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	pod.Labels = map[string]string{
		"app": "pod-value", // This should overwrite
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Labels = map[string]string{
		"app": "er-value", // This should be overwritten
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync labels
	err := c.syncLabelHandler(ctx, "default/test-pod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify pod label overwrote EvictionRequest label
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if updated.Labels["app"] != "pod-value" {
		t.Errorf("expected pod label to overwrite, got %s", updated.Labels["app"])
	}
}

func TestWatchEvictionRequests(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()

	fakeWatch := watch.NewFakeWithOptions(watch.FakeOptions{Logger: &logger})
	client.PrependWatchReactor("evictionrequests", core.DefaultWatchReactor(fakeWatch, nil))

	informerFactory := informers.NewSharedInformerFactory(client, 0)

	c, err := NewController(
		ctx,
		informerFactory.Coordination().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		client,
		testControllerName,
	)
	if err != nil {
		t.Fatalf("unexpected error creating controller: %v", err)
	}

	// Bypass cache sync check
	c.evictionRequestListerSynced = alwaysReady
	c.podListerSynced = alwaysReady

	received := make(chan string, 1)

	// Replace syncHandler to capture what gets synced
	c.syncHandler = func(ctx context.Context, key string) error {
		received <- key
		return nil
	}

	// Start only the EvictionRequest informer and controller
	go informerFactory.Coordination().V1alpha1().EvictionRequests().Informer().RunWithContext(ctx)
	go c.Run(ctx, 1)

	// Create an EvictionRequest and send it through the fake watch
	er := newEvictionRequest("default", "test-pod", "test-uid")
	fakeWatch.Add(er)

	// Verify the syncHandler received the key
	select {
	case key := <-received:
		expectedKey := "default/test-uid"
		if key != expectedKey {
			t.Errorf("expected key %s, got %s", expectedKey, key)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for sync")
	}
}

func TestPodWatch_TriggersLabelSyncAndMainSync(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()

	fakeWatch := watch.NewFakeWithOptions(watch.FakeOptions{Logger: &logger})
	client.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))

	informerFactory := informers.NewSharedInformerFactory(client, 0)

	c, err := NewController(
		ctx,
		informerFactory.Coordination().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		client,
		testControllerName,
	)
	if err != nil {
		t.Fatalf("unexpected error creating controller: %v", err)
	}

	// Bypass cache sync check
	c.evictionRequestListerSynced = alwaysReady
	c.podListerSynced = alwaysReady

	// Add an EvictionRequest to the indexer so enqueueLabelSyncForPod can find it
	er := newEvictionRequest("default", "test-pod", "test-uid")
	informerFactory.Coordination().V1alpha1().EvictionRequests().Informer().GetIndexer().Add(er)

	labelSyncReceived := make(chan string, 1)
	mainSyncReceived := make(chan string, 1)

	// Replace syncLabelHandler to capture label sync events
	c.syncLabelHandler = func(ctx context.Context, key string) error {
		labelSyncReceived <- key
		return nil
	}

	// Replace syncHandler to capture main sync events
	c.syncHandler = func(ctx context.Context, key string) error {
		mainSyncReceived <- key
		return nil
	}

	// Start only the pod informer and controller
	go informerFactory.Core().V1().Pods().Informer().RunWithContext(ctx)
	go c.Run(ctx, 1)

	// Create a pod and send an update through the fake watch (label change)
	oldPod := newPod("default", "test-pod", "test-uid")
	oldPod.Labels = map[string]string{"app": "old-value"}

	newPodObj := oldPod.DeepCopy()
	newPodObj.Labels = map[string]string{"app": "new-value"}
	newPodObj.ResourceVersion = "2"

	// First add the old pod
	fakeWatch.Add(oldPod)

	// Then modify it (label change should trigger label sync)
	fakeWatch.Modify(newPodObj)

	// Verify the syncLabelHandler received the key
	select {
	case key := <-labelSyncReceived:
		expectedKey := "default/test-pod"
		if key != expectedKey {
			t.Errorf("expected label sync key %s, got %s", expectedKey, key)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for label sync")
	}

	// Now delete the pod (should trigger main sync)
	fakeWatch.Delete(newPodObj)

	// Verify the syncHandler received the key with EvictionRequest format (namespace/uid)
	select {
	case key := <-mainSyncReceived:
		expectedKey := "default/test-uid"
		if key != expectedKey {
			t.Errorf("expected main sync key %s, got %s", expectedKey, key)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for main sync after pod deletion")
	}
}

func TestPodWatch_TerminalPhaseTriggersMainSync(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()

	fakeWatch := watch.NewFakeWithOptions(watch.FakeOptions{Logger: &logger})
	client.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))

	informerFactory := informers.NewSharedInformerFactory(client, 0)

	c, err := NewController(
		ctx,
		informerFactory.Coordination().V1alpha1().EvictionRequests(),
		informerFactory.Core().V1().Pods(),
		client,
		testControllerName,
	)
	if err != nil {
		t.Fatalf("unexpected error creating controller: %v", err)
	}

	c.evictionRequestListerSynced = alwaysReady
	c.podListerSynced = alwaysReady

	mainSyncReceived := make(chan string, 1)
	c.syncHandler = func(ctx context.Context, key string) error {
		mainSyncReceived <- key
		return nil
	}
	c.syncLabelHandler = func(ctx context.Context, key string) error {
		return nil
	}

	go informerFactory.Core().V1().Pods().Informer().RunWithContext(ctx)
	go c.Run(ctx, 1)

	// Add a running pod
	pod := newPod("default", "test-pod", "test-uid")
	fakeWatch.Add(pod)

	// Transition pod to terminal phase (Succeeded)
	terminalPod := pod.DeepCopy()
	terminalPod.Status.Phase = v1.PodSucceeded
	terminalPod.ResourceVersion = "2"
	fakeWatch.Modify(terminalPod)

	// Verify the syncHandler received the key (terminal phase triggers main sync)
	select {
	case key := <-mainSyncReceived:
		expectedKey := "default/test-uid"
		if key != expectedKey {
			t.Errorf("expected main sync key %s, got %s", expectedKey, key)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for main sync after pod terminal phase transition")
	}
}

func TestDeletePod_HandlesAllObjectTypes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := []struct {
		name      string
		obj       any
		expectKey string
		wantQueue bool
	}{
		{
			name: "regular pod deletion",
			obj: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "test-uid",
				},
			},
			expectKey: "default/test-uid",
			wantQueue: true,
		},
		{
			name: "tombstone with pod",
			obj: cache.DeletedFinalStateUnknown{
				Key: "default/test-pod",
				Obj: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "default",
						UID:       "tombstone-uid",
					},
				},
			},
			expectKey: "default/tombstone-uid",
			wantQueue: true,
		},
		{
			name: "invalid tombstone object",
			obj: cache.DeletedFinalStateUnknown{
				Key: "default/test-pod",
				Obj: "not-a-pod",
			},
			wantQueue: false,
		},
		{
			name:      "invalid object type",
			obj:       "not-a-pod-or-tombstone",
			wantQueue: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, _, _ := newTestController(t, nil, nil, nil)

			initialLen := c.queue.Len()
			c.deletePod(logger, tt.obj)
			finalLen := c.queue.Len()

			if tt.wantQueue {
				if finalLen != initialLen+1 {
					t.Errorf("expected queue to have 1 more item, got %d", finalLen-initialLen)
				}

				// Verify the key that was enqueued
				if finalLen > 0 {
					item, _ := c.queue.Get()
					if item != tt.expectKey {
						t.Errorf("expected key %s, got %s", tt.expectKey, item)
					}
					c.queue.Done(item)
				}
			} else {
				if finalLen != initialLen {
					t.Errorf("expected queue length to remain %d, got %d", initialLen, finalLen)
				}
			}
		})
	}
}
