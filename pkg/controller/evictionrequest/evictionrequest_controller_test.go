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

	"github.com/google/go-cmp/cmp"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"

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
	"k8s.io/utils/ptr"
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
				Pod: &coordinationv1alpha1.PodReference{
					Name: podName,
					UID:  types.UID(podUID),
				},
			},
			Requesters: []coordinationv1alpha1.Requester{
				{Name: "test-requester.example.com"},
			},
		},
	}
}

func hasCanceledCondition(er *coordinationv1alpha1.EvictionRequest) bool {
	return meta.IsStatusConditionTrue(er.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed))
}

func hasEvictedCondition(er *coordinationv1alpha1.EvictionRequest) bool {
	return meta.IsStatusConditionTrue(er.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted))
}

// getLastStatusPatch finds the last patch action targeting the status subresource
// and unmarshals it into an EvictionRequest. This is needed because the fake client
// doesn't implement SSA field ownership tracking, so fields absent from the apply
// config (e.g. TargetResponders with terminal states) aren't removed from the stored object.
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

// findTargetResponderState looks up the state of a named target responder.
func findTargetResponderState(targetResponders []coordinationv1alpha1.TargetResponder, name string) coordinationv1alpha1.ResponderStateType {
	for _, tr := range targetResponders {
		if tr.Name == name {
			return tr.State
		}
	}
	return ""
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

func TestSyncHandler_ValidationUnsupportedTarget(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create EvictionRequest with no pod target (unsupported target type)
	er := &coordinationv1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "no-target",
		},
		Spec: coordinationv1alpha1.EvictionRequestSpec{
			Target:     coordinationv1alpha1.EvictionTarget{},
			Requesters: []coordinationv1alpha1.Requester{{Name: "test-requester.example.com"}},
		},
	}
	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, nil, nil)

	err := c.syncHandler(ctx, "default/no-target")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "no-target", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if !hasCanceledCondition(updated) {
		t.Error("expected Canceled condition to be set for unsupported target type")
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

func TestSyncHandler_ValidationPodGroup(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Create pod with PodGroup set
	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("my-podgroup")}

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
		t.Error("expected Canceled condition to be set for PodGroup pod")
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

	// Create EvictionRequest with TargetResponders already set (means validation passed before)
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = ptr.To[int64](1) // Simulate that we've already observed this request
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction)},
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

func TestSyncHandler_PodNotFound_MovesActiveResponderToProcessed(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testResponder := "cleanup-responder.example.com"
	now := metav1.Now()

	// Create EvictionRequest with an active responder that has a heartbeat
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = ptr.To[int64](1)
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: testResponder, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	// Set responder status with heartbeat so it doesn't auto-advance
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:          testResponder,
			HeartbeatTime: &now,
		},
	}

	// Pod doesn't exist (deleted while responder was active)
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

	// No responder should be active
	if findActiveTargetResponderIdx(updated.Status.TargetResponders) != -1 {
		t.Error("expected no active target responder")
	}

	// The active responder should now have a terminal state (Interrupted, since pod is gone
	// and the responder did not report CompletionTime)
	state := findTargetResponderState(updated.Status.TargetResponders, testResponder)
	if state != coordinationv1alpha1.ResponderStateInterrupted {
		t.Errorf("expected responder %s to be in terminal state, got %s", testResponder, state)
	}

	// The responder status entry should still exist (controller includes all entries
	// by Name to prevent SSA removal), but HeartbeatTime and CompletionTime are NOT
	// included — those are responder-owned fields.
	var responderStatus *coordinationv1alpha1.ResponderStatus
	for i := range updated.Status.Responders {
		if updated.Status.Responders[i].Name == testResponder {
			responderStatus = &updated.Status.Responders[i]
			break
		}
	}
	if responderStatus == nil {
		t.Fatalf("expected responder status entry to exist")
	}
	if responderStatus.CompletionTime != nil {
		t.Error("expected controller to NOT include CompletionTime (responder-owned)")
	}
	if responderStatus.HeartbeatTime != nil {
		t.Error("expected controller to NOT include HeartbeatTime (responder-owned)")
	}
}

func TestSyncHandler_PodTerminal_DefersCompletionForActiveResponder(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testResponder := "cleanup-responder.example.com"

	now := time.Now()
	fakeClock := testingclock.NewFakeClock(now)

	// Pod is terminal (Succeeded) with DeletionTimestamp set 1s ago
	pod := newPod("default", "deleted-pod", "deleted-uid")
	deletionTime := metav1.NewTime(now.Add(-1 * time.Second))
	pod.DeletionTimestamp = &deletionTime
	pod.Status.Phase = v1.PodSucceeded

	// Create EvictionRequest with an active responder that hasn't completed
	er := newEvictionRequest("default", "deleted-pod", "deleted-uid")
	er.Generation = 1
	er.Status.ObservedGeneration = ptr.To[int64](1)
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: testResponder, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	heartbeatTime := metav1.NewTime(now.Add(-10 * time.Second))
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:          testResponder,
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

	// Should NOT be Evicted yet as we're giving the responder grace time to complete
	if hasEvictedCondition(updated) {
		t.Error("expected Evicted condition to NOT be set yet (grace period not elapsed)")
	}

	// Active responder should still be active
	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Error("expected an active responder during grace period")
	} else if updated.Status.TargetResponders[activeIdx].Name != testResponder {
		t.Errorf("expected active responder to be %s, got %s", testResponder, updated.Status.TargetResponders[activeIdx].Name)
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
	// ownership (TargetResponder state changes won't be reflected via Get).
	updated = getLastStatusPatch(t, client)

	// Now it should be Evicted
	if !hasEvictedCondition(updated) {
		t.Error("expected Evicted condition after grace period elapsed")
	}

	// No responder should be active and the responder should have a terminal state
	if findActiveTargetResponderIdx(updated.Status.TargetResponders) != -1 {
		t.Error("expected no active target responder after grace period")
	}
	state := findTargetResponderState(updated.Status.TargetResponders, testResponder)
	if state != coordinationv1alpha1.ResponderStateCompleted && state != coordinationv1alpha1.ResponderStateInterrupted {
		t.Errorf("expected responder %s to be in terminal state, got %s", testResponder, state)
	}
}

func TestSyncHandler_InitializeTargetResponders(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionResponders = []v1.EvictionResponder{
		{Name: "custom-responder.example.com"},
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify TargetResponders initialized
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	if len(updated.Status.TargetResponders) != 2 {
		t.Errorf("expected 2 target responders, got %d", len(updated.Status.TargetResponders))
	}

	if updated.Status.TargetResponders[0].Name != "custom-responder.example.com" {
		t.Errorf("expected first responder to be custom, got %s", updated.Status.TargetResponders[0].Name)
	}

	if updated.Status.TargetResponders[1].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected last responder to be imperative, got %s", updated.Status.TargetResponders[1].Name)
	}

	// Verify responder statuses were initialized for ALL target responders
	if len(updated.Status.Responders) != 2 {
		t.Fatalf("expected 2 responder statuses, got %d", len(updated.Status.Responders))
	}

	// Active responder (first one) should have StartTime set
	customStatus := findResponderStatus(updated.Status.Responders, "custom-responder.example.com")
	if customStatus == nil {
		t.Fatal("expected responder status for custom responder")
	}
	if customStatus.StartTime == nil {
		t.Error("expected active responder to have start time initialized")
	}
	if customStatus.HeartbeatTime != nil {
		t.Error("expected active responder to not have heartbeat time set by controller")
	}

	// Non-active responder should have status entry but no StartTime
	imperativeStatus := findResponderStatus(updated.Status.Responders, string(coordinationv1alpha1.EvictionResponderImperativeEviction))
	if imperativeStatus == nil {
		t.Fatal("expected responder status for imperative responder")
	}
	if imperativeStatus.StartTime != nil {
		t.Error("expected non-active responder to not have start time set")
	}
}

func TestSyncHandler_SelectFirstResponder(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")
	er := newEvictionRequest("default", "test-pod", "test-uid")

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the first responder is set to Active in TargetResponders
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Fatal("expected an active responder")
	}

	// Since pod has no EvictionResponders, first should be imperative
	if updated.Status.TargetResponders[activeIdx].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected active responder to be imperative, got %s", updated.Status.TargetResponders[activeIdx].Name)
	}

	// Verify responder status was initialized
	if len(updated.Status.Responders) != 1 {
		t.Fatalf("expected 1 responder status, got %d", len(updated.Status.Responders))
	}

	responderStatus := updated.Status.Responders[0]
	if responderStatus.Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected responder status name to be %s, got %s",
			string(coordinationv1alpha1.EvictionResponderImperativeEviction), responderStatus.Name)
	}

	if responderStatus.StartTime == nil {
		t.Error("expected responder status to have start time initialized")
	}

	if responderStatus.HeartbeatTime != nil {
		t.Error("expected responder status to not have heartbeat time set")
	}
}

func TestSyncHandler_AdvanceOnCompletion(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testEvictionResponderClass := "first-responder.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionResponders = []v1.EvictionResponder{
		{Name: testEvictionResponderClass},
	}

	er := newEvictionRequest("default", "test-pod", "test-uid")
	// Set up status as if first responder is active and completed
	now := metav1.Now()
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: testEvictionResponderClass, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:           testEvictionResponderClass,
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

	// Verify advanced to next responder
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Fatal("expected an active responder after advancement")
	}

	if updated.Status.TargetResponders[activeIdx].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected active responder to advance to imperative, got %s", updated.Status.TargetResponders[activeIdx].Name)
	}

	// Verify first responder has a terminal state (Completed)
	state := findTargetResponderState(updated.Status.TargetResponders, testEvictionResponderClass)
	if state != coordinationv1alpha1.ResponderStateCompleted {
		t.Errorf("expected first responder to be Completed, got %s", state)
	}

	// Verify newly active responder (imperative) has StartTime set
	imperativeStatus := findResponderStatus(updated.Status.Responders, string(coordinationv1alpha1.EvictionResponderImperativeEviction))
	if imperativeStatus == nil {
		t.Fatal("expected responder status for newly active imperative responder")
	}
	if imperativeStatus.StartTime == nil {
		t.Error("expected newly active responder to have start time set")
	}
	if imperativeStatus.HeartbeatTime != nil {
		t.Error("expected newly active responder to not have heartbeat time set by controller")
	}

	// Verify completed responder entry is preserved (by Name) but the controller
	// does NOT include responder-owned fields like CompletionTime or Message.
	// In a real cluster, those fields would still exist under the responder's
	// own field manager — they just aren't part of the controller's apply config.
	firstStatus := findResponderStatus(updated.Status.Responders, testEvictionResponderClass)
	if firstStatus == nil {
		t.Fatal("expected responder status entry to be preserved")
	}
}

func TestSyncHandler_AdvanceOnTimeout(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	slowEvictionResponderClass := "slow-responder.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionResponders = []v1.EvictionResponder{
		{Name: slowEvictionResponderClass},
	}

	// Create fake clock first so we can set heartbeat time before creating controller
	fakeClock := testingclock.NewFakeClock(time.Now())
	heartbeatTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: slowEvictionResponderClass, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:          slowEvictionResponderClass,
			HeartbeatTime: &heartbeatTime,
			Message:       "still working",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock past the timeout
	fakeClock.Step(2 * ResponderHeartbeatTimeout)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify advanced to next responder due to timeout
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Fatal("expected an active responder after timeout advancement")
	}

	if updated.Status.TargetResponders[activeIdx].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected active responder to advance to imperative after timeout, got %s", updated.Status.TargetResponders[activeIdx].Name)
	}

	// Verify slow responder has a terminal state (Interrupted due to timeout)
	state := findTargetResponderState(updated.Status.TargetResponders, slowEvictionResponderClass)
	if state != coordinationv1alpha1.ResponderStateInterrupted {
		t.Errorf("expected slow responder to be Interrupted, got %s", state)
	}
}

func TestSyncHandler_AdvanceOnTimeoutWithoutHeartbeat(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	staleResponder := "stale-responder.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionResponders = []v1.EvictionResponder{
		{Name: staleResponder},
	}

	fakeClock := testingclock.NewFakeClock(time.Now())
	startTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: staleResponder, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	// Responder has StartTime but never heartbeated
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:      staleResponder,
			StartTime: &startTime,
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock past the timeout
	fakeClock.Step(2 * ResponderHeartbeatTimeout)

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify advanced to next responder due to StartTime-based timeout
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Fatal("expected an active responder after StartTime timeout advancement")
	}

	if updated.Status.TargetResponders[activeIdx].Name != string(coordinationv1alpha1.EvictionResponderImperativeEviction) {
		t.Errorf("expected active responder to advance to imperative after StartTime timeout, got %s", updated.Status.TargetResponders[activeIdx].Name)
	}

	// Verify stale responder has a terminal state (Interrupted due to timeout)
	state := findTargetResponderState(updated.Status.TargetResponders, staleResponder)
	if state != coordinationv1alpha1.ResponderStateInterrupted {
		t.Errorf("expected stale responder to be Interrupted, got %s", state)
	}
}

func TestSyncHandler_NoAdvanceBeforeTimeout(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	activeEvictionResponderClass := "active-responder.example.com"

	pod := newPod("default", "test-pod", "test-uid")
	pod.Spec.EvictionResponders = []v1.EvictionResponder{
		{Name: activeEvictionResponderClass},
	}

	// Create fake clock first so we can set heartbeat time before creating controller
	fakeClock := testingclock.NewFakeClock(time.Now())
	heartbeatTime := metav1.NewTime(fakeClock.Now())

	er := newEvictionRequest("default", "test-pod", "test-uid")
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: activeEvictionResponderClass, State: coordinationv1alpha1.ResponderStateActive},
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateInactive},
	}
	er.Status.ObservedGeneration = ptr.To[int64](er.Generation)
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:          activeEvictionResponderClass,
			HeartbeatTime: &heartbeatTime,
			Message:       "working",
		},
	}

	c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, fakeClock)

	// Advance clock but NOT past the timeout
	fakeClock.Step(ResponderHeartbeatTimeout / 4)

	// Clear actions
	client.ClearActions()

	// Sync
	err := c.syncHandler(ctx, "default/test-uid")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify responder was NOT advanced (still the same)
	updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get eviction request: %v", err)
	}

	activeIdx := findActiveTargetResponderIdx(updated.Status.TargetResponders)
	if activeIdx == -1 {
		t.Fatal("expected an active responder before timeout")
	}

	if updated.Status.TargetResponders[activeIdx].Name != activeEvictionResponderClass {
		t.Errorf("expected active responder to remain unchanged before timeout, got %s", updated.Status.TargetResponders[activeIdx].Name)
	}

	// Verify no responder has a terminal state yet
	for _, tr := range updated.Status.TargetResponders {
		if tr.State == coordinationv1alpha1.ResponderStateCompleted || tr.State == coordinationv1alpha1.ResponderStateInterrupted {
			t.Errorf("expected no responder in terminal state before timeout, but %s is %s", tr.Name, tr.State)
		}
	}
}

func TestSyncHandler_AllRespondersProcessed(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	pod := newPod("default", "test-pod", "test-uid")

	er := newEvictionRequest("default", "test-pod", "test-uid")
	// Set up status where the last responder is active and has completed
	now := metav1.Now()
	er.Status.TargetResponders = []coordinationv1alpha1.TargetResponder{
		{Name: string(coordinationv1alpha1.EvictionResponderImperativeEviction), State: coordinationv1alpha1.ResponderStateActive},
	}
	er.Status.Responders = []coordinationv1alpha1.ResponderStatus{
		{
			Name:           string(coordinationv1alpha1.EvictionResponderImperativeEviction),
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
	// ownership (TargetResponder state changes won't be reflected via Get).
	updated := getLastStatusPatch(t, client)

	if findActiveTargetResponderIdx(updated.Status.TargetResponders) != -1 {
		t.Error("expected no active responder when all are in terminal state")
	}

	// Verify imperative responder has terminal state (Completed)
	state := findTargetResponderState(updated.Status.TargetResponders, string(coordinationv1alpha1.EvictionResponderImperativeEviction))
	if state != coordinationv1alpha1.ResponderStateCompleted {
		t.Errorf("expected imperative responder to be Completed, got %s", state)
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

	if !ptr.Equal(updated.Status.ObservedGeneration, ptr.To[int64](5)) {
		t.Errorf("expected ObservedGeneration to be 5, got %d", ptr.Deref(updated.Status.ObservedGeneration, -1))
	}
}

func TestSyncLabelHandler(t *testing.T) {
	testCases := []struct {
		name           string
		podLabels      map[string]string
		erLabels       map[string]string
		expectedLabels map[string]string
	}{
		{
			name: "syncs pod labels to eviction request",
			podLabels: map[string]string{
				"app":     "myapp",
				"version": "v1",
			},
			erLabels: map[string]string{
				"existing": "label",
			},
			expectedLabels: map[string]string{
				"existing": "label",
				"app":      "myapp",
				"version":  "v1",
			},
		},
		{
			name: "pod labels overwrite eviction request labels",
			podLabels: map[string]string{
				"app": "pod-value",
			},
			erLabels: map[string]string{
				"app": "er-value",
			},
			expectedLabels: map[string]string{
				"app": "pod-value",
			},
		},
		{
			name:      "nil pod labels with existing er labels",
			podLabels: nil,
			erLabels: map[string]string{
				"existing": "label",
			},
			expectedLabels: map[string]string{
				"existing": "label",
			},
		},
		{
			name: "pod labels with nil er labels",
			podLabels: map[string]string{
				"app": "myapp",
			},
			erLabels: nil,
			expectedLabels: map[string]string{
				"app": "myapp",
			},
		},
		{
			name:           "both nil labels",
			podLabels:      nil,
			erLabels:       nil,
			expectedLabels: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			pod := newPod("default", "test-pod", "test-uid")
			pod.Labels = tc.podLabels

			er := newEvictionRequest("default", "test-pod", "test-uid")
			er.Labels = tc.erLabels

			c, _, client := newTestController(t, []*coordinationv1alpha1.EvictionRequest{er}, []*v1.Pod{pod}, nil)

			err := c.syncLabelHandler(ctx, "default/test-pod")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			updated, err := client.CoordinationV1alpha1().EvictionRequests("default").Get(ctx, "test-uid", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get eviction request: %v", err)
			}

			if diff := cmp.Diff(tc.expectedLabels, updated.Labels); diff != "" {
				t.Errorf("unexpected labels (-want +got):\n%s", diff)
			}
		})
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
			fakeClock := testingclock.NewFakeClock(time.Now())
			c, _, _ := newTestController(t, nil, nil, fakeClock)

			initialLen := c.queue.Len()
			c.deletePod(logger, tt.obj)

			// The controller uses AddAfter with GracefulCompletionDelay, so advance the clock
			fakeClock.Step(GracefulCompletionDelay + time.Second)
			// Give the queue time to process the delayed item
			time.Sleep(10 * time.Millisecond)

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

func TestValidate(t *testing.T) {
	clock := testingclock.NewFakePassiveClock(time.Now())
	tests := []struct {
		name      string
		target    coordinationv1alpha1.EvictionTarget
		pod       *v1.Pod
		wantValid bool
		expected  []metav1ac.ConditionApplyConfiguration
	}{
		{
			name:     "valid pod",
			target:   makePodTarget("my-pod", "uid-1"),
			pod:      testPod("my-pod", "uid-1"),
			expected: []metav1ac.ConditionApplyConfiguration{},
		},
		{
			name:   "pod not found",
			target: makePodTarget("my-pod", "uid-1"),
			pod:    nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionFailed,
					metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid,
					"Target pod not found"),
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionEvicted,
					metav1.ConditionFalse, coordinationv1alpha1.EvictionRequestConditionReasonEvictionFailed, ""),
			},
		},
		{
			name:   "UID mismatch",
			target: makePodTarget("my-pod", "uid-1"),
			pod:    testPod("my-pod", "uid-2"),
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionFailed,
					metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid,
					"Target pod UID mismatch: expected uid-1, got uid-2"),
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionEvicted,
					metav1.ConditionFalse, coordinationv1alpha1.EvictionRequestConditionReasonEvictionFailed, ""),
			},
		},
		{
			name:   "pod with PodGroup",
			target: makePodTarget("my-pod", "uid-1"),
			pod: func() *v1.Pod {
				pod := testPod("my-pod", "uid-1")
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("my-podgroup")}
				return pod
			}(),
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionFailed,
					metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid,
					"Target pod references a SchedulingGroup. Eviction is currently not supported."),
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionEvicted,
					metav1.ConditionFalse, coordinationv1alpha1.EvictionRequestConditionReasonEvictionFailed, ""),
			},
		},
		{
			name:   "empty target",
			target: coordinationv1alpha1.EvictionTarget{},
			pod:    nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionFailed,
					metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid,
					"Unsupported target type"),
				*setCondition(clock.Now(), nil, coordinationv1alpha1.EvictionRequestConditionEvicted,
					metav1.ConditionFalse, coordinationv1alpha1.EvictionRequestConditionReasonEvictionFailed, ""),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			request := coordinationv1alpha1.EvictionRequest{
				Spec: coordinationv1alpha1.EvictionRequestSpec{
					Requesters: []coordinationv1alpha1.Requester{
						{Name: "foo.example.com"},
					},
					Target: tc.target,
				},
			}
			failed, evicted := validate(clock, &request, newTargetInfo(tc.target, tc.pod))
			got := []metav1ac.ConditionApplyConfiguration{}
			if failed != nil {
				got = append(got, *failed)
			}
			if evicted != nil {
				got = append(got, *evicted)
			}
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Errorf("unexpected conditions update (-want +got):\n%s", diff)
			}
		})
	}
}
