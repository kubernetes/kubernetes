/*
Copyright 2026 The Kubernetes Authors.

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

package podgroupprotection

import (
	"context"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/utils/dump"
	"k8s.io/utils/ptr"
)

const (
	defaultNS     = "default"
	defaultPGName = "my-podgroup"
	defaultPGUID  = "pg-uid-1"
)

// -- PodGroup helpers --

func podGroup() *schedulingv1alpha2.PodGroup {
	return &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPGName,
			Namespace: defaultNS,
			UID:       defaultPGUID,
		},
	}
}

func withFinalizer(pg *schedulingv1alpha2.PodGroup) *schedulingv1alpha2.PodGroup {
	pg.Finalizers = append(pg.Finalizers, PodGroupProtectionFinalizer)
	return pg
}

func deletedPodGroup(pg *schedulingv1alpha2.PodGroup) *schedulingv1alpha2.PodGroup {
	pg.DeletionTimestamp = &metav1.Time{}
	return pg
}

// -- Pod helpers --

func podForPG(name string, pgName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: defaultNS,
			UID:       types.UID(name + "-uid"),
		},
		Spec: v1.PodSpec{
			NodeName: "node-1",
			SchedulingGroup: &v1.PodSchedulingGroup{
				PodGroupName: ptr.To(pgName),
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

func terminatedPod(pod *v1.Pod, phase v1.PodPhase) *v1.Pod {
	pod.Status.Phase = phase
	return pod
}

func unscheduledPod(pod *v1.Pod) *v1.Pod {
	pod.Spec.NodeName = ""
	return pod
}

func podWithUID(uid types.UID, pod *v1.Pod) *v1.Pod {
	pod.UID = uid
	return pod
}

func podWithoutSchedulingGroup(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: defaultNS,
			UID:       types.UID(name + "-uid"),
		},
		Spec: v1.PodSpec{
			NodeName: "node-1",
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

func TestIsPodTerminated(t *testing.T) {
	tests := map[string]struct {
		phase v1.PodPhase
		want  bool
	}{
		"running":   {phase: v1.PodRunning, want: false},
		"pending":   {phase: v1.PodPending, want: false},
		"succeeded": {phase: v1.PodSucceeded, want: true},
		"failed":    {phase: v1.PodFailed, want: true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			pod := &v1.Pod{Status: v1.PodStatus{Phase: tc.phase}}
			if got := isPodTerminated(pod); got != tc.want {
				t.Errorf("isPodTerminated(%v) = %v, want %v", tc.phase, got, tc.want)
			}
		})
	}
}

func TestParsePod(t *testing.T) {
	tests := map[string]struct {
		obj  interface{}
		want bool
	}{
		"nil": {
			obj:  nil,
			want: false,
		},
		"pod": {
			obj:  &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}},
			want: true,
		},
		"tombstone with pod": {
			obj: cache.DeletedFinalStateUnknown{
				Key: "default/p",
				Obj: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p"}},
			},
			want: true,
		},
		"tombstone with non-pod": {
			obj: cache.DeletedFinalStateUnknown{
				Key: "default/p",
				Obj: &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "p"}},
			},
			want: false,
		},
		"non-pod object": {
			obj:  &v1.ConfigMap{},
			want: false,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got := parsePod(tc.obj)
			if (got != nil) != tc.want {
				t.Errorf("parsePod() returned pod=%v, want non-nil=%v", got, tc.want)
			}
		})
	}
}

func TestEnqueuePodGroupForPod(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := map[string]struct {
		pod      *v1.Pod
		deleted  bool
		wantSize int
	}{
		"deleted pod with schedulingGroup enqueues": {
			pod:      podForPG("pod-1", defaultPGName),
			deleted:  true,
			wantSize: 1,
		},
		"terminated pod enqueues": {
			pod:      terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
			deleted:  false,
			wantSize: 1,
		},
		"unscheduled new pod enqueues": {
			pod:      unscheduledPod(podForPG("pod-1", defaultPGName)),
			deleted:  false,
			wantSize: 1,
		},
		"scheduled running pod does not enqueue": {
			pod:      podForPG("pod-1", defaultPGName),
			deleted:  false,
			wantSize: 0,
		},
		"pod without schedulingGroup does not enqueue": {
			pod:      podWithoutSchedulingGroup("pod-1"),
			deleted:  true,
			wantSize: 0,
		},
		"pod with nil podGroupName does not enqueue": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: defaultNS},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{PodGroupName: nil},
				},
			},
			deleted:  true,
			wantSize: 0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c := &Controller{
				queue: newTestQueue(),
			}
			c.enqueuePodGroupForPod(logger, tc.pod, tc.deleted)

			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}

func TestPodAddedDeletedUpdated_UIDMismatch(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	c := &Controller{
		queue: newTestQueue(),
	}

	oldPod := podForPG("pod-1", defaultPGName)
	oldPod.UID = "old-uid"
	newPod := unscheduledPod(podForPG("pod-1", defaultPGName))
	newPod.UID = "new-uid"

	c.podAddedDeletedUpdated(logger, oldPod, newPod, false)

	// Should enqueue twice: once for the new pod (unscheduled) and once for the old pod (treated as deleted).
	if c.queue.Len() != 2 {
		t.Errorf("queue size = %d, want 2 (one for new pod, one for old pod treated as deleted)", c.queue.Len())
	}
}

func TestPodGroupProtectionController(t *testing.T) {
	pgGVR := schedulingv1alpha2.SchemeGroupVersion.WithResource("podgroups")

	tests := []struct {
		name string
		// Objects to insert into fake client before the test starts.
		initialObjects []runtime.Object
		// PodGroup event to simulate. Automatically added to initialObjects.
		updatedPG *schedulingv1alpha2.PodGroup
		// Pod events to simulate. Automatically added to initialObjects.
		updatedPod *v1.Pod
		deletedPod *v1.Pod
		// Expected client actions.
		expectedActions []clienttesting.Action
	}{
		// -- PodGroup events --
		{
			name:      "new PodGroup without finalizer, no pods -> no action",
			updatedPG: podGroup(),
			// NeedToAddFinalizer is true, but isBeingUsed is false -> no action.
			expectedActions: []clienttesting.Action{},
		},
		{
			name:      "new PodGroup without finalizer, active pod exists -> finalizer is added",
			updatedPG: podGroup(),
			initialObjects: []runtime.Object{
				podForPG("pod-1", defaultPGName),
			},
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, withFinalizer(podGroup())),
			},
		},
		{
			name:            "PodGroup with finalizer, not being deleted -> no action",
			updatedPG:       withFinalizer(podGroup()),
			expectedActions: []clienttesting.Action{},
		},
		{
			name:      "deleted PodGroup with finalizer, no active pods -> finalizer is removed",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, deletedPodGroup(podGroup())),
			},
		},
		{
			name:      "deleted PodGroup with finalizer, active pod exists -> finalizer is kept",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			initialObjects: []runtime.Object{
				podForPG("pod-1", defaultPGName),
			},
			expectedActions: []clienttesting.Action{},
		},
		{
			name:      "deleted PodGroup with finalizer, only terminated pods -> finalizer is removed",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			initialObjects: []runtime.Object{
				terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
			},
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, deletedPodGroup(podGroup())),
			},
		},
		{
			name:      "deleted PodGroup with finalizer, mix of active and terminated pods -> finalizer is kept",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			initialObjects: []runtime.Object{
				podForPG("pod-active", defaultPGName),
				terminatedPod(podForPG("pod-done", defaultPGName), v1.PodSucceeded),
			},
			expectedActions: []clienttesting.Action{},
		},
		{
			name:            "PodGroup without finalizer, already deleted -> no action (not a deletion candidate)",
			updatedPG:       deletedPodGroup(podGroup()),
			expectedActions: []clienttesting.Action{},
		},
		// -- Pod events --
		{
			name:      "pod deleted, PodGroup being deleted with finalizer, was last active pod -> finalizer is removed",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			deletedPod: podForPG("pod-1", defaultPGName),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, deletedPodGroup(podGroup())),
			},
		},
		{
			name:      "pod terminated (succeeded), PodGroup being deleted with finalizer, was last active pod -> finalizer is removed",
			updatedPG: deletedPodGroup(withFinalizer(podGroup())),
			updatedPod: terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, deletedPodGroup(podGroup())),
			},
		},
		{
			name:       "new unscheduled pod, PodGroup needs finalizer -> finalizer is added",
			updatedPG:  podGroup(),
			updatedPod: unscheduledPod(podForPG("pod-1", defaultPGName)),
			expectedActions: []clienttesting.Action{
				clienttesting.NewUpdateAction(pgGVR, defaultNS, withFinalizer(podGroup())),
			},
		},
		{
			name:       "pod without schedulingGroup -> no PodGroup action",
			updatedPG:  withFinalizer(podGroup()),
			updatedPod: podWithoutSchedulingGroup("pod-1"),
			expectedActions: []clienttesting.Action{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Build initial objects.
			var clientObjs []runtime.Object
			if test.updatedPG != nil {
				clientObjs = append(clientObjs, test.updatedPG)
			}
			if test.updatedPod != nil {
				clientObjs = append(clientObjs, test.updatedPod)
			}
			clientObjs = append(clientObjs, test.initialObjects...)

			client := fake.NewSimpleClientset(clientObjs...)
			informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
			pgInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			podInformer := informerFactory.Core().V1().Pods()

			logger, _ := ktesting.NewTestContext(t)
			ctrl, err := NewPodGroupProtectionController(logger, pgInformer, podInformer, client)
			if err != nil {
				t.Fatalf("unexpected error creating controller: %v", err)
			}

			// Populate informer caches.
			if test.updatedPG != nil {
				pgInformer.Informer().GetStore().Add(test.updatedPG)
			}
			for _, obj := range test.initialObjects {
				switch o := obj.(type) {
				case *v1.Pod:
					podInformer.Informer().GetStore().Add(o)
				case *schedulingv1alpha2.PodGroup:
					pgInformer.Informer().GetStore().Add(o)
				}
			}
			if test.updatedPod != nil {
				podInformer.Informer().GetStore().Add(test.updatedPod)
			}

			// Simulate events.
			if test.updatedPG != nil {
				ctrl.podGroupAddedUpdated(logger, test.updatedPG)
			}
			switch {
			case test.deletedPod != nil && test.updatedPod != nil && test.deletedPod.Namespace == test.updatedPod.Namespace && test.deletedPod.Name == test.updatedPod.Name:
				ctrl.podAddedDeletedUpdated(logger, test.deletedPod, test.updatedPod, false)
			case test.updatedPod != nil:
				ctrl.podAddedDeletedUpdated(logger, nil, test.updatedPod, false)
			case test.deletedPod != nil:
				ctrl.podAddedDeletedUpdated(logger, nil, test.deletedPod, true)
			}

			// Process the controller queue until expected results or timeout.
			timeout := time.Now().Add(10 * time.Second)
			for {
				if time.Now().After(timeout) {
					t.Errorf("timed out waiting for expected actions")
					break
				}
				if ctrl.queue.Len() > 0 {
					ctx := context.TODO()
					ctrl.processNextWorkItem(ctx)
				}
				if ctrl.queue.Len() > 0 {
					continue
				}
				// Filter out non-update actions (informer list/watch setup).
				actions := filterUpdateActions(client.Actions())
				if len(actions) < len(test.expectedActions) {
					time.Sleep(10 * time.Millisecond)
					continue
				}
				break
			}

			actions := filterUpdateActions(client.Actions())
			for i, action := range actions {
				if i >= len(test.expectedActions) {
					t.Errorf("%d unexpected actions: %+v", len(actions)-len(test.expectedActions), dump.Pretty(actions[i:]))
					break
				}
				expectedAction := test.expectedActions[i]
				if !reflect.DeepEqual(expectedAction, action) {
					t.Errorf("action %d\nExpected:\n%s\ngot:\n%s", i, dump.Pretty(expectedAction), dump.Pretty(action))
				}
			}
			if len(test.expectedActions) > len(actions) {
				t.Errorf("%d additional expected actions not seen", len(test.expectedActions)-len(actions))
				for _, a := range test.expectedActions[len(actions):] {
					t.Logf("    %+v", a)
				}
			}
		})
	}
}

// filterUpdateActions returns only Update actions from the list,
// filtering out List/Watch actions injected by the informer machinery.
func filterUpdateActions(actions []clienttesting.Action) []clienttesting.Action {
	var filtered []clienttesting.Action
	for _, a := range actions {
		if a.GetVerb() == "update" {
			filtered = append(filtered, a)
		}
	}
	return filtered
}

func TestPodSchedulingGroupIndexer(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := addPodSchedulingGroupIndexer(indexer); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Pod with schedulingGroup.
	pod1 := podForPG("pod-1", "pg-a")
	indexer.Add(pod1)

	// Pod without schedulingGroup.
	pod2 := podWithoutSchedulingGroup("pod-2")
	indexer.Add(pod2)

	// Pod with nil podGroupName.
	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-3", Namespace: defaultNS, UID: "pod-3-uid"},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{PodGroupName: nil},
		},
	}
	indexer.Add(pod3)

	// Look up by PodGroup key.
	objs, err := indexer.ByIndex(PodSchedulingGroupIndex, defaultNS+"/pg-a")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 1 {
		t.Fatalf("expected 1 pod for pg-a, got %d", len(objs))
	}
	if objs[0].(*v1.Pod).Name != "pod-1" {
		t.Errorf("expected pod-1, got %s", objs[0].(*v1.Pod).Name)
	}

	// Nonexistent PodGroup should return empty.
	objs, err = indexer.ByIndex(PodSchedulingGroupIndex, defaultNS+"/nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 0 {
		t.Errorf("expected 0 pods for nonexistent PodGroup, got %d", len(objs))
	}
}

func TestIsBeingUsed(t *testing.T) {
	tests := map[string]struct {
		pods []runtime.Object
		want bool
	}{
		"no pods": {
			pods: nil,
			want: false,
		},
		"active pod referencing PodGroup": {
			pods: []runtime.Object{
				podForPG("pod-1", defaultPGName),
			},
			want: true,
		},
		"only terminated pods": {
			pods: []runtime.Object{
				terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
				terminatedPod(podForPG("pod-2", defaultPGName), v1.PodFailed),
			},
			want: false,
		},
		"mix of active and terminated": {
			pods: []runtime.Object{
				podForPG("pod-active", defaultPGName),
				terminatedPod(podForPG("pod-done", defaultPGName), v1.PodSucceeded),
			},
			want: true,
		},
		"pods referencing different PodGroup": {
			pods: []runtime.Object{
				podForPG("pod-1", "other-pg"),
			},
			want: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if err := addPodSchedulingGroupIndexer(indexer); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			for _, obj := range tc.pods {
				indexer.Add(obj)
			}

			ctrl := &Controller{
				podIndexer: indexer,
			}
			pg := podGroup()
			got, err := ctrl.isBeingUsed(context.Background(), pg)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("isBeingUsed() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPodGroupAddedUpdated(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := map[string]struct {
		pg       *schedulingv1alpha2.PodGroup
		wantSize int
	}{
		"PodGroup needs finalizer -> enqueued": {
			pg:       podGroup(),
			wantSize: 1,
		},
		"PodGroup is deletion candidate -> enqueued": {
			pg:       deletedPodGroup(withFinalizer(podGroup())),
			wantSize: 1,
		},
		"PodGroup has finalizer, not deleting -> not enqueued": {
			pg:       withFinalizer(podGroup()),
			wantSize: 0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c := &Controller{
				queue: newTestQueue(),
			}
			c.podGroupAddedUpdated(logger, tc.pg)
			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}

func TestProcessPodGroup_NotFound(t *testing.T) {
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	pgInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
	podInformer := informerFactory.Core().V1().Pods()

	logger, _ := ktesting.NewTestContext(t)
	ctrl, err := NewPodGroupProtectionController(logger, pgInformer, podInformer, client)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Processing a non-existent PodGroup should not return an error.
	err = ctrl.processPodGroup(context.Background(), defaultNS+"/"+defaultPGName)
	if err != nil {
		t.Errorf("expected no error for not-found PodGroup, got: %v", err)
	}
}

// newTestQueue creates a simple rate-limiting queue for testing.
func newTestQueue() *fakeQueue {
	return &fakeQueue{
		items: make([]string, 0),
	}
}

// fakeQueue is a minimal workqueue implementation for unit tests that
// don't need rate limiting or shutdown semantics.
type fakeQueue struct {
	items []string
}

func (q *fakeQueue) Add(item string)                          { q.items = append(q.items, item) }
func (q *fakeQueue) Len() int                                 { return len(q.items) }
func (q *fakeQueue) Get() (string, bool)                      { item := q.items[0]; q.items = q.items[1:]; return item, false }
func (q *fakeQueue) Done(item string)                         {}
func (q *fakeQueue) ShutDown()                                {}
func (q *fakeQueue) ShutDownWithDrain()                       {}
func (q *fakeQueue) ShuttingDown() bool                       { return false }
func (q *fakeQueue) AddAfter(item string, d time.Duration)    {}
func (q *fakeQueue) AddRateLimited(item string)               { q.Add(item) }
func (q *fakeQueue) Forget(item string)                       {}
func (q *fakeQueue) NumRequeues(item string) int              { return 0 }
