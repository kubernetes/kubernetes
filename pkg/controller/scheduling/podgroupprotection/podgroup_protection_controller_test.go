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

package podgroupprotection

import (
	"context"
	"slices"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/utils/ptr"
)

const (
	defaultNS     = "default"
	defaultPGName = "my-podgroup"
	defaultPGUID  = "pg-uid-1"
)

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
	pg.Finalizers = append(pg.Finalizers, scheduling.PodGroupProtectionFinalizer)
	return pg
}

func deletedPodGroup(pg *schedulingv1alpha2.PodGroup) *schedulingv1alpha2.PodGroup {
	pg.DeletionTimestamp = &metav1.Time{}
	return pg
}

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

func TestGetPod(t *testing.T) {
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
			got := getPod(tc.obj)
			if (got != nil) != tc.want {
				t.Errorf("parsePod() returned pod=%v, want non-nil=%v", got, tc.want)
			}
		})
	}
}

func TestHandlePodChange(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := map[string]struct {
		old      interface{}
		new      interface{}
		wantSize int
	}{
		"deleted pod with schedulingGroup enqueues": {
			old:      podForPG("pod-1", defaultPGName),
			new:      nil,
			wantSize: 1,
		},
		"terminated pod enqueues": {
			old:      podForPG("pod-1", defaultPGName),
			new:      terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
			wantSize: 1,
		},
		"unscheduled new pod does not enqueue": {
			old:      nil,
			new:      unscheduledPod(podForPG("pod-1", defaultPGName)),
			wantSize: 0,
		},
		"scheduled running pod does not enqueue": {
			old:      nil,
			new:      podForPG("pod-1", defaultPGName),
			wantSize: 0,
		},
		"deleted pod without schedulingGroup does not enqueue": {
			old:      podWithoutSchedulingGroup("pod-1"),
			new:      nil,
			wantSize: 0,
		},
		"deleted pod with nil podGroupName does not enqueue": {
			old: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: defaultNS},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{PodGroupName: nil},
				},
			},
			new:      nil,
			wantSize: 0,
		},
		"UID mismatch with terminated new pod, same PodGroup, deduplicates enqueue": {
			old: func() *v1.Pod {
				p := podForPG("pod-1", defaultPGName)
				p.UID = "old-uid"
				return p
			}(),
			new: func() interface{} {
				p := terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded)
				p.UID = "new-uid"
				return p
			}(),
			wantSize: 1,
		},
		"UID mismatch with terminated new pod referencing different PodGroup enqueues both": {
			old: func() *v1.Pod {
				p := podForPG("pod-1", defaultPGName)
				p.UID = "old-uid"
				return p
			}(),
			new: func() interface{} {
				p := terminatedPod(podForPG("pod-1", "other-pg"), v1.PodFailed)
				p.UID = "new-uid"
				return p
			}(),
			wantSize: 2,
		},
		"UID mismatch on update with non-terminated new pod enqueues only old pod": {
			old: func() *v1.Pod {
				p := podForPG("pod-1", defaultPGName)
				p.UID = "old-uid"
				return p
			}(),
			new: func() interface{} {
				p := unscheduledPod(podForPG("pod-1", defaultPGName))
				p.UID = "new-uid"
				return p
			}(),
			wantSize: 1,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c := &Controller{
				queue: workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
			}
			defer c.queue.ShutDown()
			c.handlePodChange(logger, tc.old, tc.new)

			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}

func TestPodGroupProtectionController(t *testing.T) {
	tests := []struct {
		name string
		// Objects to seed into the fake client before the controller starts.
		initialObjects []runtime.Object
		// Pod to delete (by name in defaultNS) after the controller starts.
		podToDelete string
		// Whether the finalizer should be present on the PodGroup after the
		// controller has finished processing.
		expectFinalizer bool
	}{
		{
			name:            "new PodGroup without finalizer, no action (admission plugin handles it)",
			initialObjects:  []runtime.Object{podGroup()},
			expectFinalizer: false,
		},
		{
			name:            "new PodGroup without finalizer, active pod exists, then no action (admission plugin handles it)",
			initialObjects:  []runtime.Object{podGroup(), podForPG("pod-1", defaultPGName)},
			expectFinalizer: false,
		},
		{
			name:            "PodGroup with finalizer, not being deleted, then no action",
			initialObjects:  []runtime.Object{withFinalizer(podGroup())},
			expectFinalizer: true,
		},
		{
			name:            "deleted PodGroup with finalizer, no active pods, then finalizer is removed",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup()))},
			expectFinalizer: false,
		},
		{
			name:            "deleted PodGroup with finalizer, active pod exists, then finalizer is kept",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup())), podForPG("pod-1", defaultPGName)},
			expectFinalizer: true,
		},
		{
			name:            "deleted PodGroup with finalizer, only terminated pods, then finalizer is removed",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup())), terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded)},
			expectFinalizer: false,
		},
		{
			name: "deleted PodGroup with finalizer, mix of active and terminated pods, then finalizer is kept",
			initialObjects: []runtime.Object{
				deletedPodGroup(withFinalizer(podGroup())),
				podForPG("pod-active", defaultPGName),
				terminatedPod(podForPG("pod-done", defaultPGName), v1.PodSucceeded),
			},
			expectFinalizer: true,
		},
		{
			name:            "PodGroup without finalizer, already deleted, then no action (not a deletion candidate)",
			initialObjects:  []runtime.Object{deletedPodGroup(podGroup())},
			expectFinalizer: false,
		},
		{
			name:            "PodGroup without finalizer, already deleted, active pods exist, then no action (should not add finalizer to deleting object)",
			initialObjects:  []runtime.Object{deletedPodGroup(podGroup()), podForPG("pod-1", defaultPGName)},
			expectFinalizer: false,
		},
		{
			name:            "pod deleted, PodGroup being deleted with finalizer, was last active pod, finalizer is removed",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup())), podForPG("pod-1", defaultPGName)},
			podToDelete:     "pod-1",
			expectFinalizer: false,
		},
		{
			name:            "pod terminated succeeded, PodGroup being deleted with finalizer, was last active pod, then finalizer is removed",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup())), terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded)},
			expectFinalizer: false,
		},
		{
			name:            "pod terminated failed, PodGroup being deleted with finalizer, was last active pod, then finalizer is removed",
			initialObjects:  []runtime.Object{deletedPodGroup(withFinalizer(podGroup())), terminatedPod(podForPG("pod-1", defaultPGName), v1.PodFailed)},
			expectFinalizer: false,
		},
		{
			name:            "new unscheduled pod, PodGroup without finalizer, then no action (admission plugin handles it)",
			initialObjects:  []runtime.Object{podGroup(), unscheduledPod(podForPG("pod-1", defaultPGName))},
			expectFinalizer: false,
		},
		{
			name:            "pod without schedulingGroup -> no PodGroup action",
			initialObjects:  []runtime.Object{withFinalizer(podGroup()), podWithoutSchedulingGroup("pod-1")},
			expectFinalizer: true,
		},
		{
			name:            "terminated pod references non-existent PodGroup, controller handles gracefully",
			initialObjects:  []runtime.Object{terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded)},
			expectFinalizer: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			t.Cleanup(cancel)

			client := fake.NewClientset(test.initialObjects...)
			informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
			pgInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			podInformer := informerFactory.Core().V1().Pods()

			ctrl, err := NewPodGroupProtectionController(klog.FromContext(ctx), pgInformer, podInformer, client)
			if err != nil {
				t.Fatalf("unexpected error creating controller: %v", err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			go ctrl.Run(ctx, 1)

			if test.podToDelete != "" {
				if err := client.CoreV1().Pods(defaultNS).Delete(ctx, test.podToDelete, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("deleting pod: %v", err)
				}
			}

			if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				pg, err := client.SchedulingV1alpha2().PodGroups(defaultNS).Get(ctx, defaultPGName, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return !test.expectFinalizer, nil
				}
				if err != nil {
					return false, err
				}
				hasFinalizer := slices.Contains(pg.Finalizers, scheduling.PodGroupProtectionFinalizer)
				return hasFinalizer == test.expectFinalizer, nil
			}); err != nil {
				t.Fatalf("timed out waiting for expected finalizer state (want present=%v): %v", test.expectFinalizer, err)
			}
		})
	}
}

func TestActivePodSchedulingGroupIndexer(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := addActivePodSchedulingGroupIndexer(indexer); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pod1 := podForPG("pod-1", "pg-a")
	_ = indexer.Add(pod1)

	pod2 := podWithoutSchedulingGroup("pod-2")
	_ = indexer.Add(pod2)

	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-3", Namespace: defaultNS, UID: "pod-3-uid"},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{PodGroupName: nil},
		},
	}
	_ = indexer.Add(pod3)

	// Terminated pod should not appear in the index.
	pod4 := terminatedPod(podForPG("pod-4", "pg-a"), v1.PodSucceeded)
	_ = indexer.Add(pod4)

	objs, err := indexer.ByIndex(activePodSchedulingGroupIndex, defaultNS+"/pg-a")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 1 {
		t.Fatalf("expected 1 active pod for pg-a, got %d", len(objs))
	}
	if objs[0].(*v1.Pod).Name != "pod-1" {
		t.Errorf("expected pod-1, got %s", objs[0].(*v1.Pod).Name)
	}

	// Nonexistent PodGroup should return empty.
	objs, err = indexer.ByIndex(activePodSchedulingGroupIndex, defaultNS+"/nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 0 {
		t.Errorf("expected 0 pods for nonexistent PodGroup, got %d", len(objs))
	}
}

func TestHasActivePods(t *testing.T) {
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
			if err := addActivePodSchedulingGroupIndexer(indexer); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			for _, obj := range tc.pods {
				_ = indexer.Add(obj)
			}

			ctrl := &Controller{
				podIndexer: indexer,
			}
			pg := podGroup()
			got, err := ctrl.hasActivePods(context.Background(), pg)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("hasActivePods() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHandlePodGroupUpdate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := map[string]struct {
		pg       *schedulingv1alpha2.PodGroup
		wantSize int
	}{
		"PodGroup without finalizer, not deleting/not enqueued": {
			pg:       podGroup(),
			wantSize: 0,
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
				queue: workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
			}
			defer c.queue.ShutDown()
			c.handlePodGroupUpdate(logger, tc.pg)
			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}
