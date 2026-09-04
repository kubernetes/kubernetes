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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
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
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

const (
	defaultNS     = "default"
	defaultPGName = "my-podgroup"
	defaultPGUID  = "pg-uid-1"
)

func podGroup() *schedulingv1beta1.PodGroup {
	return &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      defaultPGName,
			Namespace: defaultNS,
			UID:       defaultPGUID,
		},
	}
}

func withFinalizer(pg *schedulingv1beta1.PodGroup) *schedulingv1beta1.PodGroup {
	pg.Finalizers = append(pg.Finalizers, scheduling.PodGroupProtectionFinalizer)
	return pg
}

func deletedPodGroup(pg *schedulingv1beta1.PodGroup) *schedulingv1beta1.PodGroup {
	pg.DeletionTimestamp = &metav1.Time{}
	return pg
}

func withCPGFinalizer(cpg *schedulingv1alpha3.CompositePodGroup) *schedulingv1alpha3.CompositePodGroup {
	cpg.Finalizers = append(cpg.Finalizers, scheduling.CompositePodGroupProtectionFinalizer)
	return cpg
}

func deletedCompositePodGroup(cpg *schedulingv1alpha3.CompositePodGroup) *schedulingv1alpha3.CompositePodGroup {
	cpg.DeletionTimestamp = &metav1.Time{}
	return cpg
}

func podGroupWithParent(name string, parentName string) *schedulingv1beta1.PodGroup {
	pg := podGroup()
	pg.Name = name
	pg.UID = types.UID(name + "-uid")
	pg.Spec.ParentCompositePodGroupName = &parentName
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

func TestObjectOf(t *testing.T) {
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
			got, ok := objectOf[*v1.Pod](tc.obj)
			if (got != nil && ok) != tc.want {
				t.Errorf("objectOf() returned pod=%v, ok=%v, want non-nil=%v", got, ok, tc.want)
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
				queue: workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[queueKey]()),
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
			pgInformer := informerFactory.Scheduling().V1beta1().PodGroups()
			cpgInformer := informerFactory.Scheduling().V1alpha3().CompositePodGroups()
			podInformer := informerFactory.Core().V1().Pods()

			ctrl, err := NewPodGroupProtectionController(klog.FromContext(ctx), pgInformer, cpgInformer, podInformer, client, true)
			if err != nil {
				t.Fatalf("unexpected error creating controller: %v", err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			go ctrl.Run(ctx, 1)

			// In order to reduce test flakiness, make sure that the pod-to-delete is visible in the client set. Create a dummy pod to "warm up" the watch pipe.
			// Since it's created after LIST (WaitForCacheSync), the informer must see it via WATCH.
			syncPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "sync-pod", Namespace: defaultNS}}
			_, err = client.CoreV1().Pods(defaultNS).Create(ctx, syncPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("creating sync pod: %v", err)
			}

			err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				_, err = podInformer.Lister().Pods(defaultNS).Get(syncPod.Name)
				return err == nil, nil
			})
			if err != nil {
				t.Fatalf("timed out waiting for informer to see the sync pod: %v", err)
			}

			if test.podToDelete != "" {
				if err := client.CoreV1().Pods(defaultNS).Delete(ctx, test.podToDelete, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("deleting pod: %v", err)
				}
			}

			if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				pg, err := client.SchedulingV1beta1().PodGroups(defaultNS).Get(ctx, defaultPGName, metav1.GetOptions{})
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

func TestCompositePodGroupProtectionController(t *testing.T) {
	cpgRootName := "cpg-root"
	cpgChildName := "cpg-child"
	pgChildName := "pg-child"

	tests := []struct {
		name            string
		initialObjects  []runtime.Object
		pgToDelete      string
		cpgToDelete     string
		cpgToCheck      string
		expectFinalizer bool
	}{
		{
			name:            "deleted CompositePodGroup with finalizer, no children, finalizer is removed",
			initialObjects:  []runtime.Object{deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name(cpgRootName).Namespace(defaultNS).UID(cpgRootName + "-uid").Obj()))},
			cpgToCheck:      cpgRootName,
			expectFinalizer: false,
		},
		{
			name:            "deleted CompositePodGroup with finalizer, child PodGroup exists, finalizer is kept",
			initialObjects:  []runtime.Object{deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name(cpgRootName).Namespace(defaultNS).UID(cpgRootName + "-uid").Obj())), st.MakePodGroup().Name(pgChildName).Namespace(defaultNS).UID(types.UID(pgChildName + "-uid")).ParentCompositePodGroup(cpgRootName).Obj()},
			cpgToCheck:      cpgRootName,
			expectFinalizer: true,
		},
		{
			name:            "deleted CompositePodGroup with finalizer, child CompositePodGroup exists, finalizer is kept",
			initialObjects:  []runtime.Object{deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name(cpgRootName).Namespace(defaultNS).UID(cpgRootName + "-uid").Obj())), st.MakeCompositePodGroup().Name(cpgChildName).Namespace(defaultNS).UID(cpgChildName + "-uid").ParentCompositePodGroup(cpgRootName).Obj()},
			cpgToCheck:      cpgRootName,
			expectFinalizer: true,
		},
		{
			name:            "deleted CompositePodGroup with finalizer, child PodGroup is deleted, finalizer is removed",
			initialObjects:  []runtime.Object{deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name(cpgRootName).Namespace(defaultNS).UID(cpgRootName + "-uid").Obj())), st.MakePodGroup().Name(pgChildName).Namespace(defaultNS).UID(types.UID(pgChildName + "-uid")).ParentCompositePodGroup(cpgRootName).Obj()},
			pgToDelete:      pgChildName,
			cpgToCheck:      cpgRootName,
			expectFinalizer: false,
		},
		{
			name:            "deleted CompositePodGroup with finalizer, child CompositePodGroup is deleted, finalizer is removed",
			initialObjects:  []runtime.Object{deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name(cpgRootName).Namespace(defaultNS).UID(cpgRootName + "-uid").Obj())), st.MakeCompositePodGroup().Name(cpgChildName).Namespace(defaultNS).UID(cpgChildName + "-uid").ParentCompositePodGroup(cpgRootName).Obj()},
			cpgToDelete:     cpgChildName,
			cpgToCheck:      cpgRootName,
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
			pgInformer := informerFactory.Scheduling().V1beta1().PodGroups()
			cpgInformer := informerFactory.Scheduling().V1alpha3().CompositePodGroups()
			podInformer := informerFactory.Core().V1().Pods()

			ctrl, err := NewPodGroupProtectionController(klog.FromContext(ctx), pgInformer, cpgInformer, podInformer, client, true)
			if err != nil {
				t.Fatalf("unexpected error creating controller: %v", err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			go ctrl.Run(ctx, 1)

			// In order to reduce test flakiness, make sure that the object-to-delete is visible in the client set. Create a dummy CPG and PG to "warm up" the watch pipe.
			// Since they are created after LIST (WaitForCacheSync), the informer must see them via WATCH before any deletions occur.
			syncCPG := st.MakeCompositePodGroup().Name("sync-cpg").Namespace(defaultNS).Obj()
			if _, err := client.SchedulingV1alpha3().CompositePodGroups(defaultNS).Create(ctx, syncCPG, metav1.CreateOptions{}); err != nil {
				t.Fatalf("creating sync cpg: %v", err)
			}
			if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				_, err := cpgInformer.Lister().CompositePodGroups(defaultNS).Get(syncCPG.Name)
				return err == nil, nil
			}); err != nil {
				t.Fatalf("timed out waiting for informer to see the sync cpg: %v", err)
			}

			syncPG := st.MakePodGroup().Name("sync-pg").Namespace(defaultNS).Obj()
			if _, err := client.SchedulingV1beta1().PodGroups(defaultNS).Create(ctx, syncPG, metav1.CreateOptions{}); err != nil {
				t.Fatalf("creating sync pg: %v", err)
			}
			if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				_, err := pgInformer.Lister().PodGroups(defaultNS).Get(syncPG.Name)
				return err == nil, nil
			}); err != nil {
				t.Fatalf("timed out waiting for informer to see the sync pg: %v", err)
			}

			if test.pgToDelete != "" {
				if err := client.SchedulingV1beta1().PodGroups(defaultNS).Delete(ctx, test.pgToDelete, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("deleting pod group: %v", err)
				}
			}

			if test.cpgToDelete != "" {
				if err := client.SchedulingV1alpha3().CompositePodGroups(defaultNS).Delete(ctx, test.cpgToDelete, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("deleting composite pod group: %v", err)
				}
			}

			if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				cpg, err := client.SchedulingV1alpha3().CompositePodGroups(defaultNS).Get(ctx, test.cpgToCheck, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return !test.expectFinalizer, nil
				}
				if err != nil {
					return false, err
				}
				hasFinalizer := slices.Contains(cpg.Finalizers, scheduling.CompositePodGroupProtectionFinalizer)
				return hasFinalizer == test.expectFinalizer, nil
			}); err != nil {
				t.Fatalf("timed out waiting for expected CPG finalizer state (want present=%v): %v", test.expectFinalizer, err)
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

func cpg(name string) *schedulingv1alpha3.CompositePodGroup {
	return &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: defaultNS,
			UID:       types.UID(name + "-uid"),
		},
	}
}

func cpgWithParent(name, parentName string) *schedulingv1alpha3.CompositePodGroup {
	group := cpg(name)
	group.Spec.ParentCompositePodGroupName = &parentName
	return group
}

func TestHasActivePods(t *testing.T) {
	tests := map[string]struct {
		pods []runtime.Object
		want bool
	}{
		"no pods": {
			want: false,
		},
		"active pod in cache referencing PodGroup": {
			pods: []runtime.Object{
				podForPG("pod-1", defaultPGName),
			},
			want: true,
		},
		"only terminated pods in cache": {
			pods: []runtime.Object{
				terminatedPod(podForPG("pod-1", defaultPGName), v1.PodSucceeded),
				terminatedPod(podForPG("pod-2", defaultPGName), v1.PodFailed),
			},
			want: false,
		},
		"mix of active and terminated in cache": {
			pods: []runtime.Object{
				podForPG("pod-active", defaultPGName),
				terminatedPod(podForPG("pod-done", defaultPGName), v1.PodSucceeded),
			},
			want: true,
		},
		"pods in cache referencing different PodGroup": {
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

func TestHasChildGroups(t *testing.T) {
	tests := map[string]struct {
		cachedPGs                  []runtime.Object
		cachedCPGs                 []runtime.Object
		isCompositePodGroupEnabled bool
		want                       bool
	}{
		"no children": {
			isCompositePodGroupEnabled: true,
			want:                       false,
		},
		"child PodGroup in cache, CPG feature enabled": {
			cachedPGs: []runtime.Object{
				podGroupWithParent("child-pg", "parent-cpg"),
			},
			isCompositePodGroupEnabled: true,
			want:                       true,
		},
		"child PodGroup in cache, CPG feature disabled": {
			cachedPGs: []runtime.Object{
				podGroupWithParent("child-pg", "parent-cpg"),
			},
			isCompositePodGroupEnabled: false,
			want:                       true,
		},
		"child CompositePodGroup in cache, CPG feature enabled": {
			cachedCPGs: []runtime.Object{
				cpgWithParent("child-cpg", "parent-cpg"),
			},
			isCompositePodGroupEnabled: true,
			want:                       true,
		},
		"child CompositePodGroup in cache, CPG feature disabled": {
			cachedCPGs: []runtime.Object{
				cpgWithParent("child-cpg", "parent-cpg"),
			},
			isCompositePodGroupEnabled: false,
			want:                       false,
		},
		"children referencing different parent": {
			cachedPGs: []runtime.Object{
				podGroupWithParent("other-child-pg", "different-parent"),
			},
			cachedCPGs: []runtime.Object{
				cpgWithParent("other-child-cpg", "different-parent"),
			},
			isCompositePodGroupEnabled: true,
			want:                       false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			pgIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if err := addChildIndexer(pgIndexer, childPodGroupParentCompositeGroupIndex, podGroupParent); err != nil {
				t.Fatalf("unexpected error adding PG indexer: %v", err)
			}
			for _, obj := range tc.cachedPGs {
				_ = pgIndexer.Add(obj)
			}

			var cpgIndexer cache.Indexer
			if tc.isCompositePodGroupEnabled {
				cpgIndexer = cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
				if err := addChildIndexer(cpgIndexer, childCompositePodGroupParentCompositeGroupIndex, compositePodGroupParent); err != nil {
					t.Fatalf("unexpected error adding CPG indexer: %v", err)
				}
				for _, obj := range tc.cachedCPGs {
					_ = cpgIndexer.Add(obj)
				}
			}

			ctrl := &Controller{
				podGroupIndexer:            pgIndexer,
				compositePodGroupIndexer:   cpgIndexer,
				isCompositePodGroupEnabled: tc.isCompositePodGroupEnabled,
			}

			parentCPG := cpg("parent-cpg")
			got, err := ctrl.hasChildGroups(context.Background(), parentCPG)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("hasChildGroups() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHandlePodGroupUpdate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	tests := map[string]struct {
		old                        any
		new                        any
		isCompositePodGroupEnabled bool
		wantSize                   int
	}{
		"both old and new are nil -> not enqueued, no panic": {
			old:                        nil,
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"non-PodGroup object -> not enqueued, no panic": {
			old:                        nil,
			new:                        &v1.ConfigMap{},
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"invalid tombstone -> not enqueued, no panic": {
			old: cache.DeletedFinalStateUnknown{
				Key: "default/pg",
				Obj: &v1.ConfigMap{},
			},
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"PodGroup without finalizer, not deleting/not enqueued": {
			new:                        podGroup(),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"PodGroup is deletion candidate, CPG enabled -> enqueued": {
			new:                        deletedPodGroup(withFinalizer(podGroup())),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"PodGroup is deletion candidate, CPG disabled -> enqueued": {
			new:                        deletedPodGroup(withFinalizer(podGroup())),
			isCompositePodGroupEnabled: false,
			wantSize:                   1,
		},
		"PodGroup has finalizer, not deleting -> not enqueued": {
			new:                        withFinalizer(podGroup()),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"PodGroup with parent CompositePodGroup on add -> parent not enqueued": {
			new:                        podGroupWithParent("pg-1", "cpg-parent"),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"PodGroup with parent CompositePodGroup on delete, CPG enabled -> enqueued for parent": {
			old:                        podGroupWithParent("pg-1", "cpg-parent"),
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"PodGroup with parent CompositePodGroup on delete, CPG disabled -> parent not enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent"),
			new:                        nil,
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
		"PodGroup with parent CompositePodGroup on update (same UID) -> parent not enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent"),
			new:                        podGroupWithParent("pg-1", "cpg-parent"),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"PodGroup with parent CompositePodGroup on update (same UID) and deletion candidate, CPG enabled -> only child enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent"),
			new:                        deletedPodGroup(withFinalizer(podGroupWithParent("pg-1", "cpg-parent"))),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"PodGroup with parent CompositePodGroup on update (same UID) and deletion candidate, CPG disabled -> only child enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent"),
			new:                        deletedPodGroup(withFinalizer(podGroupWithParent("pg-1", "cpg-parent"))),
			isCompositePodGroupEnabled: false,
			wantSize:                   1,
		},
		"PodGroup with parent CompositePodGroup on update (UID mismatch), CPG enabled -> parent enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent-1"),
			new:                        podGroupWithParent("pg-2", "cpg-parent-2"),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"PodGroup with parent CompositePodGroup on update (UID mismatch), CPG disabled -> parent not enqueued": {
			old:                        podGroupWithParent("pg-1", "cpg-parent-1"),
			new:                        podGroupWithParent("pg-2", "cpg-parent-2"),
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c := &Controller{
				isCompositePodGroupEnabled: tc.isCompositePodGroupEnabled,
				queue:                      workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[queueKey]()),
			}
			defer c.queue.ShutDown()
			c.handlePodGroupUpdate(logger, tc.old, tc.new)
			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}

func TestHandleCompositePodGroupUpdate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	cpgWithParent := func(name, parentName string) *schedulingv1alpha3.CompositePodGroup {
		cpg := st.MakeCompositePodGroup().Name(name).Namespace(defaultNS).UID(name + "-uid").ParentCompositePodGroup(parentName).Obj()
		return cpg
	}

	tests := map[string]struct {
		old                        any
		new                        any
		isCompositePodGroupEnabled bool
		wantSize                   int
	}{
		"both old and new are nil -> not enqueued, no panic": {
			old:                        nil,
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"non-CPG object -> not enqueued, no panic": {
			old:                        nil,
			new:                        &v1.ConfigMap{},
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"invalid tombstone -> not enqueued, no panic": {
			old: cache.DeletedFinalStateUnknown{
				Key: "default/cpg",
				Obj: &v1.ConfigMap{},
			},
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"CPG without finalizer, not deleting -> not enqueued": {
			new:                        st.MakeCompositePodGroup().Name("cpg-1").Namespace(defaultNS).Obj(),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"CPG is deletion candidate, CPG enabled -> enqueued": {
			new:                        deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name("cpg-1").Namespace(defaultNS).Obj())),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"CPG is deletion candidate, CPG disabled -> not enqueued": {
			new:                        deletedCompositePodGroup(withCPGFinalizer(st.MakeCompositePodGroup().Name("cpg-1").Namespace(defaultNS).Obj())),
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
		"CPG with parent CompositePodGroup on add -> parent not enqueued": {
			new:                        cpgWithParent("cpg-child", "cpg-parent"),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"CPG with parent CompositePodGroup on delete, CPG enabled -> enqueued for parent": {
			old:                        cpgWithParent("cpg-child", "cpg-parent"),
			new:                        nil,
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"CPG with parent CompositePodGroup on delete, CPG disabled -> not enqueued": {
			old:                        cpgWithParent("cpg-child", "cpg-parent"),
			new:                        nil,
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
		"CPG with parent CompositePodGroup on update (same UID) -> parent not enqueued": {
			old:                        cpgWithParent("cpg-child", "cpg-parent"),
			new:                        cpgWithParent("cpg-child", "cpg-parent"),
			isCompositePodGroupEnabled: true,
			wantSize:                   0,
		},
		"CPG with parent CompositePodGroup on update (same UID) and deletion candidate, CPG enabled -> only child enqueued": {
			old:                        cpgWithParent("cpg-child", "cpg-parent"),
			new:                        deletedCompositePodGroup(withCPGFinalizer(cpgWithParent("cpg-child", "cpg-parent"))),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"CPG with parent CompositePodGroup on update (same UID) and deletion candidate, CPG disabled -> not enqueued": {
			old:                        cpgWithParent("cpg-child", "cpg-parent"),
			new:                        deletedCompositePodGroup(withCPGFinalizer(cpgWithParent("cpg-child", "cpg-parent"))),
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
		"CPG with parent CompositePodGroup on update (UID mismatch), CPG enabled -> parent enqueued": {
			old:                        cpgWithParent("cpg-child-1", "cpg-parent-1"),
			new:                        cpgWithParent("cpg-child-2", "cpg-parent-2"),
			isCompositePodGroupEnabled: true,
			wantSize:                   1,
		},
		"CPG with parent CompositePodGroup on update (UID mismatch), CPG disabled -> not enqueued": {
			old:                        cpgWithParent("cpg-child-1", "cpg-parent-1"),
			new:                        cpgWithParent("cpg-child-2", "cpg-parent-2"),
			isCompositePodGroupEnabled: false,
			wantSize:                   0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c := &Controller{
				isCompositePodGroupEnabled: tc.isCompositePodGroupEnabled,
				queue:                      workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[queueKey]()),
			}
			defer c.queue.ShutDown()
			c.handleCompositePodGroupUpdate(logger, tc.old, tc.new)
			if c.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", c.queue.Len(), tc.wantSize)
			}
		})
	}
}
