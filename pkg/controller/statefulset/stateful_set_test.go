/*
Copyright 2016 The Kubernetes Authors.

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

package statefulset

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/history"
	"k8s.io/kubernetes/pkg/features"
)

var parentKind = apps.SchemeGroupVersion.WithKind("StatefulSet")

func alwaysReady() bool { return true }

func TestStatefulSetControllerCreates(t *testing.T) {
	set := newStatefulSet(3)
	logger, ctx := ktesting.NewTestContext(t)
	ssc, spc, om, _ := newFakeStatefulSetController(ctx, set)
	if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Errorf("set.Status.Replicas = %v; want 3", set.Status.Replicas)
	}
}

func TestStatefulSetControllerDeletes(t *testing.T) {
	set := newStatefulSet(3)
	logger, ctx := ktesting.NewTestContext(t)
	ssc, spc, om, _ := newFakeStatefulSetController(ctx, set)
	if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Errorf("set.Status.Replicas = %v; want 3", set.Status.Replicas)
	}
	*set.Spec.Replicas = 0
	if err := scaleDownStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn down StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 0 {
		t.Errorf("set.Status.Replicas = %v; want 0", set.Status.Replicas)
	}
}

func TestStatefulSetControllerRespectsTermination(t *testing.T) {
	set := newStatefulSet(3)
	logger, ctx := ktesting.NewTestContext(t)
	ssc, spc, om, _ := newFakeStatefulSetController(ctx, set)
	if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Errorf("set.Status.Replicas = %v; want 3", set.Status.Replicas)
	}
	_, err := om.addTerminatingPod(set, 3)
	if err != nil {
		t.Error(err)
	}
	pods, err := om.addTerminatingPod(set, 4)
	if err != nil {
		t.Error(err)
	}
	ssc.syncStatefulSet(ctx, set, pods)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err = om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if len(pods) != 5 {
		t.Error("StatefulSet does not respect termination")
	}
	sort.Sort(ascendingOrdinal(pods))
	spc.DeleteStatefulPod(set, pods[3])
	spc.DeleteStatefulPod(set, pods[4])
	*set.Spec.Replicas = 0
	if err := scaleDownStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn down StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 0 {
		t.Errorf("set.Status.Replicas = %v; want 0", set.Status.Replicas)
	}
}

func TestStatefulSetControllerBlocksScaling(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	set := newStatefulSet(3)
	ssc, spc, om, _ := newFakeStatefulSetController(ctx, set)
	if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := om.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Errorf("set.Status.Replicas = %v; want 3", set.Status.Replicas)
	}
	*set.Spec.Replicas = 5
	fakeResourceVersion(set)
	om.setsIndexer.Update(set)
	_, err := om.setPodTerminated(set, 0)
	if err != nil {
		t.Error("Failed to set pod terminated at ordinal 0")
	}
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if len(pods) != 3 {
		t.Error("StatefulSet does not block scaling")
	}
	sort.Sort(ascendingOrdinal(pods))
	spc.DeleteStatefulPod(set, pods[0])
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	pods, err = om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if len(pods) != 3 {
		t.Error("StatefulSet does not resume when terminated Pod is removed")
	}
}

func TestStatefulSetControllerDeletionTimestamp(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	set := newStatefulSet(3)
	set.DeletionTimestamp = new(metav1.Time)
	ssc, _, om, _ := newFakeStatefulSetController(ctx, set)

	om.setsIndexer.Add(set)

	// Force a sync. It should not try to create any Pods.
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)

	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}
	pods, err := om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(pods), 0; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerDeletionTimestampRace(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	set := newStatefulSet(3)
	// The bare client says it IS deleted.
	set.DeletionTimestamp = new(metav1.Time)
	ssc, _, om, ssh := newFakeStatefulSetController(ctx, set)

	// The lister (cache) says it's NOT deleted.
	set2 := *set
	set2.DeletionTimestamp = nil
	om.setsIndexer.Add(&set2)

	// The recheck occurs in the presence of a matching orphan.
	pod := newStatefulSetPod(set, 1)
	pod.OwnerReferences = nil
	om.podsIndexer.Add(pod)
	set.Status.CollisionCount = new(int32)
	revision, err := newRevision(set, 1, set.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}
	revision.OwnerReferences = nil
	_, err = ssh.CreateControllerRevision(set, revision, set.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}

	// Force a sync. It should not try to create any Pods.
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)

	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}
	pods, err := om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}

	// It should not adopt pods.
	for _, pod := range pods {
		if len(pod.OwnerReferences) > 0 {
			t.Errorf("unexpected pod owner references: %v", pod.OwnerReferences)
		}
	}

	// It should not adopt revisions.
	revisions, err := ssh.ListControllerRevisions(set, selector)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(revisions), 1; got != want {
		t.Errorf("len(revisions) = %v, want %v", got, want)
	}
	for _, revision := range revisions {
		if len(revision.OwnerReferences) > 0 {
			t.Errorf("unexpected revision owner references: %v", revision.OwnerReferences)
		}
	}
}

func TestStatefulSetControllerAddPod(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	pod1 := newStatefulSetPod(set1, 0)
	pod2 := newStatefulSetPod(set2, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)

	ssc.addPod(logger, pod1)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set1); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}
	ssc.queue.Done(key)

	ssc.addPod(logger, pod2)
	key, done = ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set2); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}
	ssc.queue.Done(key)
}

func TestStatefulSetControllerAddPodOrphan(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	set3 := newStatefulSet(3)
	set3.Name = "foo3"
	set3.Spec.Selector.MatchLabels = map[string]string{"foo3": "bar"}
	pod := newStatefulSetPod(set1, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)
	om.setsIndexer.Add(set3)

	// Make pod an orphan. Expect matching sets to be queued.
	pod.OwnerReferences = nil
	ssc.addPod(logger, pod)
	if got, want := ssc.queue.Len(), 2; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerAddPodNoSet(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, _, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	ssc.addPod(logger, pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePod(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod1 := newStatefulSetPod(set1, 0)
	pod2 := newStatefulSetPod(set2, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)

	prev := *pod1
	fakeResourceVersion(pod1)
	ssc.updatePod(logger, &prev, pod1)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set1); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}

	prev = *pod2
	fakeResourceVersion(pod2)
	ssc.updatePod(logger, &prev, pod2)
	key, done = ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set2); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerUpdatePodWithNoSet(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, _, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	prev := *pod
	fakeResourceVersion(pod)
	ssc.updatePod(logger, &prev, pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePodWithSameVersion(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	om.setsIndexer.Add(set)
	ssc.updatePod(logger, pod, pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePodOrphanWithNewLabels(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	pod.OwnerReferences = nil
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	om.setsIndexer.Add(set)
	om.setsIndexer.Add(set2)
	clone := *pod
	clone.Labels = map[string]string{"foo2": "bar2"}
	fakeResourceVersion(&clone)
	ssc.updatePod(logger, &clone, pod)
	if got, want := ssc.queue.Len(), 2; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerUpdatePodChangeControllerRef(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod := newStatefulSetPod(set, 0)
	pod2 := newStatefulSetPod(set2, 0)
	om.setsIndexer.Add(set)
	om.setsIndexer.Add(set2)
	clone := *pod
	clone.OwnerReferences = pod2.OwnerReferences
	fakeResourceVersion(&clone)
	ssc.updatePod(logger, &clone, pod)
	if got, want := ssc.queue.Len(), 2; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerUpdatePodRelease(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod := newStatefulSetPod(set, 0)
	om.setsIndexer.Add(set)
	om.setsIndexer.Add(set2)
	clone := *pod
	clone.OwnerReferences = nil
	fakeResourceVersion(&clone)
	ssc.updatePod(logger, pod, &clone)
	if got, want := ssc.queue.Len(), 2; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerDeletePod(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod1 := newStatefulSetPod(set1, 0)
	pod2 := newStatefulSetPod(set2, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)

	ssc.deletePod(logger, pod1)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set1); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}

	ssc.deletePod(logger, pod2)
	key, done = ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set2); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerDeletePodOrphan(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod1 := newStatefulSetPod(set1, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)

	pod1.OwnerReferences = nil
	ssc.deletePod(logger, pod1)
	if got, want := ssc.queue.Len(), 0; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestStatefulSetControllerDeletePodTombstone(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	om.setsIndexer.Add(set)
	tombstoneKey, _ := controller.KeyFunc(pod)
	tombstone := cache.DeletedFinalStateUnknown{Key: tombstoneKey, Obj: pod}
	ssc.deletePod(logger, tombstone)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerGetStatefulSetsForPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx)
	set1 := newStatefulSet(3)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	pod := newStatefulSetPod(set1, 0)
	om.setsIndexer.Add(set1)
	om.setsIndexer.Add(set2)
	om.podsIndexer.Add(pod)
	sets := ssc.getStatefulSetsForPod(pod)
	if got, want := len(sets), 2; got != want {
		t.Errorf("len(sets) = %v, want %v", got, want)
	}
}

func TestGetPodsForStatefulSetAdopt(t *testing.T) {
	set := newStatefulSet(5)
	pod1 := newStatefulSetPod(set, 1)
	// pod2 is an orphan with matching labels and name.
	pod2 := newStatefulSetPod(set, 2)
	pod2.OwnerReferences = nil
	// pod3 has wrong labels.
	pod3 := newStatefulSetPod(set, 3)
	pod3.OwnerReferences = nil
	pod3.Labels = nil
	// pod4 has wrong name.
	pod4 := newStatefulSetPod(set, 4)
	pod4.OwnerReferences = nil
	pod4.Name = "x" + pod4.Name

	_, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx, set, pod1, pod2, pod3, pod4)

	om.podsIndexer.Add(pod1)
	om.podsIndexer.Add(pod2)
	om.podsIndexer.Add(pod3)
	om.podsIndexer.Add(pod4)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}
	pods, err := ssc.getPodsForStatefulSet(context.TODO(), set, selector)
	if err != nil {
		t.Fatalf("getPodsForStatefulSet() error: %v", err)
	}
	got := sets.NewString()
	for _, pod := range pods {
		got.Insert(pod.Name)
	}
	// pod2 should be claimed, pod3 and pod4 ignored
	want := sets.NewString(pod1.Name, pod2.Name)
	if !got.Equal(want) {
		t.Errorf("getPodsForStatefulSet() = %v, want %v", got, want)
	}
}

func TestAdoptOrphanRevisions(t *testing.T) {
	ss1 := newStatefulSetWithLabels(3, "ss1", types.UID("ss1"), map[string]string{"foo": "bar"})
	ss1.Status.CollisionCount = new(int32)
	ss1Rev1, err := history.NewControllerRevision(ss1, parentKind, ss1.Spec.Template.Labels, rawTemplate(&ss1.Spec.Template), 1, ss1.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}
	ss1Rev1.Namespace = ss1.Namespace
	ss1.Spec.Template.Annotations = make(map[string]string)
	ss1.Spec.Template.Annotations["ss1"] = "ss1"
	ss1Rev2, err := history.NewControllerRevision(ss1, parentKind, ss1.Spec.Template.Labels, rawTemplate(&ss1.Spec.Template), 2, ss1.Status.CollisionCount)
	if err != nil {
		t.Fatal(err)
	}
	ss1Rev2.Namespace = ss1.Namespace
	ss1Rev2.OwnerReferences = []metav1.OwnerReference{}

	_, ctx := ktesting.NewTestContext(t)
	ssc, _, om, _ := newFakeStatefulSetController(ctx, ss1, ss1Rev1, ss1Rev2)

	om.revisionsIndexer.Add(ss1Rev1)
	om.revisionsIndexer.Add(ss1Rev2)

	err = ssc.adoptOrphanRevisions(context.TODO(), ss1)
	if err != nil {
		t.Errorf("adoptOrphanRevisions() error: %v", err)
	}

	if revisions, err := ssc.control.ListRevisions(ss1); err != nil {
		t.Errorf("ListRevisions() error: %v", err)
	} else {
		var adopted bool
		for i := range revisions {
			if revisions[i].Name == ss1Rev2.Name && metav1.GetControllerOf(revisions[i]) != nil {
				adopted = true
			}
		}
		if !adopted {
			t.Error("adoptOrphanRevisions() not adopt orphan revisions")
		}
	}
}

func TestGetPodsForStatefulSetRelease(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	set := newStatefulSet(3)
	ssc, _, om, _ := newFakeStatefulSetController(ctx, set)
	pod1 := newStatefulSetPod(set, 1)
	// pod2 is owned but has wrong name.
	pod2 := newStatefulSetPod(set, 2)
	pod2.Name = "x" + pod2.Name
	// pod3 is owned but has wrong labels.
	pod3 := newStatefulSetPod(set, 3)
	pod3.Labels = nil
	// pod4 is an orphan that doesn't match.
	pod4 := newStatefulSetPod(set, 4)
	pod4.OwnerReferences = nil
	pod4.Labels = nil

	om.podsIndexer.Add(pod1)
	om.podsIndexer.Add(pod2)
	om.podsIndexer.Add(pod3)
	om.podsIndexer.Add(pod4)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Fatal(err)
	}
	pods, err := ssc.getPodsForStatefulSet(context.TODO(), set, selector)
	if err != nil {
		t.Fatalf("getPodsForStatefulSet() error: %v", err)
	}
	got := sets.NewString()
	for _, pod := range pods {
		got.Insert(pod.Name)
	}

	// Expect only pod1 (pod2 and pod3 should be released, pod4 ignored).
	want := sets.NewString(pod1.Name)
	if !got.Equal(want) {
		t.Errorf("getPodsForStatefulSet() = %v, want %v", got, want)
	}
}

func TestOrphanedPodsWithPVCDeletePolicy(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)

	testFn := func(t *testing.T, scaledownPolicy, deletionPolicy apps.PersistentVolumeClaimRetentionPolicyType) {
		set := newStatefulSet(4)
		*set.Spec.Replicas = 2
		set.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled = scaledownPolicy
		set.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted = deletionPolicy
		_, ctx := ktesting.NewTestContext(t)
		ssc, _, om, _ := newFakeStatefulSetController(ctx, set)
		om.setsIndexer.Add(set)

		pods := []*v1.Pod{}
		pods = append(pods, newStatefulSetPod(set, 0))
		// pod1 is orphaned
		pods = append(pods, newStatefulSetPod(set, 1))
		pods[1].OwnerReferences = nil
		// pod2 is owned but has wrong name.
		pods = append(pods, newStatefulSetPod(set, 2))
		pods[2].Name = "x" + pods[2].Name

		ssc.kubeClient.(*fake.Clientset).PrependReactor("patch", "pods", func(action core.Action) (bool, runtime.Object, error) {
			patch := action.(core.PatchAction).GetPatch()
			target := action.(core.PatchAction).GetName()
			var pod *v1.Pod
			for _, p := range pods {
				if p.Name == target {
					pod = p
					break
				}
			}
			if pod == nil {
				t.Fatalf("Can't find patch target %s", target)
			}
			original, err := json.Marshal(pod)
			if err != nil {
				t.Fatalf("failed to marshal original pod %s: %v", pod.Name, err)
			}
			updated, err := strategicpatch.StrategicMergePatch(original, patch, v1.Pod{})
			if err != nil {
				t.Fatalf("failed to apply strategic merge patch %q on node %s: %v", patch, pod.Name, err)
			}
			if err := json.Unmarshal(updated, pod); err != nil {
				t.Fatalf("failed to unmarshal updated pod %s: %v", pod.Name, err)
			}

			return true, pod, nil
		})

		for _, pod := range pods {
			om.podsIndexer.Add(pod)
			claims := getPersistentVolumeClaims(set, pod)
			for _, claim := range claims {
				om.CreateClaim(&claim)
			}
		}

		for i := range pods {
			if _, err := om.setPodReady(set, i); err != nil {
				t.Errorf("%d: %v", i, err)
			}
			if _, err := om.setPodRunning(set, i); err != nil {
				t.Errorf("%d: %v", i, err)
			}
		}

		// First sync to manage orphaned pod, then set replicas.
		ssc.enqueueStatefulSet(set)
		fakeWorker(ssc)
		*set.Spec.Replicas = 0 // Put an ownerRef for all scale-down deleted PVCs.
		ssc.enqueueStatefulSet(set)
		fakeWorker(ssc)

		hasNamedOwnerRef := func(claim *v1.PersistentVolumeClaim, name string) bool {
			for _, ownerRef := range claim.GetOwnerReferences() {
				if ownerRef.Name == name {
					return true
				}
			}
			return false
		}
		verifyOwnerRefs := func(claim *v1.PersistentVolumeClaim, condemned bool) {
			podName := getClaimPodName(set, claim)
			const retain = apps.RetainPersistentVolumeClaimRetentionPolicyType
			const delete = apps.DeletePersistentVolumeClaimRetentionPolicyType
			switch {
			case scaledownPolicy == retain && deletionPolicy == retain:
				if hasNamedOwnerRef(claim, podName) || hasNamedOwnerRef(claim, set.Name) {
					t.Errorf("bad claim ownerRefs: %s: %v", claim.Name, claim.GetOwnerReferences())
				}
			case scaledownPolicy == retain && deletionPolicy == delete:
				if hasNamedOwnerRef(claim, podName) || !hasNamedOwnerRef(claim, set.Name) {
					t.Errorf("bad claim ownerRefs: %s: %v", claim.Name, claim.GetOwnerReferences())
				}
			case scaledownPolicy == delete && deletionPolicy == retain:
				if hasNamedOwnerRef(claim, podName) != condemned || hasNamedOwnerRef(claim, set.Name) {
					t.Errorf("bad claim ownerRefs: %s: %v", claim.Name, claim.GetOwnerReferences())
				}
			case scaledownPolicy == delete && deletionPolicy == delete:
				if hasNamedOwnerRef(claim, podName) != condemned || !hasNamedOwnerRef(claim, set.Name) {
					t.Errorf("bad claim ownerRefs: %s: %v", claim.Name, claim.GetOwnerReferences())
				}
			}
		}

		claims, _ := om.claimsLister.PersistentVolumeClaims(set.Namespace).List(labels.Everything())
		if len(claims) != len(pods) {
			t.Errorf("Unexpected number of claims: %d", len(claims))
		}
		for _, claim := range claims {
			// Only the first pod and the reclaimed orphan pod should have owner refs.
			switch claim.Name {
			case "datadir-foo-0", "datadir-foo-1":
				verifyOwnerRefs(claim, false)
			case "datadir-foo-2":
				if hasNamedOwnerRef(claim, getClaimPodName(set, claim)) || hasNamedOwnerRef(claim, set.Name) {
					t.Errorf("unexpected ownerRefs for %s: %v", claim.Name, claim.GetOwnerReferences())
				}
			default:
				t.Errorf("Unexpected claim %s", claim.Name)
			}
		}
	}
	policies := []apps.PersistentVolumeClaimRetentionPolicyType{
		apps.RetainPersistentVolumeClaimRetentionPolicyType,
		apps.DeletePersistentVolumeClaimRetentionPolicyType,
	}
	for _, scaledownPolicy := range policies {
		for _, deletionPolicy := range policies {
			testName := fmt.Sprintf("ScaleDown:%s/SetDeletion:%s", scaledownPolicy, deletionPolicy)
			t.Run(testName, func(t *testing.T) { testFn(t, scaledownPolicy, deletionPolicy) })
		}
	}
}

func TestStaleOwnerRefOnScaleup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)

	for _, policy := range []*apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
		{
			WhenScaled:  apps.DeletePersistentVolumeClaimRetentionPolicyType,
			WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
		},
		{
			WhenScaled:  apps.DeletePersistentVolumeClaimRetentionPolicyType,
			WhenDeleted: apps.DeletePersistentVolumeClaimRetentionPolicyType,
		},
	} {
		onPolicy := func(msg string, args ...interface{}) string {
			return fmt.Sprintf(fmt.Sprintf("(%s) %s", policy, msg), args...)
		}
		set := newStatefulSet(3)
		set.Spec.PersistentVolumeClaimRetentionPolicy = policy
		logger, ctx := ktesting.NewTestContext(t)
		ssc, spc, om, _ := newFakeStatefulSetController(ctx, set)
		if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
			t.Errorf(onPolicy("Failed to turn up StatefulSet : %s", err))
		}
		var err error
		if set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name); err != nil {
			t.Errorf(onPolicy("Could not get scaled up set: %v", err))
		}
		if set.Status.Replicas != 3 {
			t.Errorf(onPolicy("set.Status.Replicas = %v; want 3", set.Status.Replicas))
		}
		*set.Spec.Replicas = 2
		if err := scaleDownStatefulSetController(logger, set, ssc, spc, om); err != nil {
			t.Errorf(onPolicy("Failed to scale down StatefulSet : msg, %s", err))
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Errorf(onPolicy("Could not get scaled down StatefulSet: %v", err))
		}
		if set.Status.Replicas != 2 {
			t.Errorf(onPolicy("Failed to scale statefulset to 2 replicas"))
		}

		var claim *v1.PersistentVolumeClaim
		claim, err = om.claimsLister.PersistentVolumeClaims(set.Namespace).Get("datadir-foo-2")
		if err != nil {
			t.Errorf(onPolicy("Could not find expected pvc datadir-foo-2"))
		}
		refs := claim.GetOwnerReferences()
		if len(refs) != 1 {
			t.Errorf(onPolicy("Expected only one refs: %v", refs))
		}
		// Make the pod ref stale.
		for i := range refs {
			if refs[i].Name == "foo-2" {
				refs[i].UID = "stale"
				break
			}
		}
		claim.SetOwnerReferences(refs)
		if err = om.claimsIndexer.Update(claim); err != nil {
			t.Errorf(onPolicy("Could not update claim with new owner ref: %v", err))
		}

		*set.Spec.Replicas = 3
		// Until the stale PVC goes away, the scale up should never finish. Run 10 iterations, then delete the PVC.
		if err := scaleUpStatefulSetControllerBounded(logger, set, ssc, spc, om, 10); err != nil {
			t.Errorf(onPolicy("Failed attempt to scale StatefulSet back up: %v", err))
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Errorf(onPolicy("Could not get scaled down StatefulSet: %v", err))
		}
		if set.Status.Replicas != 2 {
			t.Errorf(onPolicy("Expected set to stay at two replicas"))
		}

		claim, err = om.claimsLister.PersistentVolumeClaims(set.Namespace).Get("datadir-foo-2")
		if err != nil {
			t.Errorf(onPolicy("Could not find expected pvc datadir-foo-2"))
		}
		refs = claim.GetOwnerReferences()
		if len(refs) != 1 {
			t.Errorf(onPolicy("Unexpected change to condemned pvc ownerRefs: %v", refs))
		}
		foundPodRef := false
		for i := range refs {
			if refs[i].UID == "stale" {
				foundPodRef = true
				break
			}
		}
		if !foundPodRef {
			t.Errorf(onPolicy("Claim ref unexpectedly changed: %v", refs))
		}
		if err = om.claimsIndexer.Delete(claim); err != nil {
			t.Errorf(onPolicy("Could not delete stale pvc: %v", err))
		}

		if err := scaleUpStatefulSetController(logger, set, ssc, spc, om); err != nil {
			t.Errorf(onPolicy("Failed to scale StatefulSet back up: %v", err))
		}
		set, err = om.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Errorf(onPolicy("Could not get scaled down StatefulSet: %v", err))
		}
		if set.Status.Replicas != 3 {
			t.Errorf(onPolicy("Failed to scale set back up once PVC was deleted"))
		}
	}
}

func newFakeStatefulSetController(ctx context.Context, initialObjects ...runtime.Object) (*StatefulSetController, *StatefulPodControl, *fakeObjectManager, history.Interface) {
	client := fake.NewSimpleClientset(initialObjects...)
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	om := newFakeObjectManager(informerFactory)
	spc := NewStatefulPodControlFromManager(om, &noopRecorder{})
	ssu := newFakeStatefulSetStatusUpdater(informerFactory.Apps().V1().StatefulSets())
	ssc := NewStatefulSetController(
		ctx,
		informerFactory.Core().V1().Pods(),
		informerFactory.Apps().V1().StatefulSets(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Apps().V1().ControllerRevisions(),
		client,
	)
	ssh := history.NewFakeHistory(informerFactory.Apps().V1().ControllerRevisions())
	ssc.podListerSynced = alwaysReady
	ssc.setListerSynced = alwaysReady
	ssc.control = NewDefaultStatefulSetControl(spc, ssu, ssh)

	return ssc, spc, om, ssh
}

func fakeWorker(ssc *StatefulSetController) {
	if obj, done := ssc.queue.Get(); !done {
		ssc.sync(context.TODO(), obj.(string))
		ssc.queue.Done(obj)
	}
}

func getPodAtOrdinal(pods []*v1.Pod, ordinal int) *v1.Pod {
	if 0 > ordinal || ordinal >= len(pods) {
		return nil
	}
	sort.Sort(ascendingOrdinal(pods))
	return pods[ordinal]
}

func scaleUpStatefulSetController(logger klog.Logger, set *apps.StatefulSet, ssc *StatefulSetController, spc *StatefulPodControl, om *fakeObjectManager) error {
	return scaleUpStatefulSetControllerBounded(logger, set, ssc, spc, om, -1)
}

func scaleUpStatefulSetControllerBounded(logger klog.Logger, set *apps.StatefulSet, ssc *StatefulSetController, spc *StatefulPodControl, om *fakeObjectManager, maxIterations int) error {
	om.setsIndexer.Add(set)
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	iterations := 0
	for (maxIterations < 0 || iterations < maxIterations) && set.Status.ReadyReplicas < *set.Spec.Replicas {
		iterations++
		pods, err := om.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			return err
		}
		ord := len(pods) - 1
		if pods, err = om.setPodPending(set, ord); err != nil {
			return err
		}
		pod := getPodAtOrdinal(pods, ord)
		ssc.addPod(logger, pod)
		fakeWorker(ssc)
		pod = getPodAtOrdinal(pods, ord)
		prev := *pod
		if pods, err = om.setPodRunning(set, ord); err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(logger, &prev, pod)
		fakeWorker(ssc)
		pod = getPodAtOrdinal(pods, ord)
		prev = *pod
		if pods, err = om.setPodReady(set, ord); err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(logger, &prev, pod)
		fakeWorker(ssc)
		if err := assertMonotonicInvariants(set, om); err != nil {
			return err
		}
		obj, _, err := om.setsIndexer.Get(set)
		if err != nil {
			return err
		}
		set = obj.(*apps.StatefulSet)

	}
	return assertMonotonicInvariants(set, om)
}

func scaleDownStatefulSetController(logger klog.Logger, set *apps.StatefulSet, ssc *StatefulSetController, spc *StatefulPodControl, om *fakeObjectManager) error {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	pods, err := om.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return err
	}
	ord := len(pods) - 1
	pod := getPodAtOrdinal(pods, ord)
	prev := *pod
	fakeResourceVersion(set)
	om.setsIndexer.Add(set)
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	pods, err = om.addTerminatingPod(set, ord)
	if err != nil {
		return err
	}
	pod = getPodAtOrdinal(pods, ord)
	ssc.updatePod(logger, &prev, pod)
	fakeWorker(ssc)
	spc.DeleteStatefulPod(set, pod)
	ssc.deletePod(logger, pod)
	fakeWorker(ssc)
	for set.Status.Replicas > *set.Spec.Replicas {
		pods, err = om.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			return err
		}

		ord := len(pods)
		pods, err = om.addTerminatingPod(set, ord)
		if err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(logger, &prev, pod)
		fakeWorker(ssc)
		spc.DeleteStatefulPod(set, pod)
		ssc.deletePod(logger, pod)
		fakeWorker(ssc)
		obj, _, err := om.setsIndexer.Get(set)
		if err != nil {
			return err
		}
		set = obj.(*apps.StatefulSet)

	}
	return assertMonotonicInvariants(set, om)
}

func rawTemplate(template *v1.PodTemplateSpec) runtime.RawExtension {
	buf := new(bytes.Buffer)
	enc := json.NewEncoder(buf)
	if err := enc.Encode(template); err != nil {
		panic(err)
	}
	return runtime.RawExtension{Raw: buf.Bytes()}
}
