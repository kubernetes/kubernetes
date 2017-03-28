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
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
)

func TestStatefulSetControllerCreates(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	if err := scaleUpStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
}

func TestStatefulSetControllerDeletes(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	if err := scaleUpStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
	*set.Spec.Replicas = 0
	if err := scaleDownStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn down StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 0 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
}

func TestStatefulSetControllerRespectsTermination(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	if err := scaleUpStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
	pods, err := spc.addTerminatedPod(set, 3)
	if err != nil {
		t.Error(err)
	}
	pods, err = spc.addTerminatedPod(set, 4)
	if err != nil {
		t.Error(err)
	}
	ssc.syncStatefulSet(set, pods)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
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
	if err := scaleDownStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn down StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 0 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
}

func TestStatefulSetControllerBlocksScaling(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	if err := scaleUpStatefulSetController(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	if obj, _, err := spc.setsIndexer.Get(set); err != nil {
		t.Error(err)
	} else {
		set = obj.(*apps.StatefulSet)
	}
	if set.Status.Replicas != 3 {
		t.Error("Falied to scale statefulset to 3 replicas")
	}
	*set.Spec.Replicas = 5
	fakeResourceVersion(set)
	spc.setsIndexer.Update(set)
	pods, err := spc.setPodTerminated(set, 0)
	if err != nil {
		t.Error("Failed to set pod terminated at ordinal 0")
	}
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
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
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if len(pods) != 3 {
		t.Error("StatefulSet does not resume when terminated Pod is removed")
	}
}

func TestStateSetControllerAddPod(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	ssc.addPod(pod)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStateSetControllerAddPodNoSet(t *testing.T) {
	ssc, _ := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	ssc.addPod(pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePod(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	prev := *pod
	fakeResourceVersion(pod)
	ssc.updatePod(&prev, pod)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerUpdatePodWithNoSet(t *testing.T) {
	ssc, _ := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	prev := *pod
	fakeResourceVersion(pod)
	ssc.updatePod(&prev, pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePodWithSameVersion(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	ssc.updatePod(pod, pod)
	ssc.queue.ShutDown()
	key, _ := ssc.queue.Get()
	if key != nil {
		t.Errorf("StatefulSet enqueued key for Pod with no Set %s", key)
	}
}

func TestStatefulSetControllerUpdatePodWithNewLabels(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	set2.Spec.Selector.MatchLabels = map[string]string{"foo2": "bar2"}
	set2.Spec.Template.Labels = map[string]string{"foo2": "bar2"}
	spc.setsIndexer.Add(set)
	spc.setsIndexer.Add(set2)
	clone := *pod
	clone.Labels = map[string]string{"foo2": "bar2"}
	fakeResourceVersion(&clone)
	ssc.updatePod(pod, &clone)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set2); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
	key, done = ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerDeletePod(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	ssc.deletePod(pod)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
}
func TestStatefulSetControllerDeletePodTombstone(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	tombstoneKey, _ := controller.KeyFunc(pod)
	tombstone := cache.DeletedFinalStateUnknown{Key: tombstoneKey, Obj: pod}
	ssc.deletePod(tombstone)
	key, done := ssc.queue.Get()
	if key == nil || done {
		t.Error("Failed to enqueue StatefulSet")
	} else if key, ok := key.(string); !ok {
		t.Error("Key is not a string")
	} else if expectedKey, _ := controller.KeyFunc(set); expectedKey != key {
		t.Errorf("Expected StatefulSet key %s found %s", expectedKey, key)
	}
}

func TestStatefulSetControllerGetStatefulSetForPod(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	spc.setsIndexer.Add(set)
	spc.podsIndexer.Add(pod)
	if set := ssc.getStatefulSetForPod(pod); set == nil {
		t.Error("Failed to get StatefulSet for Pod ")
	}
}

func TestStatefulSetControllerGetStatefulSetForPodOverlapping(t *testing.T) {
	ssc, spc := newFakeStatefulSetController()
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	set2 := newStatefulSet(3)
	set2.Name = "foo2"
	set3 := newStatefulSet(3)
	set3.Name = "foo3"
	set3.CreationTimestamp.Add(1 * time.Second)
	spc.setsIndexer.Add(set3)
	spc.setsIndexer.Add(set2)
	spc.setsIndexer.Add(set)
	spc.podsIndexer.Add(pod)
	if found := ssc.getStatefulSetForPod(pod); found == nil {
		t.Error("Failed to get StatefulSet for Pod")
	} else if found.Name != set.Name {
		t.Errorf("Returned wrong StatefulSet %s for Pod", set.Name)
	}
}

func newFakeStatefulSetController() (*StatefulSetController, *fakeStatefulPodControl) {
	fpc := newFakeStatefulPodControl()
	ssc := &StatefulSetController{
		kubeClient:     nil,
		podStoreSynced: func() bool { return true },
		setStore:       fpc.setsLister,
		podStore:       fpc.podsLister,
		control:        NewDefaultStatefulSetControl(fpc),
		queue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "statefulset"),
	}
	return ssc, fpc
}

func fakeWorker(ssc *StatefulSetController) {
	if obj, done := ssc.queue.Get(); !done {
		ssc.sync(obj.(string))
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

func scaleUpStatefulSetController(set *apps.StatefulSet, ssc *StatefulSetController, spc *fakeStatefulPodControl) error {
	spc.setsIndexer.Add(set)
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	for set.Status.Replicas < *set.Spec.Replicas {
		pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
		ord := len(pods) - 1
		pod := getPodAtOrdinal(pods, ord)
		if pods, err = spc.setPodPending(set, ord); err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.addPod(pod)
		fakeWorker(ssc)
		pod = getPodAtOrdinal(pods, ord)
		prev := *pod
		if pods, err = spc.setPodRunning(set, ord); err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(&prev, pod)
		fakeWorker(ssc)
		pod = getPodAtOrdinal(pods, ord)
		prev = *pod
		if pods, err = spc.setPodReady(set, ord); err != nil {
			return err
		}
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(&prev, pod)
		fakeWorker(ssc)
		if err := assertInvariants(set, spc); err != nil {
			return err
		}
		if obj, _, err := spc.setsIndexer.Get(set); err != nil {
			return err
		} else {
			set = obj.(*apps.StatefulSet)
		}

	}
	return assertInvariants(set, spc)
}

func scaleDownStatefulSetController(set *apps.StatefulSet, ssc *StatefulSetController, spc *fakeStatefulPodControl) error {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return err
	}
	ord := len(pods) - 1
	pod := getPodAtOrdinal(pods, ord)
	prev := *pod
	fakeResourceVersion(set)
	spc.setsIndexer.Add(set)
	ssc.enqueueStatefulSet(set)
	fakeWorker(ssc)
	pods, err = spc.addTerminatedPod(set, ord)
	pod = getPodAtOrdinal(pods, ord)
	ssc.updatePod(&prev, pod)
	fakeWorker(ssc)
	spc.DeleteStatefulPod(set, pod)
	ssc.deletePod(pod)
	fakeWorker(ssc)
	for set.Status.Replicas > *set.Spec.Replicas {
		pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
		ord := len(pods)
		pods, err = spc.addTerminatedPod(set, ord)
		pod = getPodAtOrdinal(pods, ord)
		ssc.updatePod(&prev, pod)
		fakeWorker(ssc)
		spc.DeleteStatefulPod(set, pod)
		ssc.deletePod(pod)
		fakeWorker(ssc)
		if obj, _, err := spc.setsIndexer.Get(set); err != nil {
			return err
		} else {
			set = obj.(*apps.StatefulSet)
		}
	}
	return assertInvariants(set, spc)
}
