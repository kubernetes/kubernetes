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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"errors"
	"fmt"
	"math/rand"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/securitycontext"
)

func testNewReplicaSetControllerFromClient(client clientset.Interface, stopCh chan struct{}, burstReplicas int) (*ReplicaSetController, informers.SharedInformerFactory) {
	informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	ret := NewReplicaSetController(
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Core().V1().Pods(),
		client,
		burstReplicas,
	)

	ret.podListerSynced = alwaysReady
	ret.rsListerSynced = alwaysReady

	return ret, informers
}

func skipListerFunc(verb string, url url.URL) bool {
	if verb != "GET" {
		return false
	}
	if strings.HasSuffix(url.Path, "/pods") || strings.Contains(url.Path, "/replicasets") {
		return true
	}
	return false
}

var alwaysReady = func() bool { return true }

func getKey(rs *extensions.ReplicaSet, t *testing.T) string {
	if key, err := controller.KeyFunc(rs); err != nil {
		t.Errorf("Unexpected error getting key for ReplicaSet %v: %v", rs.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicaSet(replicas int, selectorMap map[string]string) *extensions.ReplicaSet {
	rs := &extensions.ReplicaSet{
		TypeMeta: metav1.TypeMeta{APIVersion: legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: metav1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Selector: &metav1.LabelSelector{MatchLabels: selectorMap},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: v1.TerminationMessagePathDefault,
							ImagePullPolicy:        v1.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					RestartPolicy: v1.RestartPolicyAlways,
					DNSPolicy:     v1.DNSDefault,
					NodeSelector: map[string]string{
						"baz": "blah",
					},
				},
			},
		},
	}
	return rs
}

// create a pod with the given phase for the given rs (same selectors and namespace)
func newPod(name string, rs *extensions.ReplicaSet, status v1.PodPhase, lastTransitionTime *metav1.Time, properlyOwned bool) *v1.Pod {
	var conditions []v1.PodCondition
	if status == v1.PodRunning {
		condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
		if lastTransitionTime != nil {
			condition.LastTransitionTime = *lastTransitionTime
		}
		conditions = append(conditions, condition)
	}
	var controllerReference metav1.OwnerReference
	if properlyOwned {
		var trueVar = true
		controllerReference = metav1.OwnerReference{UID: rs.UID, APIVersion: "v1beta1", Kind: "ReplicaSet", Name: rs.Name, Controller: &trueVar}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       rs.Namespace,
			Labels:          rs.Spec.Selector.MatchLabels,
			OwnerReferences: []metav1.OwnerReference{controllerReference},
		},
		Status: v1.PodStatus{Phase: status, Conditions: conditions},
	}
}

// create count pods with the given phase for the given ReplicaSet (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status v1.PodPhase, labelMap map[string]string, rs *extensions.ReplicaSet, name string) *v1.PodList {
	pods := []v1.Pod{}
	var trueVar = true
	controllerReference := metav1.OwnerReference{UID: rs.UID, APIVersion: "v1beta1", Kind: "ReplicaSet", Name: rs.Name, Controller: &trueVar}
	for i := 0; i < count; i++ {
		pod := newPod(fmt.Sprintf("%s%d", name, i), rs, status, nil, false)
		pod.ObjectMeta.Labels = labelMap
		pod.OwnerReferences = []metav1.OwnerReference{controllerReference}
		if store != nil {
			store.Add(pod)
		}
		pods = append(pods, *pod)
	}
	return &v1.PodList{
		Items: pods,
	}
}

// processSync initiates a sync via processNextWorkItem() to test behavior that
// depends on both functions (such as re-queueing on sync error).
func processSync(rsc *ReplicaSetController, key string) error {
	// Save old syncHandler and replace with one that captures the error.
	oldSyncHandler := rsc.syncHandler
	defer func() {
		rsc.syncHandler = oldSyncHandler
	}()
	var syncErr error
	rsc.syncHandler = func(key string) error {
		syncErr = oldSyncHandler(key)
		return syncErr
	}
	rsc.queue.Add(key)
	rsc.processNextWorkItem()
	return syncErr
}

func validateSyncReplicaSet(t *testing.T, fakePodControl *controller.FakePodControl, expectedCreates, expectedDeletes, expectedPatches int) {
	if e, a := expectedCreates, len(fakePodControl.Templates); e != a {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", e, a)
	}
	if e, a := expectedDeletes, len(fakePodControl.DeletePodName); e != a {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", e, a)
	}
	if e, a := expectedPatches, len(fakePodControl.Patches); e != a {
		t.Errorf("Unexpected number of patches.  Expected %d, saw %d\n", e, a)
	}
}

func replicaSetResourceName() string {
	return "replicasets"
}

func TestSyncReplicaSetDoesNothing(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)

	// 2 running pods, a controller with 2 replicas, sync is a no-op
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)
	newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 2, v1.PodRunning, labelMap, rsSpec, "pod")

	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)
	manager.podControl = &fakePodControl

	received := make(chan string)
	manager.syncHandler = func(key string) error {
		received <- key
		return nil
	}

	// The DeletedFinalStateUnknown object should cause the ReplicaSet manager to insert
	// the controller matching the selectors of the deleted pod into the work queue.
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(1, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)
	pods := newPodList(nil, 1, v1.PodRunning, labelMap, rsSpec, "pod")
	manager.deletePod(cache.DeletedFinalStateUnknown{Key: "foo", Obj: &pods.Items[0]})

	go manager.worker()

	expected := getKey(rsSpec, t)
	select {
	case key := <-received:
		if key != expected {
			t.Errorf("Unexpected sync all for ReplicaSet %v, expected %v", key, expected)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Processing DeleteFinalStateUnknown took longer than expected")
	}
}

// Tell the rs to create 100 replicas, but simulate a limit (like a quota limit)
// of 10, and verify that the rs doesn't make 100 create calls per sync pass
func TestSyncReplicaSetCreateFailures(t *testing.T) {
	fakePodControl := controller.FakePodControl{}
	fakePodControl.CreateLimit = 10

	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(fakePodControl.CreateLimit*10, labelMap)
	client := fake.NewSimpleClientset(rs)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)

	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)

	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, fakePodControl.CreateLimit, 0, 0)
	expectedLimit := 0
	for pass := uint8(0); expectedLimit <= fakePodControl.CreateLimit; pass++ {
		expectedLimit += controller.SlowStartInitialBatchSize << pass
	}
	if fakePodControl.CreateCallCount > expectedLimit {
		t.Errorf("Unexpected number of create calls.  Expected <= %d, saw %d\n", fakePodControl.CreateLimit*2, fakePodControl.CreateCallCount)
	}
}

func TestSyncReplicaSetDormancy(t *testing.T) {
	// Setup a test server so we can lie about the current state of pods
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    200,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
		T:             t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)

	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)
	newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap, rsSpec, "pod")

	// Creates a replica and sets expectations
	rsSpec.Status.Replicas = 1
	rsSpec.Status.ReadyReplicas = 1
	rsSpec.Status.AvailableReplicas = 1
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0, 0)

	// Expectations prevents replicas but not an update on status
	rsSpec.Status.Replicas = 0
	rsSpec.Status.ReadyReplicas = 0
	rsSpec.Status.AvailableReplicas = 0
	fakePodControl.Clear()
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)

	// Get the key for the controller
	rsKey, err := controller.KeyFunc(rsSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rsSpec, err)
	}

	// Lowering expectations should lead to a sync that creates a replica, however the
	// fakePodControl error will prevent this, leaving expectations at 0, 0
	manager.expectations.CreationObserved(rsKey)
	rsSpec.Status.Replicas = 1
	rsSpec.Status.ReadyReplicas = 1
	rsSpec.Status.AvailableReplicas = 1
	fakePodControl.Clear()
	fakePodControl.Err = fmt.Errorf("Fake Error")

	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0, 0)

	// This replica should not need a Lowering of expectations, since the previous create failed
	fakePodControl.Clear()
	fakePodControl.Err = nil
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0, 0)

	// 2 PUT for the ReplicaSet status during dormancy window.
	// Note that the pod creates go through pod control so they're not recorded.
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestPodControllerLookup(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}}), stopCh, BurstReplicas)
	testCases := []struct {
		inRSs     []*extensions.ReplicaSet
		pod       *v1.Pod
		outRSName string
	}{
		// pods without labels don't match any ReplicaSets
		{
			inRSs: []*extensions.ReplicaSet{
				{ObjectMeta: metav1.ObjectMeta{Name: "basic"}}},
			pod:       &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: metav1.NamespaceAll}},
			outRSName: "",
		},
		// Matching labels, not namespace
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "foo"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo2", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "",
		},
		// Matching ns and labels returns the key to the ReplicaSet, not the ReplicaSet name
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo3", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "bar",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRSs {
			informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(r)
		}
		if rss := manager.getPodReplicaSets(c.pod); rss != nil {
			if len(rss) != 1 {
				t.Errorf("len(rss) = %v, want %v", len(rss), 1)
				continue
			}
			rs := rss[0]
			if c.outRSName != rs.Name {
				t.Errorf("Got replica set %+v expected %+v", rs.Name, c.outRSName)
			}
		} else if c.outRSName != "" {
			t.Errorf("Expected a replica set %v pod %v, found none", c.outRSName, c.pod.Name)
		}
	}
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := fake.NewSimpleClientset()
	client.PrependWatchReactor("replicasets", core.DefaultWatchReactor(fakeWatch, nil))
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	manager := NewReplicaSetController(
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Core().V1().Pods(),
		client,
		BurstReplicas,
	)
	informers.Start(stopCh)

	var testRSSpec extensions.ReplicaSet
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler. The handler validates the received controller
	// and closes the received channel to indicate that the test can finish.
	manager.syncHandler = func(key string) error {
		obj, exists, err := informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		rsSpec := *obj.(*extensions.ReplicaSet)
		if !apiequality.Semantic.DeepDerivative(rsSpec, testRSSpec) {
			t.Errorf("Expected %#v, but got %#v", testRSSpec, rsSpec)
		}
		close(received)
		return nil
	}
	// Start only the ReplicaSet watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	testRSSpec.Name = "foo"
	fakeWatch.Add(&testRSSpec)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("unexpected timeout from result channel")
	}
}

func TestWatchPods(t *testing.T) {
	client := fake.NewSimpleClientset()

	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)

	// Put one ReplicaSet into the shared informer
	labelMap := map[string]string{"foo": "bar"}
	testRSSpec := newReplicaSet(1, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(testRSSpec)

	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing ReplicaSet and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			t.Errorf("Error splitting key: %v", err)
		}
		rsSpec, err := manager.rsLister.ReplicaSets(namespace).Get(name)
		if err != nil {
			t.Errorf("Expected to find replica set under key %v: %v", key, err)
		}
		if !apiequality.Semantic.DeepDerivative(rsSpec, testRSSpec) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testRSSpec, rsSpec)
		}
		close(received)
		return nil
	}

	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right ReplicaSet.
	go informers.Core().V1().Pods().Informer().Run(stopCh)
	go manager.Run(1, stopCh)

	pods := newPodList(nil, 1, v1.PodRunning, labelMap, testRSSpec, "pod")
	testPod := pods.Items[0]
	testPod.Status.Phase = v1.PodFailed
	fakeWatch.Add(&testPod)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("unexpected timeout from result channel")
	}
}

func TestUpdatePods(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(fake.NewSimpleClientset(), stopCh, BurstReplicas)

	received := make(chan string)

	manager.syncHandler = func(key string) error {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			t.Errorf("Error splitting key: %v", err)
		}
		rsSpec, err := manager.rsLister.ReplicaSets(namespace).Get(name)
		if err != nil {
			t.Errorf("Expected to find replica set under key %v: %v", key, err)
		}
		received <- rsSpec.Name
		return nil
	}

	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	// Put 2 ReplicaSets and one pod into the informers
	labelMap1 := map[string]string{"foo": "bar"}
	testRSSpec1 := newReplicaSet(1, labelMap1)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(testRSSpec1)
	testRSSpec2 := *testRSSpec1
	labelMap2 := map[string]string{"bar": "foo"}
	testRSSpec2.Spec.Selector = &metav1.LabelSelector{MatchLabels: labelMap2}
	testRSSpec2.Name = "barfoo"
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(&testRSSpec2)

	isController := true
	controllerRef1 := metav1.OwnerReference{UID: testRSSpec1.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: testRSSpec1.Name, Controller: &isController}
	controllerRef2 := metav1.OwnerReference{UID: testRSSpec2.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: testRSSpec2.Name, Controller: &isController}

	// case 1: Pod with a ControllerRef
	pod1 := newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap1, testRSSpec1, "pod").Items[0]
	pod1.OwnerReferences = []metav1.OwnerReference{controllerRef1}
	pod1.ResourceVersion = "1"
	pod2 := pod1
	pod2.Labels = labelMap2
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected := sets.NewString(testRSSpec1.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets")
		}
	}

	// case 2: Remove ControllerRef (orphan). Expect to sync label-matching RS.
	pod1 = newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap1, testRSSpec1, "pod").Items[0]
	pod1.ResourceVersion = "1"
	pod1.Labels = labelMap2
	pod1.OwnerReferences = []metav1.OwnerReference{controllerRef2}
	pod2 = pod1
	pod2.OwnerReferences = nil
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected = sets.NewString(testRSSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets")
		}
	}

	// case 2: Remove ControllerRef (orphan). Expect to sync both former owner and
	// any label-matching RS.
	pod1 = newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap1, testRSSpec1, "pod").Items[0]
	pod1.ResourceVersion = "1"
	pod1.Labels = labelMap2
	pod1.OwnerReferences = []metav1.OwnerReference{controllerRef1}
	pod2 = pod1
	pod2.OwnerReferences = nil
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected = sets.NewString(testRSSpec1.Name, testRSSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets")
		}
	}

	// case 4: Keep ControllerRef, change labels. Expect to sync owning RS.
	pod1 = newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap1, testRSSpec1, "pod").Items[0]
	pod1.ResourceVersion = "1"
	pod1.Labels = labelMap1
	pod1.OwnerReferences = []metav1.OwnerReference{controllerRef2}
	pod2 = pod1
	pod2.Labels = labelMap2
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected = sets.NewString(testRSSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets")
		}
	}
}

func TestControllerUpdateRequeue(t *testing.T) {
	// This server should force a requeue of the controller because it fails to update status.Replicas.
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(1, labelMap)
	client := fake.NewSimpleClientset(rs)
	client.PrependReactor("update", "replicasets",
		func(action core.Action) (bool, runtime.Object, error) {
			if action.GetSubresource() != "status" {
				return false, nil, nil
			}
			return true, nil, errors.New("failed to update status")
		})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas)

	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2}
	newPodList(informers.Core().V1().Pods().Informer().GetIndexer(), 1, v1.PodRunning, labelMap, rs, "pod")

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// Enqueue once. Then process it. Disable rate-limiting for this.
	manager.queue = workqueue.NewRateLimitingQueue(workqueue.NewMaxOfRateLimiter())
	manager.enqueueReplicaSet(rs)
	manager.processNextWorkItem()
	// It should have been requeued.
	if got, want := manager.queue.Len(), 1; got != want {
		t.Errorf("queue.Len() = %v, want %v", got, want)
	}
}

func TestControllerUpdateStatusWithFailure(t *testing.T) {
	rs := newReplicaSet(1, map[string]string{"foo": "bar"})
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "replicasets", func(action core.Action) (bool, runtime.Object, error) { return true, rs, nil })
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, &extensions.ReplicaSet{}, fmt.Errorf("Fake error")
	})
	fakeRSClient := fakeClient.Extensions().ReplicaSets("default")
	numReplicas := int32(10)
	newStatus := extensions.ReplicaSetStatus{Replicas: numReplicas}
	updateReplicaSetStatus(fakeRSClient, rs, newStatus)
	updates, gets := 0, 0
	for _, a := range fakeClient.Actions() {
		if a.GetResource().Resource != "replicasets" {
			t.Errorf("Unexpected action %+v", a)
			continue
		}

		switch action := a.(type) {
		case core.GetAction:
			gets++
			// Make sure the get is for the right ReplicaSet even though the update failed.
			if action.GetName() != rs.Name {
				t.Errorf("Expected get for ReplicaSet %v, got %+v instead", rs.Name, action.GetName())
			}
		case core.UpdateAction:
			updates++
			// Confirm that the update has the right status.Replicas even though the Get
			// returned a ReplicaSet with replicas=1.
			if c, ok := action.GetObject().(*extensions.ReplicaSet); !ok {
				t.Errorf("Expected a ReplicaSet as the argument to update, got %T", c)
			} else if c.Status.Replicas != numReplicas {
				t.Errorf("Expected update for ReplicaSet to contain replicas %v, got %v instead",
					numReplicas, c.Status.Replicas)
			}
		default:
			t.Errorf("Unexpected action %+v", a)
			break
		}
	}
	if gets != 1 || updates != 2 {
		t.Errorf("Expected 1 get and 2 updates, got %d gets %d updates", gets, updates)
	}
}

// TODO: This test is too hairy for a unittest. It should be moved to an E2E suite.
func doTestControllerBurstReplicas(t *testing.T, burstReplicas, numReplicas int) {
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(numReplicas, labelMap)
	client := fake.NewSimpleClientset(rsSpec)
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, burstReplicas)
	manager.podControl = &fakePodControl

	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)

	expectedPods := int32(0)
	pods := newPodList(nil, numReplicas, v1.PodPending, labelMap, rsSpec, "pod")

	rsKey, err := controller.KeyFunc(rsSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rsSpec, err)
	}

	// Size up the controller, then size it down, and confirm the expected create/delete pattern
	for _, replicas := range []int32{int32(numReplicas), 0} {

		*(rsSpec.Spec.Replicas) = replicas
		informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)

		for i := 0; i < numReplicas; i += burstReplicas {
			manager.syncReplicaSet(getKey(rsSpec, t))

			// The store accrues active pods. It's also used by the ReplicaSet to determine how many
			// replicas to create.
			activePods := int32(len(informers.Core().V1().Pods().Informer().GetIndexer().List()))
			if replicas != 0 {
				// This is the number of pods currently "in flight". They were created by the
				// ReplicaSet controller above, which then puts the ReplicaSet to sleep till
				// all of them have been observed.
				expectedPods = replicas - activePods
				if expectedPods > int32(burstReplicas) {
					expectedPods = int32(burstReplicas)
				}
				// This validates the ReplicaSet manager sync actually created pods
				validateSyncReplicaSet(t, &fakePodControl, int(expectedPods), 0, 0)

				// This simulates the watch events for all but 1 of the expected pods.
				// None of these should wake the controller because it has expectations==BurstReplicas.
				for i := int32(0); i < expectedPods-1; i++ {
					informers.Core().V1().Pods().Informer().GetIndexer().Add(&pods.Items[i])
					manager.addPod(&pods.Items[i])
				}

				podExp, exists, err := manager.expectations.GetExpectations(rsKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for rs.")
				}
				if add, _ := podExp.GetExpectations(); add != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			} else {
				expectedPods = (replicas - activePods) * -1
				if expectedPods > int32(burstReplicas) {
					expectedPods = int32(burstReplicas)
				}
				validateSyncReplicaSet(t, &fakePodControl, 0, int(expectedPods), 0)

				// To accurately simulate a watch we must delete the exact pods
				// the rs is waiting for.
				expectedDels := manager.expectations.GetUIDs(getKey(rsSpec, t))
				podsToDelete := []*v1.Pod{}
				isController := true
				for _, key := range expectedDels.List() {
					nsName := strings.Split(key, "/")
					podsToDelete = append(podsToDelete, &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:      nsName[1],
							Namespace: nsName[0],
							Labels:    rsSpec.Spec.Selector.MatchLabels,
							OwnerReferences: []metav1.OwnerReference{
								{UID: rsSpec.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: rsSpec.Name, Controller: &isController},
							},
						},
					})
				}
				// Don't delete all pods because we confirm that the last pod
				// has exactly one expectation at the end, to verify that we
				// don't double delete.
				for i := range podsToDelete[1:] {
					informers.Core().V1().Pods().Informer().GetIndexer().Delete(podsToDelete[i])
					manager.deletePod(podsToDelete[i])
				}
				podExp, exists, err := manager.expectations.GetExpectations(rsKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for ReplicaSet.")
				}
				if _, del := podExp.GetExpectations(); del != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			}

			// Check that the ReplicaSet didn't take any action for all the above pods
			fakePodControl.Clear()
			manager.syncReplicaSet(getKey(rsSpec, t))
			validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)

			// Create/Delete the last pod
			// The last add pod will decrease the expectation of the ReplicaSet to 0,
			// which will cause it to create/delete the remaining replicas up to burstReplicas.
			if replicas != 0 {
				informers.Core().V1().Pods().Informer().GetIndexer().Add(&pods.Items[expectedPods-1])
				manager.addPod(&pods.Items[expectedPods-1])
			} else {
				expectedDel := manager.expectations.GetUIDs(getKey(rsSpec, t))
				if expectedDel.Len() != 1 {
					t.Fatalf("Waiting on unexpected number of deletes.")
				}
				nsName := strings.Split(expectedDel.List()[0], "/")
				isController := true
				lastPod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      nsName[1],
						Namespace: nsName[0],
						Labels:    rsSpec.Spec.Selector.MatchLabels,
						OwnerReferences: []metav1.OwnerReference{
							{UID: rsSpec.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: rsSpec.Name, Controller: &isController},
						},
					},
				}
				informers.Core().V1().Pods().Informer().GetIndexer().Delete(lastPod)
				manager.deletePod(lastPod)
			}
			pods.Items = pods.Items[expectedPods:]
		}

		// Confirm that we've created the right number of replicas
		activePods := int32(len(informers.Core().V1().Pods().Informer().GetIndexer().List()))
		if activePods != *(rsSpec.Spec.Replicas) {
			t.Fatalf("Unexpected number of active pods, expected %d, got %d", *(rsSpec.Spec.Replicas), activePods)
		}
		// Replenish the pod list, since we cut it down sizing up
		pods = newPodList(nil, int(replicas), v1.PodRunning, labelMap, rsSpec, "pod")
	}
}

func TestControllerBurstReplicas(t *testing.T) {
	doTestControllerBurstReplicas(t, 5, 30)
	doTestControllerBurstReplicas(t, 5, 12)
	doTestControllerBurstReplicas(t, 3, 2)
}

type FakeRSExpectations struct {
	*controller.ControllerExpectations
	satisfied    bool
	expSatisfied func()
}

func (fe FakeRSExpectations) SatisfiedExpectations(controllerKey string) bool {
	fe.expSatisfied()
	return fe.satisfied
}

// TestRSSyncExpectations tests that a pod cannot sneak in between counting active pods
// and checking expectations.
func TestRSSyncExpectations(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, 2)
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rsSpec)
	pods := newPodList(nil, 2, v1.PodPending, labelMap, rsSpec, "pod")
	informers.Core().V1().Pods().Informer().GetIndexer().Add(&pods.Items[0])
	postExpectationsPod := pods.Items[1]

	manager.expectations = controller.NewUIDTrackingControllerExpectations(FakeRSExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the
			// ReplicaSet will create a new replica because it doesn't see
			// this pod, but has fulfilled its expectations.
			informers.Core().V1().Pods().Informer().GetIndexer().Add(&postExpectationsPod)
		},
	})
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

func TestDeleteControllerAndExpectations(t *testing.T) {
	rs := newReplicaSet(1, map[string]string{"foo": "bar"})
	client := fake.NewSimpleClientset(rs)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, 10)

	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// This should set expectations for the ReplicaSet
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 1, 0, 0)
	fakePodControl.Clear()

	// Get the ReplicaSet key
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rs, err)
	}

	// This is to simulate a concurrent addPod, that has a handle on the expectations
	// as the controller deletes it.
	podExp, exists, err := manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil {
		t.Errorf("No expectations found for ReplicaSet")
	}
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Delete(rs)
	manager.syncReplicaSet(getKey(rs, t))

	if _, exists, err = manager.expectations.GetExpectations(rsKey); exists {
		t.Errorf("Found expectaions, expected none since the ReplicaSet has been deleted.")
	}

	// This should have no effect, since we've deleted the ReplicaSet.
	podExp.Add(-1, 0)
	informers.Core().V1().Pods().Informer().GetIndexer().Replace(make([]interface{}, 0), "0")
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

// shuffle returns a new shuffled list of container controllers.
func shuffle(controllers []*extensions.ReplicaSet) []*extensions.ReplicaSet {
	numControllers := len(controllers)
	randIndexes := rand.Perm(numControllers)
	shuffled := make([]*extensions.ReplicaSet, numControllers)
	for i := 0; i < numControllers; i++ {
		shuffled[i] = controllers[randIndexes[i]]
	}
	return shuffled
}

func TestOverlappingRSs(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	labelMap := map[string]string{"foo": "bar"}

	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(client, stopCh, 10)

	// Create 10 ReplicaSets, shuffled them randomly and insert them into the
	// ReplicaSet controller's store.
	// All use the same CreationTimestamp since ControllerRef should be able
	// to handle that.
	timestamp := metav1.Date(2014, time.December, 0, 0, 0, 0, 0, time.Local)
	var controllers []*extensions.ReplicaSet
	for j := 1; j < 10; j++ {
		rsSpec := newReplicaSet(1, labelMap)
		rsSpec.CreationTimestamp = timestamp
		rsSpec.Name = fmt.Sprintf("rs%d", j)
		controllers = append(controllers, rsSpec)
	}
	shuffledControllers := shuffle(controllers)
	for j := range shuffledControllers {
		informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(shuffledControllers[j])
	}
	// Add a pod with a ControllerRef and make sure only the corresponding
	// ReplicaSet is synced. Pick a RS in the middle since the old code used to
	// sort by name if all timestamps were equal.
	rs := controllers[3]
	pods := newPodList(nil, 1, v1.PodPending, labelMap, rs, "pod")
	pod := &pods.Items[0]
	isController := true
	pod.OwnerReferences = []metav1.OwnerReference{
		{UID: rs.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: rs.Name, Controller: &isController},
	}
	rsKey := getKey(rs, t)

	manager.addPod(pod)
	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
}

func TestDeletionTimestamp(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	labelMap := map[string]string{"foo": "bar"}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, informers := testNewReplicaSetControllerFromClient(c, stopCh, 10)

	rs := newReplicaSet(1, labelMap)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rs, err)
	}
	pod := newPodList(nil, 1, v1.PodPending, labelMap, rs, "pod").Items[0]
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	pod.ResourceVersion = "1"
	manager.expectations.ExpectDeletions(rsKey, []string{controller.PodKey(&pod)})

	// A pod added with a deletion timestamp should decrement deletions, not creations.
	manager.addPod(&pod)

	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
	manager.queue.Done(rsKey)

	podExp, exists, err := manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil || !podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// An update from no deletion timestamp to having one should be treated
	// as a deletion.
	oldPod := newPodList(nil, 1, v1.PodPending, labelMap, rs, "pod").Items[0]
	oldPod.ResourceVersion = "2"
	manager.expectations.ExpectDeletions(rsKey, []string{controller.PodKey(&pod)})
	manager.updatePod(&oldPod, &pod)

	queueRS, _ = manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
	manager.queue.Done(rsKey)

	podExp, exists, err = manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil || !podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// An update to the pod (including an update to the deletion timestamp)
	// should not be counted as a second delete.
	isController := true
	secondPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: pod.Namespace,
			Name:      "secondPod",
			Labels:    pod.Labels,
			OwnerReferences: []metav1.OwnerReference{
				{UID: rs.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: rs.Name, Controller: &isController},
			},
		},
	}
	manager.expectations.ExpectDeletions(rsKey, []string{controller.PodKey(secondPod)})
	oldPod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	oldPod.ResourceVersion = "2"
	manager.updatePod(&oldPod, &pod)

	podExp, exists, err = manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil || podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// A pod with a non-nil deletion timestamp should also be ignored by the
	// delete handler, because it's already been counted in the update.
	manager.deletePod(&pod)
	podExp, exists, err = manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil || podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// Deleting the second pod should clear expectations.
	manager.deletePod(secondPod)

	queueRS, _ = manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
	manager.queue.Done(rsKey)

	podExp, exists, err = manager.expectations.GetExpectations(rsKey)
	if !exists || err != nil || !podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}
}

// setupManagerWithGCEnabled creates a RS manager with a fakePodControl
func setupManagerWithGCEnabled(stopCh chan struct{}, objs ...runtime.Object) (manager *ReplicaSetController, fakePodControl *controller.FakePodControl, informers informers.SharedInformerFactory) {
	c := fakeclientset.NewSimpleClientset(objs...)
	fakePodControl = &controller.FakePodControl{}
	manager, informers = testNewReplicaSetControllerFromClient(c, stopCh, BurstReplicas)

	manager.podControl = fakePodControl
	return manager, fakePodControl, informers
}

func TestDoNotPatchPodWithOtherControlRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl, informers := setupManagerWithGCEnabled(stopCh, rs)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	var trueVar = true
	otherControllerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "v1beta1", Kind: "ReplicaSet", Name: "AnotherRS", Controller: &trueVar}
	// add to podLister a matching Pod controlled by another controller. Expect no patch.
	pod := newPod("pod", rs, v1.PodRunning, nil, true)
	pod.OwnerReferences = []metav1.OwnerReference{otherControllerReference}
	informers.Core().V1().Pods().Informer().GetIndexer().Add(pod)
	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	// because the matching pod already has a controller, so 2 pods should be created.
	validateSyncReplicaSet(t, fakePodControl, 2, 0, 0)
}

func TestPatchPodFails(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl, informers := setupManagerWithGCEnabled(stopCh, rs)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	// add to podLister two matching pods. Expect two patches to take control
	// them.
	informers.Core().V1().Pods().Informer().GetIndexer().Add(newPod("pod1", rs, v1.PodRunning, nil, false))
	informers.Core().V1().Pods().Informer().GetIndexer().Add(newPod("pod2", rs, v1.PodRunning, nil, false))
	// let both patches fail. The rs controller will assume it fails to take
	// control of the pods and requeue to try again.
	fakePodControl.Err = fmt.Errorf("Fake Error")
	rsKey := getKey(rs, t)
	err := processSync(manager, rsKey)
	if err == nil || !strings.Contains(err.Error(), "Fake Error") {
		t.Errorf("expected Fake Error, got %+v", err)
	}
	// 2 patches to take control of pod1 and pod2 (both fail).
	validateSyncReplicaSet(t, fakePodControl, 0, 0, 2)
	// RS should requeue itself.
	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
}

// RS controller shouldn't adopt or create more pods if the rc is about to be
// deleted.
func TestDoNotAdoptOrCreateIfBeingDeleted(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	now := metav1.Now()
	rs.DeletionTimestamp = &now
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl, informers := setupManagerWithGCEnabled(stopCh, rs)
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(rs)
	pod1 := newPod("pod1", rs, v1.PodRunning, nil, false)
	informers.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	// no patch, no create
	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	validateSyncReplicaSet(t, fakePodControl, 0, 0, 0)
}

func TestDoNotAdoptOrCreateIfBeingDeletedRace(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	// Bare client says it IS deleted.
	rs := newReplicaSet(2, labelMap)
	now := metav1.Now()
	rs.DeletionTimestamp = &now
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl, informers := setupManagerWithGCEnabled(stopCh, rs)
	// Lister (cache) says it's NOT deleted.
	rs2 := *rs
	rs2.DeletionTimestamp = nil
	informers.Extensions().V1beta1().ReplicaSets().Informer().GetIndexer().Add(&rs2)

	// Recheck occurs if a matching orphan is present.
	pod1 := newPod("pod1", rs, v1.PodRunning, nil, false)
	informers.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	// sync should abort.
	err := manager.syncReplicaSet(getKey(rs, t))
	if err == nil {
		t.Error("syncReplicaSet() err = nil, expected non-nil")
	}
	// no patch, no create.
	validateSyncReplicaSet(t, fakePodControl, 0, 0, 0)
}

var (
	imagePullBackOff extensions.ReplicaSetConditionType = "ImagePullBackOff"

	condImagePullBackOff = func() extensions.ReplicaSetCondition {
		return extensions.ReplicaSetCondition{
			Type:   imagePullBackOff,
			Status: v1.ConditionTrue,
			Reason: "NonExistentImage",
		}
	}

	condReplicaFailure = func() extensions.ReplicaSetCondition {
		return extensions.ReplicaSetCondition{
			Type:   extensions.ReplicaSetReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "OtherFailure",
		}
	}

	condReplicaFailure2 = func() extensions.ReplicaSetCondition {
		return extensions.ReplicaSetCondition{
			Type:   extensions.ReplicaSetReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "AnotherFailure",
		}
	}

	status = func() *extensions.ReplicaSetStatus {
		return &extensions.ReplicaSetStatus{
			Conditions: []extensions.ReplicaSetCondition{condReplicaFailure()},
		}
	}
)

func TestGetCondition(t *testing.T) {
	exampleStatus := status()

	tests := []struct {
		name string

		status     extensions.ReplicaSetStatus
		condType   extensions.ReplicaSetConditionType
		condStatus v1.ConditionStatus
		condReason string

		expected bool
	}{
		{
			name: "condition exists",

			status:   *exampleStatus,
			condType: extensions.ReplicaSetReplicaFailure,

			expected: true,
		},
		{
			name: "condition does not exist",

			status:   *exampleStatus,
			condType: imagePullBackOff,

			expected: false,
		},
	}

	for _, test := range tests {
		cond := GetCondition(test.status, test.condType)
		exists := cond != nil
		if exists != test.expected {
			t.Errorf("%s: expected condition to exist: %t, got: %t", test.name, test.expected, exists)
		}
	}
}

func TestSetCondition(t *testing.T) {
	tests := []struct {
		name string

		status *extensions.ReplicaSetStatus
		cond   extensions.ReplicaSetCondition

		expectedStatus *extensions.ReplicaSetStatus
	}{
		{
			name: "set for the first time",

			status: &extensions.ReplicaSetStatus{},
			cond:   condReplicaFailure(),

			expectedStatus: &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condReplicaFailure()}},
		},
		{
			name: "simple set",

			status: &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condImagePullBackOff()}},
			cond:   condReplicaFailure(),

			expectedStatus: &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condImagePullBackOff(), condReplicaFailure()}},
		},
		{
			name: "overwrite",

			status: &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condReplicaFailure()}},
			cond:   condReplicaFailure2(),

			expectedStatus: &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condReplicaFailure2()}},
		},
	}

	for _, test := range tests {
		SetCondition(test.status, test.cond)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}

func TestRemoveCondition(t *testing.T) {
	tests := []struct {
		name string

		status   *extensions.ReplicaSetStatus
		condType extensions.ReplicaSetConditionType

		expectedStatus *extensions.ReplicaSetStatus
	}{
		{
			name: "remove from empty status",

			status:   &extensions.ReplicaSetStatus{},
			condType: extensions.ReplicaSetReplicaFailure,

			expectedStatus: &extensions.ReplicaSetStatus{},
		},
		{
			name: "simple remove",

			status:   &extensions.ReplicaSetStatus{Conditions: []extensions.ReplicaSetCondition{condReplicaFailure()}},
			condType: extensions.ReplicaSetReplicaFailure,

			expectedStatus: &extensions.ReplicaSetStatus{},
		},
		{
			name: "doesn't remove anything",

			status:   status(),
			condType: imagePullBackOff,

			expectedStatus: status(),
		},
	}

	for _, test := range tests {
		RemoveCondition(test.status, test.condType)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}
