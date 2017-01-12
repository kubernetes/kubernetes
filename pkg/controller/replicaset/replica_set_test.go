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
	"fmt"
	"math/rand"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	fakeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/securitycontext"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/util/uuid"
)

func testNewReplicaSetControllerFromClient(client clientset.Interface, stopCh chan struct{}, burstReplicas int, lookupCacheSize int) *ReplicaSetController {
	informers := informers.NewSharedInformerFactory(client, nil, controller.NoResyncPeriodFunc())
	ret := NewReplicaSetController(informers.ReplicaSets(), informers.Pods(), client, burstReplicas, lookupCacheSize, false)
	ret.podLister = &cache.StoreToPodLister{Indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})}
	ret.rsLister = &cache.StoreToReplicaSetLister{Indexer: cache.NewIndexer(cache.DeletionHandlingMetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})}
	informers.Start(stopCh)
	return ret
}

func filterInformerActions(actions []core.Action) []core.Action {
	ret := []core.Action{}
	for _, action := range actions {
		if len(action.GetNamespace()) == 0 &&
			(action.Matches("list", "pods") ||
				action.Matches("list", "replicasets") ||
				action.Matches("watch", "pods") ||
				action.Matches("watch", "replicasets")) {
			continue
		}
		ret = append(ret, action)
	}

	return ret
}

func skipListerFunc(verb string, url url.URL) bool {
	if verb != "GET" {
		return false
	}
	if strings.HasSuffix(url.Path, "/pods") || strings.HasSuffix(url.Path, "/replicasets") {
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
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: v1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       v1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Selector: &metav1.LabelSelector{MatchLabels: selectorMap},
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
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
func newPod(name string, rs *extensions.ReplicaSet, status v1.PodPhase, lastTransitionTime *metav1.Time) *v1.Pod {
	var conditions []v1.PodCondition
	if status == v1.PodRunning {
		condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
		if lastTransitionTime != nil {
			condition.LastTransitionTime = *lastTransitionTime
		}
		conditions = append(conditions, condition)
	}
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: rs.Namespace,
			Labels:    rs.Spec.Selector.MatchLabels,
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
		pod := newPod(fmt.Sprintf("%s%d", name, i), rs, status, nil)
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

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func TestSyncReplicaSetDoesNothing(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// 2 running pods, a controller with 2 replicas, sync is a no-op
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsLister.Indexer.Add(rsSpec)
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, labelMap, rsSpec, "pod")

	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

func TestSyncReplicaSetDeletes(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	// 2 running pods and a controller with 1 replica, one pod delete expected
	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(1, labelMap)
	manager.rsLister.Indexer.Add(rsSpec)
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, labelMap, rsSpec, "pod")

	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 1, 0)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
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
	manager.rsLister.Indexer.Add(rsSpec)
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

func TestSyncReplicaSetCreates(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// A controller with 2 replicas and no pods in the store, 2 creates expected
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	manager.rsLister.Indexer.Add(rs)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rs, t))
	validateSyncReplicaSet(t, &fakePodControl, 2, 0, 0)
}

func TestStatusUpdatesWithoutReplicasChange(t *testing.T) {
	// Setup a fake server to listen for requests, and run the ReplicaSet controller in steady state
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    200,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Steady state for the ReplicaSet, no Status.Replicas updates expected
	activePods := 5
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(activePods, labelMap)
	manager.rsLister.Indexer.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: int32(activePods), ReadyReplicas: int32(activePods), AvailableReplicas: int32(activePods)}
	newPodList(manager.podLister.Indexer, activePods, v1.PodRunning, labelMap, rs, "pod")

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicaSet(getKey(rs, t))

	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
	if fakeHandler.RequestReceived != nil {
		t.Errorf("Unexpected update when pods and ReplicaSets are in a steady state")
	}

	// This response body is just so we don't err out decoding the http response, all
	// we care about is the request body sent below.
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.SetResponseBody(response)

	rs.Generation = rs.Generation + 1
	manager.syncReplicaSet(getKey(rs, t))

	rs.Status.ObservedGeneration = rs.Generation
	updatedRc := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &updatedRc)
}

func TestControllerUpdateReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    200,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Insufficient number of pods in the system, and Status.Replicas is wrong;
	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	labelMap := map[string]string{"foo": "bar"}
	extraLabelMap := map[string]string{"foo": "bar", "extraKey": "extraValue"}
	rs := newReplicaSet(5, labelMap)
	rs.Spec.Template.Labels = extraLabelMap
	manager.rsLister.Indexer.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, FullyLabeledReplicas: 6, ReadyReplicas: 2, AvailableReplicas: 2, ObservedGeneration: 0}
	rs.Generation = 1
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, labelMap, rs, "pod")
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, extraLabelMap, rs, "podWithExtraLabel")

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.SetResponseBody(response)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicaSet(getKey(rs, t))

	// 1. Status.Replicas should go up from 2->4 even though we created 5-4=1 pod.
	// 2. Status.FullyLabeledReplicas should equal to the number of pods that
	// has the extra labels, i.e., 2.
	// 3. Every update to the status should include the Generation of the spec.
	rs.Status = extensions.ReplicaSetStatus{Replicas: 4, FullyLabeledReplicas: 2, ReadyReplicas: 4, AvailableReplicas: 4, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &decRc)
	validateSyncReplicaSet(t, &fakePodControl, 1, 0, 0)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsLister.Indexer.Add(rsSpec)
	newPodList(manager.podLister.Indexer, 1, v1.PodRunning, labelMap, rsSpec, "pod")

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
	manager := testNewReplicaSetControllerFromClient(clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}}), stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	testCases := []struct {
		inRSs     []*extensions.ReplicaSet
		pod       *v1.Pod
		outRSName string
	}{
		// pods without labels don't match any ReplicaSets
		{
			inRSs: []*extensions.ReplicaSet{
				{ObjectMeta: v1.ObjectMeta{Name: "basic"}}},
			pod:       &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo1", Namespace: v1.NamespaceAll}},
			outRSName: "",
		},
		// Matching labels, not namespace
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: v1.ObjectMeta{Name: "foo"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo2", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "",
		},
		// Matching ns and labels returns the key to the ReplicaSet, not the ReplicaSet name
		{
			inRSs: []*extensions.ReplicaSet{
				{
					ObjectMeta: v1.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: extensions.ReplicaSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo3", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRSName: "bar",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRSs {
			manager.rsLister.Indexer.Add(r)
		}
		if rs := manager.getPodReplicaSet(c.pod); rs != nil {
			if c.outRSName != rs.Name {
				t.Errorf("Got replica set %+v expected %+v", rs.Name, c.outRSName)
			}
		} else if c.outRSName != "" {
			t.Errorf("Expected a replica set %v pod %v, found none", c.outRSName, c.pod.Name)
		}
	}
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*fake.Clientset
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &fake.Clientset{}
	client.AddWatchReactor("replicasets", core.DefaultWatchReactor(fakeWatch, nil))
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers := informers.NewSharedInformerFactory(client, nil, controller.NoResyncPeriodFunc())
	manager := NewReplicaSetController(informers.ReplicaSets(), informers.Pods(), client, BurstReplicas, 0, false)
	informers.Start(stopCh)
	manager.podListerSynced = alwaysReady

	var testRSSpec extensions.ReplicaSet
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler. The handler validates the received controller
	// and closes the received channel to indicate that the test can finish.
	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rsLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		rsSpec := *obj.(*extensions.ReplicaSet)
		if !api.Semantic.DeepDerivative(rsSpec, testRSSpec) {
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
	fakeWatch := watch.NewFake()
	client := &fake.Clientset{}
	client.AddWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Put one ReplicaSet and one pod into the controller's stores
	labelMap := map[string]string{"foo": "bar"}
	testRSSpec := newReplicaSet(1, labelMap)
	manager.rsLister.Indexer.Add(testRSSpec)
	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing ReplicaSet and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rsLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		rsSpec := obj.(*extensions.ReplicaSet)
		if !api.Semantic.DeepDerivative(rsSpec, testRSSpec) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testRSSpec, rsSpec)
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right ReplicaSet.
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

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
	manager := testNewReplicaSetControllerFromClient(fake.NewSimpleClientset(), stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	received := make(chan string)

	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rsLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find replica set under key %v", key)
		}
		received <- obj.(*extensions.ReplicaSet).Name
		return nil
	}

	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	// Put 2 ReplicaSets and one pod into the controller's stores
	labelMap1 := map[string]string{"foo": "bar"}
	testRSSpec1 := newReplicaSet(1, labelMap1)
	manager.rsLister.Indexer.Add(testRSSpec1)
	testRSSpec2 := *testRSSpec1
	labelMap2 := map[string]string{"bar": "foo"}
	testRSSpec2.Spec.Selector = &metav1.LabelSelector{MatchLabels: labelMap2}
	testRSSpec2.Name = "barfoo"
	manager.rsLister.Indexer.Add(&testRSSpec2)

	// case 1: We put in the podLister a pod with labels matching testRSSpec1,
	// then update its labels to match testRSSpec2.  We expect to receive a sync
	// request for both replica sets.
	pod1 := newPodList(manager.podLister.Indexer, 1, v1.PodRunning, labelMap1, testRSSpec1, "pod").Items[0]
	pod1.ResourceVersion = "1"
	pod2 := pod1
	pod2.Labels = labelMap2
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected := sets.NewString(testRSSpec1.Name, testRSSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets within 100ms each")
		}
	}

	// case 2: pod1 in the podLister has labels matching testRSSpec1. We update
	// its labels to match no replica set. We expect to receive a sync request
	// for testRSSpec1.
	pod2.Labels = make(map[string]string)
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected = sets.NewString(testRSSpec1.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for replica sets within 100ms each")
		}
	}
}

func TestControllerUpdateRequeue(t *testing.T) {
	// This server should force a requeue of the controller because it fails to update status.Replicas.
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    500,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	stopCh := make(chan struct{})
	defer close(stopCh)
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(1, labelMap)
	manager.rsLister.Indexer.Add(rs)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2}
	newPodList(manager.podLister.Indexer, 1, v1.PodRunning, labelMap, rs, "pod")

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// an error from the sync function will be requeued, check to make sure we returned an error
	err := manager.syncReplicaSet(getKey(rs, t))
	if err == nil {
		t.Errorf("missing error for requeue")
	}
	// 1 Update and 1 GET, both of which fail
	fakeHandler.ValidateRequestCount(t, 2)
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
	updateReplicaSetStatus(fakeRSClient, *rs, newStatus)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, burstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(numReplicas, labelMap)
	manager.rsLister.Indexer.Add(rsSpec)

	expectedPods := int32(0)
	pods := newPodList(nil, numReplicas, v1.PodPending, labelMap, rsSpec, "pod")

	rsKey, err := controller.KeyFunc(rsSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rsSpec, err)
	}

	// Size up the controller, then size it down, and confirm the expected create/delete pattern
	for _, replicas := range []int32{int32(numReplicas), 0} {

		*(rsSpec.Spec.Replicas) = replicas
		manager.rsLister.Indexer.Add(rsSpec)

		for i := 0; i < numReplicas; i += burstReplicas {
			manager.syncReplicaSet(getKey(rsSpec, t))

			// The store accrues active pods. It's also used by the ReplicaSet to determine how many
			// replicas to create.
			activePods := int32(len(manager.podLister.Indexer.List()))
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
					manager.podLister.Indexer.Add(&pods.Items[i])
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
				for _, key := range expectedDels.List() {
					nsName := strings.Split(key, "/")
					podsToDelete = append(podsToDelete, &v1.Pod{
						ObjectMeta: v1.ObjectMeta{
							Name:      nsName[1],
							Namespace: nsName[0],
							Labels:    rsSpec.Spec.Selector.MatchLabels,
						},
					})
				}
				// Don't delete all pods because we confirm that the last pod
				// has exactly one expectation at the end, to verify that we
				// don't double delete.
				for i := range podsToDelete[1:] {
					manager.podLister.Indexer.Delete(podsToDelete[i])
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
				manager.podLister.Indexer.Add(&pods.Items[expectedPods-1])
				manager.addPod(&pods.Items[expectedPods-1])
			} else {
				expectedDel := manager.expectations.GetUIDs(getKey(rsSpec, t))
				if expectedDel.Len() != 1 {
					t.Fatalf("Waiting on unexpected number of deletes.")
				}
				nsName := strings.Split(expectedDel.List()[0], "/")
				lastPod := &v1.Pod{
					ObjectMeta: v1.ObjectMeta{
						Name:      nsName[1],
						Namespace: nsName[0],
						Labels:    rsSpec.Spec.Selector.MatchLabels,
					},
				}
				manager.podLister.Indexer.Delete(lastPod)
				manager.deletePod(lastPod)
			}
			pods.Items = pods.Items[expectedPods:]
		}

		// Confirm that we've created the right number of replicas
		activePods := int32(len(manager.podLister.Indexer.List()))
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, 2, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	labelMap := map[string]string{"foo": "bar"}
	rsSpec := newReplicaSet(2, labelMap)
	manager.rsLister.Indexer.Add(rsSpec)
	pods := newPodList(nil, 2, v1.PodPending, labelMap, rsSpec, "pod")
	manager.podLister.Indexer.Add(&pods.Items[0])
	postExpectationsPod := pods.Items[1]

	manager.expectations = controller.NewUIDTrackingControllerExpectations(FakeRSExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the
			// ReplicaSet will create a new replica because it doesn't see
			// this pod, but has fulfilled its expectations.
			manager.podLister.Indexer.Add(&postExpectationsPod)
		},
	})
	manager.syncReplicaSet(getKey(rsSpec, t))
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

func TestDeleteControllerAndExpectations(t *testing.T) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, 10, 0)
	manager.podListerSynced = alwaysReady

	rs := newReplicaSet(1, map[string]string{"foo": "bar"})
	manager.rsLister.Indexer.Add(rs)

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
	manager.rsLister.Indexer.Delete(rs)
	manager.syncReplicaSet(getKey(rs, t))

	if _, exists, err = manager.expectations.GetExpectations(rsKey); exists {
		t.Errorf("Found expectaions, expected none since the ReplicaSet has been deleted.")
	}

	// This should have no effect, since we've deleted the ReplicaSet.
	podExp.Add(-1, 0)
	manager.podLister.Indexer.Replace(make([]interface{}, 0), "0")
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	labelMap := map[string]string{"foo": "bar"}

	for i := 0; i < 5; i++ {
		func() {
			stopCh := make(chan struct{})
			defer close(stopCh)
			manager := testNewReplicaSetControllerFromClient(client, stopCh, 10, 0)
			manager.podListerSynced = alwaysReady

			// Create 10 ReplicaSets, shuffled them randomly and insert them into the ReplicaSet controller's store
			var controllers []*extensions.ReplicaSet
			for j := 1; j < 10; j++ {
				rsSpec := newReplicaSet(1, labelMap)
				rsSpec.CreationTimestamp = metav1.Date(2014, time.December, j, 0, 0, 0, 0, time.Local)
				rsSpec.Name = string(uuid.NewUUID())
				controllers = append(controllers, rsSpec)
			}
			shuffledControllers := shuffle(controllers)
			for j := range shuffledControllers {
				manager.rsLister.Indexer.Add(shuffledControllers[j])
			}
			// Add a pod and make sure only the oldest ReplicaSet is synced
			pods := newPodList(nil, 1, v1.PodPending, labelMap, controllers[0], "pod")
			rsKey := getKey(controllers[0], t)

			manager.addPod(&pods.Items[0])
			queueRS, _ := manager.queue.Get()
			if queueRS != rsKey {
				t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
			}
		}()
	}
}

func TestDeletionTimestamp(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	labelMap := map[string]string{"foo": "bar"}
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(c, stopCh, 10, 0)
	manager.podListerSynced = alwaysReady

	rs := newReplicaSet(1, labelMap)
	manager.rsLister.Indexer.Add(rs)
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
	secondPod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Namespace: pod.Namespace,
			Name:      "secondPod",
			Labels:    pod.Labels,
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
// and with garbageCollectorEnabled set to true
func setupManagerWithGCEnabled(stopCh chan struct{}, objs ...runtime.Object) (manager *ReplicaSetController, fakePodControl *controller.FakePodControl) {
	c := fakeclientset.NewSimpleClientset(objs...)
	fakePodControl = &controller.FakePodControl{}
	manager = testNewReplicaSetControllerFromClient(c, stopCh, BurstReplicas, 0)
	manager.garbageCollectorEnabled = true
	manager.podListerSynced = alwaysReady

	manager.podControl = fakePodControl
	return manager, fakePodControl
}

func TestDoNotPatchPodWithOtherControlRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	var trueVar = true
	otherControllerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "v1beta1", Kind: "ReplicaSet", Name: "AnotherRS", Controller: &trueVar}
	// add to podLister a matching Pod controlled by another controller. Expect no patch.
	pod := newPod("pod", rs, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{otherControllerReference}
	manager.podLister.Indexer.Add(pod)
	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	// because the matching pod already has a controller, so 2 pods should be created.
	validateSyncReplicaSet(t, fakePodControl, 2, 0, 0)
}

func TestPatchPodWithOtherOwnerRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	// add to podLister one more matching pod that doesn't have a controller
	// ref, but has an owner ref pointing to other object. Expect a patch to
	// take control of it.
	unrelatedOwnerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "batch/v1", Kind: "Job", Name: "Job"}
	pod := newPod("pod", rs, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{unrelatedOwnerReference}
	manager.podLister.Indexer.Add(pod)

	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	// 1 patch to take control of pod, and 1 create of new pod.
	validateSyncReplicaSet(t, fakePodControl, 1, 0, 1)
}

func TestPatchPodWithCorrectOwnerRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	// add to podLister a matching pod that has an ownerRef pointing to the rs,
	// but ownerRef.Controller is false. Expect a patch to take control it.
	rsOwnerReference := metav1.OwnerReference{UID: rs.UID, APIVersion: "v1", Kind: "ReplicaSet", Name: rs.Name}
	pod := newPod("pod", rs, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{rsOwnerReference}
	manager.podLister.Indexer.Add(pod)

	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	// 1 patch to take control of pod, and 1 create of new pod.
	validateSyncReplicaSet(t, fakePodControl, 1, 0, 1)
}

func TestPatchPodFails(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	// add to podLister two matching pods. Expect two patches to take control
	// them.
	manager.podLister.Indexer.Add(newPod("pod1", rs, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod2", rs, v1.PodRunning, nil))
	// let both patches fail. The rs controller will assume it fails to take
	// control of the pods and create new ones.
	fakePodControl.Err = fmt.Errorf("Fake Error")
	err := manager.syncReplicaSet(getKey(rs, t))
	if err == nil || err.Error() != "Fake Error" {
		t.Errorf("expected Fake Error, got %+v", err)
	}
	// 2 patches to take control of pod1 and pod2 (both fail), 2 creates.
	validateSyncReplicaSet(t, fakePodControl, 2, 0, 2)
}

func TestPatchExtraPodsThenDelete(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	// add to podLister three matching pods. Expect three patches to take control
	// them, and later delete one of them.
	manager.podLister.Indexer.Add(newPod("pod1", rs, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod2", rs, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod3", rs, v1.PodRunning, nil))
	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	// 3 patches to take control of the pods, and 1 deletion because there is an extra pod.
	validateSyncReplicaSet(t, fakePodControl, 0, 1, 3)
}

func TestUpdateLabelsRemoveControllerRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	manager.rsLister.Indexer.Add(rs)
	// put one pod in the podLister
	pod := newPod("pod", rs, v1.PodRunning, nil)
	pod.ResourceVersion = "1"
	var trueVar = true
	rsOwnerReference := metav1.OwnerReference{UID: rs.UID, APIVersion: "v1beta1", Kind: "ReplicaSet", Name: rs.Name, Controller: &trueVar}
	pod.OwnerReferences = []metav1.OwnerReference{rsOwnerReference}
	updatedPod := *pod
	// reset the labels
	updatedPod.Labels = make(map[string]string)
	updatedPod.ResourceVersion = "2"
	// add the updatedPod to the store. This is consistent with the behavior of
	// the Informer: Informer updates the store before call the handler
	// (updatePod() in this case).
	manager.podLister.Indexer.Add(&updatedPod)
	// send a update of the same pod with modified labels
	manager.updatePod(pod, &updatedPod)
	// verifies that rs is added to the queue
	rsKey := getKey(rs, t)
	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
	manager.queue.Done(queueRS)
	err := manager.syncReplicaSet(rsKey)
	if err != nil {
		t.Fatal(err)
	}
	// expect 1 patch to be sent to remove the controllerRef for the pod.
	// expect 2 creates because the *(rs.Spec.Replicas)=2 and there exists no
	// matching pod.
	validateSyncReplicaSet(t, fakePodControl, 2, 0, 1)
	fakePodControl.Clear()
}

func TestUpdateSelectorControllerRef(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	// put 2 pods in the podLister
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, labelMap, rs, "pod")
	// update the RS so that its selector no longer matches the pods
	updatedRS := *rs
	updatedRS.Spec.Selector.MatchLabels = map[string]string{"foo": "baz"}
	// put the updatedRS into the store. This is consistent with the behavior of
	// the Informer: Informer updates the store before call the handler
	// (updateRS() in this case).
	manager.rsLister.Indexer.Add(&updatedRS)
	manager.updateRS(rs, &updatedRS)
	// verifies that the rs is added to the queue
	rsKey := getKey(rs, t)
	queueRS, _ := manager.queue.Get()
	if queueRS != rsKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rsKey, queueRS)
	}
	manager.queue.Done(queueRS)
	err := manager.syncReplicaSet(rsKey)
	if err != nil {
		t.Fatal(err)
	}
	// expect 2 patches to be sent to remove the controllerRef for the pods.
	// expect 2 creates because the *(rc.Spec.Replicas)=2 and there exists no
	// matching pod.
	validateSyncReplicaSet(t, fakePodControl, 2, 0, 2)
	fakePodControl.Clear()
}

// RS controller shouldn't adopt or create more pods if the rc is about to be
// deleted.
func TestDoNotAdoptOrCreateIfBeingDeleted(t *testing.T) {
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager, fakePodControl := setupManagerWithGCEnabled(stopCh, rs)
	now := metav1.Now()
	rs.DeletionTimestamp = &now
	manager.rsLister.Indexer.Add(rs)
	pod1 := newPod("pod1", rs, v1.PodRunning, nil)
	manager.podLister.Indexer.Add(pod1)

	// no patch, no create
	err := manager.syncReplicaSet(getKey(rs, t))
	if err != nil {
		t.Fatal(err)
	}
	validateSyncReplicaSet(t, fakePodControl, 0, 0, 0)
}

func TestReadyReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    200,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, ReadyReplicas: 0, AvailableReplicas: 0, ObservedGeneration: 1}
	rs.Generation = 1
	manager.rsLister.Indexer.Add(rs)

	newPodList(manager.podLister.Indexer, 2, v1.PodPending, labelMap, rs, "pod")
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, labelMap, rs, "pod")

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.SetResponseBody(response)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicaSet(getKey(rs, t))

	// ReadyReplicas should go from 0 to 2.
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, ReadyReplicas: 2, AvailableReplicas: 2, ObservedGeneration: 1}

	decRs := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &decRs)
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
}

func TestAvailableReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:    200,
		ResponseBody:  "{}",
		SkipRequestFn: skipListerFunc,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	manager := testNewReplicaSetControllerFromClient(client, stopCh, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	labelMap := map[string]string{"foo": "bar"}
	rs := newReplicaSet(2, labelMap)
	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, ReadyReplicas: 0, AvailableReplicas: 0, ObservedGeneration: 1}
	rs.Generation = 1
	// minReadySeconds set to 15s
	rs.Spec.MinReadySeconds = 15
	manager.rsLister.Indexer.Add(rs)

	// First pod becomes ready 20s ago
	moment := metav1.Time{Time: time.Now().Add(-2e10)}
	pod := newPod("pod", rs, v1.PodRunning, &moment)
	manager.podLister.Indexer.Add(pod)

	// Second pod becomes ready now
	otherMoment := metav1.Now()
	otherPod := newPod("otherPod", rs, v1.PodRunning, &otherMoment)
	manager.podLister.Indexer.Add(otherPod)

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Extensions.Codec(), &extensions.ReplicaSet{})
	fakeHandler.SetResponseBody(response)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// The controller should see only one available pod.
	manager.syncReplicaSet(getKey(rs, t))

	rs.Status = extensions.ReplicaSetStatus{Replicas: 2, ReadyReplicas: 2, AvailableReplicas: 1, ObservedGeneration: 1}

	decRs := runtime.EncodeOrDie(testapi.Extensions.Codec(), rs)
	fakeHandler.ValidateRequest(t, testapi.Extensions.ResourcePath(replicaSetResourceName(), rs.Namespace, rs.Name)+"/status", "PUT", &decRs)
	validateSyncReplicaSet(t, &fakePodControl, 0, 0, 0)
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
