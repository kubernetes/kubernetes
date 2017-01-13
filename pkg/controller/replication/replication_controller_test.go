/*
Copyright 2014 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replication

import (
	"fmt"
	"math/rand"
	"net/http/httptest"
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

var alwaysReady = func() bool { return true }

func getKey(rc *v1.ReplicationController, t *testing.T) string {
	if key, err := controller.KeyFunc(rc); err != nil {
		t.Errorf("Unexpected error getting key for rc %v: %v", rc.Name, err)
		return ""
	} else {
		return key
	}
}

func newReplicationController(replicas int) *v1.ReplicationController {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String()},
		ObjectMeta: v1.ObjectMeta{
			UID:             uuid.NewUUID(),
			Name:            "foobar",
			Namespace:       v1.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func() *int32 { i := int32(replicas); return &i }(),
			Selector: map[string]string{"foo": "bar"},
			Template: &v1.PodTemplateSpec{
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
	return rc
}

// create a pod with the given phase for the given rc (same selectors and namespace).
func newPod(name string, rc *v1.ReplicationController, status v1.PodPhase, lastTransitionTime *metav1.Time) *v1.Pod {
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
			Labels:    rc.Spec.Selector,
			Namespace: rc.Namespace,
		},
		Status: v1.PodStatus{Phase: status, Conditions: conditions},
	}
}

// create count pods with the given phase for the given rc (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status v1.PodPhase, rc *v1.ReplicationController, name string) *v1.PodList {
	pods := []v1.Pod{}
	var trueVar = true
	controllerReference := metav1.OwnerReference{UID: rc.UID, APIVersion: "v1", Kind: "ReplicationController", Name: rc.Name, Controller: &trueVar}
	for i := 0; i < count; i++ {
		pod := newPod(fmt.Sprintf("%s%d", name, i), rc, status, nil)
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

func validateSyncReplication(t *testing.T, fakePodControl *controller.FakePodControl, expectedCreates, expectedDeletes, expectedPatches int) {
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

func replicationControllerResourceName() string {
	return "replicationcontrollers"
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func TestSyncReplicationControllerDoesNothing(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// 2 running pods, a controller with 2 replicas, sync is a no-op
	controllerSpec := newReplicationController(2)
	manager.rcLister.Indexer.Add(controllerSpec)
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, controllerSpec, "pod")

	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
}

func TestSyncReplicationControllerDeletes(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	// 2 running pods and a controller with 1 replica, one pod delete expected
	controllerSpec := newReplicationController(1)
	manager.rcLister.Indexer.Add(controllerSpec)
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, controllerSpec, "pod")

	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 1, 0)
}

func TestDeleteFinalStateUnknown(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	received := make(chan string)
	manager.syncHandler = func(key string) error {
		received <- key
		return nil
	}

	// The DeletedFinalStateUnknown object should cause the rc manager to insert
	// the controller matching the selectors of the deleted pod into the work queue.
	controllerSpec := newReplicationController(1)
	manager.rcLister.Indexer.Add(controllerSpec)
	pods := newPodList(nil, 1, v1.PodRunning, controllerSpec, "pod")
	manager.deletePod(cache.DeletedFinalStateUnknown{Key: "foo", Obj: &pods.Items[0]})

	go manager.worker()

	expected := getKey(controllerSpec, t)
	select {
	case key := <-received:
		if key != expected {
			t.Errorf("Unexpected sync all for rc %v, expected %v", key, expected)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Processing DeleteFinalStateUnknown took longer than expected")
	}
}

func TestSyncReplicationControllerCreates(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// A controller with 2 replicas and no pods in the store, 2 creates expected
	rc := newReplicationController(2)
	manager.rcLister.Indexer.Add(rc)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(rc, t))
	validateSyncReplication(t, &fakePodControl, 2, 0, 0)
}

func TestStatusUpdatesWithoutReplicasChange(t *testing.T) {
	// Setup a fake server to listen for requests, and run the rc manager in steady state
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Steady state for the replication controller, no Status.Replicas updates expected
	activePods := 5
	rc := newReplicationController(activePods)
	manager.rcLister.Indexer.Add(rc)
	rc.Status = v1.ReplicationControllerStatus{Replicas: int32(activePods), ReadyReplicas: int32(activePods), AvailableReplicas: int32(activePods)}
	newPodList(manager.podLister.Indexer, activePods, v1.PodRunning, rc, "pod")

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.syncReplicationController(getKey(rc, t))

	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
	if fakeHandler.RequestReceived != nil {
		t.Errorf("Unexpected update when pods and rcs are in a steady state")
	}

	// This response body is just so we don't err out decoding the http response, all
	// we care about is the request body sent below.
	response := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.ReplicationController{})
	fakeHandler.ResponseBody = response

	rc.Generation = rc.Generation + 1
	manager.syncReplicationController(getKey(rc, t))

	rc.Status.ObservedGeneration = rc.Generation
	updatedRc := runtime.EncodeOrDie(testapi.Default.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name)+"/status", "PUT", &updatedRc)
}

func TestControllerUpdateReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Insufficient number of pods in the system, and Status.Replicas is wrong;
	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	rc := newReplicationController(5)
	manager.rcLister.Indexer.Add(rc)
	rc.Status = v1.ReplicationControllerStatus{Replicas: 2, FullyLabeledReplicas: 6, ReadyReplicas: 2, AvailableReplicas: 2, ObservedGeneration: 0}
	rc.Generation = 1
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, rc, "pod")
	rcCopy := *rc
	extraLabelMap := map[string]string{"foo": "bar", "extraKey": "extraValue"}
	rcCopy.Spec.Selector = extraLabelMap
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, &rcCopy, "podWithExtraLabel")

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.ReplicationController{})
	fakeHandler.ResponseBody = response

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicationController(getKey(rc, t))

	// 1. Status.Replicas should go up from 2->4 even though we created 5-4=1 pod.
	// 2. Status.FullyLabeledReplicas should equal to the number of pods that
	// has the extra labels, i.e., 2.
	// 3. Every update to the status should include the Generation of the spec.
	rc.Status = v1.ReplicationControllerStatus{Replicas: 4, ReadyReplicas: 4, AvailableReplicas: 4, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Default.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name)+"/status", "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 1, 0, 0)
}

func TestSyncReplicationControllerDormancy(t *testing.T) {
	// Setup a test server so we can lie about the current state of pods
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "{}",
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)
	manager.rcLister.Indexer.Add(controllerSpec)
	newPodList(manager.podLister.Indexer, 1, v1.PodRunning, controllerSpec, "pod")

	// Creates a replica and sets expectations
	controllerSpec.Status.Replicas = 1
	controllerSpec.Status.ReadyReplicas = 1
	controllerSpec.Status.AvailableReplicas = 1
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 1, 0, 0)

	// Expectations prevents replicas but not an update on status
	controllerSpec.Status.Replicas = 0
	controllerSpec.Status.ReadyReplicas = 0
	controllerSpec.Status.AvailableReplicas = 0
	fakePodControl.Clear()
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)

	// Get the key for the controller
	rcKey, err := controller.KeyFunc(controllerSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", controllerSpec, err)
	}

	// Lowering expectations should lead to a sync that creates a replica, however the
	// fakePodControl error will prevent this, leaving expectations at 0, 0.
	manager.expectations.CreationObserved(rcKey)
	controllerSpec.Status.Replicas = 1
	controllerSpec.Status.ReadyReplicas = 1
	controllerSpec.Status.AvailableReplicas = 1
	fakePodControl.Clear()
	fakePodControl.Err = fmt.Errorf("Fake Error")

	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 1, 0, 0)

	// This replica should not need a Lowering of expectations, since the previous create failed
	fakePodControl.Clear()
	fakePodControl.Err = nil
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 1, 0, 0)

	// 2 PUT for the rc status during dormancy window.
	// Note that the pod creates go through pod control so they're not recorded.
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestPodControllerLookup(t *testing.T) {
	manager := NewReplicationManagerFromClient(clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}}), controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady
	testCases := []struct {
		inRCs     []*v1.ReplicationController
		pod       *v1.Pod
		outRCName string
	}{
		// pods without labels don't match any rcs
		{
			inRCs: []*v1.ReplicationController{
				{ObjectMeta: v1.ObjectMeta{Name: "basic"}}},
			pod:       &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo1", Namespace: v1.NamespaceAll}},
			outRCName: "",
		},
		// Matching labels, not namespace
		{
			inRCs: []*v1.ReplicationController{
				{
					ObjectMeta: v1.ObjectMeta{Name: "foo"},
					Spec: v1.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo2", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRCName: "",
		},
		// Matching ns and labels returns the key to the rc, not the rc name
		{
			inRCs: []*v1.ReplicationController{
				{
					ObjectMeta: v1.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: v1.ReplicationControllerSpec{
						Selector: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo3", Namespace: "ns", Labels: map[string]string{"foo": "bar"}}},
			outRCName: "bar",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inRCs {
			manager.rcLister.Indexer.Add(r)
		}
		if rc := manager.getPodController(c.pod); rc != nil {
			if c.outRCName != rc.Name {
				t.Errorf("Got controller %+v expected %+v", rc.Name, c.outRCName)
			}
		} else if c.outRCName != "" {
			t.Errorf("Expected a controller %v pod %v, found none", c.outRCName, c.pod.Name)
		}
	}
}

func TestWatchControllers(t *testing.T) {
	fakeWatch := watch.NewFake()
	c := &fake.Clientset{}
	c.AddWatchReactor("replicationcontrollers", core.DefaultWatchReactor(fakeWatch, nil))
	stopCh := make(chan struct{})
	defer close(stopCh)
	informers := informers.NewSharedInformerFactory(c, nil, controller.NoResyncPeriodFunc())
	podInformer := informers.Pods().Informer()
	rcInformer := informers.ReplicationControllers().Informer()
	manager := NewReplicationManager(podInformer, rcInformer, c, BurstReplicas, 0, false)
	informers.Start(stopCh)
	manager.podListerSynced = alwaysReady

	var testControllerSpec v1.ReplicationController
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler. The handler validates the received controller
	// and closes the received channel to indicate that the test can finish.
	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rcLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		controllerSpec := *obj.(*v1.ReplicationController)
		if !api.Semantic.DeepDerivative(controllerSpec, testControllerSpec) {
			t.Errorf("Expected %#v, but got %#v", testControllerSpec, controllerSpec)
		}
		close(received)
		return nil
	}

	// Start only the rc watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	testControllerSpec.Name = "foo"
	fakeWatch.Add(&testControllerSpec)

	select {
	case <-received:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("unexpected timeout from result channel")
	}
}

func TestWatchPods(t *testing.T) {
	fakeWatch := watch.NewFake()
	c := &fake.Clientset{}
	c.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Put one rc and one pod into the controller's stores
	testControllerSpec := newReplicationController(1)
	manager.rcLister.Indexer.Add(testControllerSpec)
	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing rc and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.rcLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		controllerSpec := obj.(*v1.ReplicationController)
		if !api.Semantic.DeepDerivative(controllerSpec, testControllerSpec) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testControllerSpec, controllerSpec)
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right rc.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.internalPodInformer.Run(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(nil, 1, v1.PodRunning, testControllerSpec, "pod")
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
	manager := NewReplicationManagerFromClient(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	received := make(chan string)

	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.rcLister.Indexer.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find controller under key %v", key)
		}
		received <- obj.(*v1.ReplicationController).Name
		return nil
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	// Put 2 rcs and one pod into the controller's stores
	testControllerSpec1 := newReplicationController(1)
	manager.rcLister.Indexer.Add(testControllerSpec1)
	testControllerSpec2 := *testControllerSpec1
	testControllerSpec2.Spec.Selector = map[string]string{"bar": "foo"}
	testControllerSpec2.Name = "barfoo"
	manager.rcLister.Indexer.Add(&testControllerSpec2)

	// case 1: We put in the podLister a pod with labels matching
	// testControllerSpec1, then update its labels to match testControllerSpec2.
	// We expect to receive a sync request for both controllers.
	pod1 := newPodList(manager.podLister.Indexer, 1, v1.PodRunning, testControllerSpec1, "pod").Items[0]
	pod1.ResourceVersion = "1"
	pod2 := pod1
	pod2.Labels = testControllerSpec2.Spec.Selector
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected := sets.NewString(testControllerSpec1.Name, testControllerSpec2.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for controllers within 100ms each")
		}
	}

	// case 2: pod1 in the podLister has labels matching testControllerSpec1.
	// We update its labels to match no replication controller.  We expect to
	// receive a sync request for testControllerSpec1.
	pod2.Labels = make(map[string]string)
	pod2.ResourceVersion = "2"
	manager.updatePod(&pod1, &pod2)
	expected = sets.NewString(testControllerSpec1.Name)
	for _, name := range expected.List() {
		t.Logf("Expecting update for %+v", name)
		select {
		case got := <-received:
			if !expected.Has(got) {
				t.Errorf("Expected keys %#v got %v", expected, got)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Expected update notifications for controllers within 100ms each")
		}
	}
}

func TestControllerUpdateRequeue(t *testing.T) {
	// This server should force a requeue of the controller because it fails to update status.Replicas.
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	rc := newReplicationController(1)
	manager.rcLister.Indexer.Add(rc)
	rc.Status = v1.ReplicationControllerStatus{Replicas: 2}
	newPodList(manager.podLister.Indexer, 1, v1.PodRunning, rc, "pod")

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// an error from the sync function will be requeued, check to make sure we returned an error
	if err := manager.syncReplicationController(getKey(rc, t)); err == nil {
		t.Errorf("missing error for requeue")
	}

	// 1 Update and 1 GET, both of which fail
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestControllerUpdateStatusWithFailure(t *testing.T) {
	rc := newReplicationController(1)
	c := &fake.Clientset{}
	c.AddReactor("get", "replicationcontrollers", func(action core.Action) (bool, runtime.Object, error) {
		return true, rc, nil
	})
	c.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, &v1.ReplicationController{}, fmt.Errorf("Fake error")
	})
	fakeRCClient := c.Core().ReplicationControllers("default")
	numReplicas := int32(10)
	status := v1.ReplicationControllerStatus{Replicas: numReplicas}
	updateReplicationControllerStatus(fakeRCClient, *rc, status)
	updates, gets := 0, 0
	for _, a := range c.Actions() {
		if a.GetResource().Resource != "replicationcontrollers" {
			t.Errorf("Unexpected action %+v", a)
			continue
		}

		switch action := a.(type) {
		case core.GetAction:
			gets++
			// Make sure the get is for the right rc even though the update failed.
			if action.GetName() != rc.Name {
				t.Errorf("Expected get for rc %v, got %+v instead", rc.Name, action.GetName())
			}
		case core.UpdateAction:
			updates++
			// Confirm that the update has the right status.Replicas even though the Get
			// returned an rc with replicas=1.
			if c, ok := action.GetObject().(*v1.ReplicationController); !ok {
				t.Errorf("Expected an rc as the argument to update, got %T", c)
			} else if c.Status.Replicas != numReplicas {
				t.Errorf("Expected update for rc to contain replicas %v, got %v instead",
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
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, burstReplicas, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(numReplicas)
	manager.rcLister.Indexer.Add(controllerSpec)

	expectedPods := 0
	pods := newPodList(nil, numReplicas, v1.PodPending, controllerSpec, "pod")

	rcKey, err := controller.KeyFunc(controllerSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", controllerSpec, err)
	}

	// Size up the controller, then size it down, and confirm the expected create/delete pattern
	for _, replicas := range []int{numReplicas, 0} {

		*(controllerSpec.Spec.Replicas) = int32(replicas)
		manager.rcLister.Indexer.Add(controllerSpec)

		for i := 0; i < numReplicas; i += burstReplicas {
			manager.syncReplicationController(getKey(controllerSpec, t))

			// The store accrues active pods. It's also used by the rc to determine how many
			// replicas to create.
			activePods := len(manager.podLister.Indexer.List())
			if replicas != 0 {
				// This is the number of pods currently "in flight". They were created by the rc manager above,
				// which then puts the rc to sleep till all of them have been observed.
				expectedPods = replicas - activePods
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				// This validates the rc manager sync actually created pods
				validateSyncReplication(t, &fakePodControl, expectedPods, 0, 0)

				// This simulates the watch events for all but 1 of the expected pods.
				// None of these should wake the controller because it has expectations==BurstReplicas.
				for i := 0; i < expectedPods-1; i++ {
					manager.podLister.Indexer.Add(&pods.Items[i])
					manager.addPod(&pods.Items[i])
				}

				podExp, exists, err := manager.expectations.GetExpectations(rcKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for rc.")
				}
				if add, _ := podExp.GetExpectations(); add != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			} else {
				expectedPods = (replicas - activePods) * -1
				if expectedPods > burstReplicas {
					expectedPods = burstReplicas
				}
				validateSyncReplication(t, &fakePodControl, 0, expectedPods, 0)

				// To accurately simulate a watch we must delete the exact pods
				// the rc is waiting for.
				expectedDels := manager.expectations.GetUIDs(getKey(controllerSpec, t))
				podsToDelete := []*v1.Pod{}
				for _, key := range expectedDels.List() {
					nsName := strings.Split(key, "/")
					podsToDelete = append(podsToDelete, &v1.Pod{
						ObjectMeta: v1.ObjectMeta{
							Name:      nsName[1],
							Namespace: nsName[0],
							Labels:    controllerSpec.Spec.Selector,
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
				podExp, exists, err := manager.expectations.GetExpectations(rcKey)
				if !exists || err != nil {
					t.Fatalf("Did not find expectations for rc.")
				}
				if _, del := podExp.GetExpectations(); del != 1 {
					t.Fatalf("Expectations are wrong %v", podExp)
				}
			}

			// Check that the rc didn't take any action for all the above pods
			fakePodControl.Clear()
			manager.syncReplicationController(getKey(controllerSpec, t))
			validateSyncReplication(t, &fakePodControl, 0, 0, 0)

			// Create/Delete the last pod
			// The last add pod will decrease the expectation of the rc to 0,
			// which will cause it to create/delete the remaining replicas up to burstReplicas.
			if replicas != 0 {
				manager.podLister.Indexer.Add(&pods.Items[expectedPods-1])
				manager.addPod(&pods.Items[expectedPods-1])
			} else {
				expectedDel := manager.expectations.GetUIDs(getKey(controllerSpec, t))
				if expectedDel.Len() != 1 {
					t.Fatalf("Waiting on unexpected number of deletes.")
				}
				nsName := strings.Split(expectedDel.List()[0], "/")
				lastPod := &v1.Pod{
					ObjectMeta: v1.ObjectMeta{
						Name:      nsName[1],
						Namespace: nsName[0],
						Labels:    controllerSpec.Spec.Selector,
					},
				}
				manager.podLister.Indexer.Delete(lastPod)
				manager.deletePod(lastPod)
			}
			pods.Items = pods.Items[expectedPods:]
		}

		// Confirm that we've created the right number of replicas
		activePods := int32(len(manager.podLister.Indexer.List()))
		if activePods != *(controllerSpec.Spec.Replicas) {
			t.Fatalf("Unexpected number of active pods, expected %d, got %d", *(controllerSpec.Spec.Replicas), activePods)
		}
		// Replenish the pod list, since we cut it down sizing up
		pods = newPodList(nil, replicas, v1.PodRunning, controllerSpec, "pod")
	}
}

func TestControllerBurstReplicas(t *testing.T) {
	doTestControllerBurstReplicas(t, 5, 30)
	doTestControllerBurstReplicas(t, 5, 12)
	doTestControllerBurstReplicas(t, 3, 2)
}

type FakeRCExpectations struct {
	*controller.ControllerExpectations
	satisfied    bool
	expSatisfied func()
}

func (fe FakeRCExpectations) SatisfiedExpectations(controllerKey string) bool {
	fe.expSatisfied()
	return fe.satisfied
}

// TestRCSyncExpectations tests that a pod cannot sneak in between counting active pods
// and checking expectations.
func TestRCSyncExpectations(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	fakePodControl := controller.FakePodControl{}
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, 2, 0)
	manager.podListerSynced = alwaysReady
	manager.podControl = &fakePodControl

	controllerSpec := newReplicationController(2)
	manager.rcLister.Indexer.Add(controllerSpec)
	pods := newPodList(nil, 2, v1.PodPending, controllerSpec, "pod")
	manager.podLister.Indexer.Add(&pods.Items[0])
	postExpectationsPod := pods.Items[1]

	manager.expectations = controller.NewUIDTrackingControllerExpectations(FakeRCExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the rc
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			manager.podLister.Indexer.Add(&postExpectationsPod)
		},
	})
	manager.syncReplicationController(getKey(controllerSpec, t))
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
}

func TestDeleteControllerAndExpectations(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, 10, 0)
	manager.podListerSynced = alwaysReady

	rc := newReplicationController(1)
	manager.rcLister.Indexer.Add(rc)

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// This should set expectations for the rc
	manager.syncReplicationController(getKey(rc, t))
	validateSyncReplication(t, &fakePodControl, 1, 0, 0)
	fakePodControl.Clear()

	// Get the RC key
	rcKey, err := controller.KeyFunc(rc)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rc, err)
	}

	// This is to simulate a concurrent addPod, that has a handle on the expectations
	// as the controller deletes it.
	podExp, exists, err := manager.expectations.GetExpectations(rcKey)
	if !exists || err != nil {
		t.Errorf("No expectations found for rc")
	}
	manager.rcLister.Indexer.Delete(rc)
	manager.syncReplicationController(getKey(rc, t))

	if _, exists, err = manager.expectations.GetExpectations(rcKey); exists {
		t.Errorf("Found expectaions, expected none since the rc has been deleted.")
	}

	// This should have no effect, since we've deleted the rc.
	podExp.Add(-1, 0)
	manager.podLister.Indexer.Replace(make([]interface{}, 0), "0")
	manager.syncReplicationController(getKey(rc, t))
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
}

// shuffle returns a new shuffled list of container controllers.
func shuffle(controllers []*v1.ReplicationController) []*v1.ReplicationController {
	numControllers := len(controllers)
	randIndexes := rand.Perm(numControllers)
	shuffled := make([]*v1.ReplicationController, numControllers)
	for i := 0; i < numControllers; i++ {
		shuffled[i] = controllers[randIndexes[i]]
	}
	return shuffled
}

func TestOverlappingRCs(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	for i := 0; i < 5; i++ {
		manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, 10, 0)
		manager.podListerSynced = alwaysReady

		// Create 10 rcs, shuffled them randomly and insert them into the rc manager's store
		var controllers []*v1.ReplicationController
		for j := 1; j < 10; j++ {
			controllerSpec := newReplicationController(1)
			controllerSpec.CreationTimestamp = metav1.Date(2014, time.December, j, 0, 0, 0, 0, time.Local)
			controllerSpec.Name = string(uuid.NewUUID())
			controllers = append(controllers, controllerSpec)
		}
		shuffledControllers := shuffle(controllers)
		for j := range shuffledControllers {
			manager.rcLister.Indexer.Add(shuffledControllers[j])
		}
		// Add a pod and make sure only the oldest rc is synced
		pods := newPodList(nil, 1, v1.PodPending, controllers[0], "pod")
		rcKey := getKey(controllers[0], t)

		manager.addPod(&pods.Items[0])
		queueRC, _ := manager.queue.Get()
		if queueRC != rcKey {
			t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
		}
	}
}

func TestDeletionTimestamp(t *testing.T) {
	c := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, 10, 0)
	manager.podListerSynced = alwaysReady

	controllerSpec := newReplicationController(1)
	manager.rcLister.Indexer.Add(controllerSpec)
	rcKey, err := controller.KeyFunc(controllerSpec)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", controllerSpec, err)
	}
	pod := newPodList(nil, 1, v1.PodPending, controllerSpec, "pod").Items[0]
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	pod.ResourceVersion = "1"
	manager.expectations.ExpectDeletions(rcKey, []string{controller.PodKey(&pod)})

	// A pod added with a deletion timestamp should decrement deletions, not creations.
	manager.addPod(&pod)

	queueRC, _ := manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}
	manager.queue.Done(rcKey)

	podExp, exists, err := manager.expectations.GetExpectations(rcKey)
	if !exists || err != nil || !podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// An update from no deletion timestamp to having one should be treated
	// as a deletion.
	oldPod := newPodList(nil, 1, v1.PodPending, controllerSpec, "pod").Items[0]
	oldPod.ResourceVersion = "2"
	manager.expectations.ExpectDeletions(rcKey, []string{controller.PodKey(&pod)})
	manager.updatePod(&oldPod, &pod)

	queueRC, _ = manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}
	manager.queue.Done(rcKey)

	podExp, exists, err = manager.expectations.GetExpectations(rcKey)
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
	manager.expectations.ExpectDeletions(rcKey, []string{controller.PodKey(secondPod)})
	oldPod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	oldPod.ResourceVersion = "2"
	manager.updatePod(&oldPod, &pod)

	podExp, exists, err = manager.expectations.GetExpectations(rcKey)
	if !exists || err != nil || podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// A pod with a non-nil deletion timestamp should also be ignored by the
	// delete handler, because it's already been counted in the update.
	manager.deletePod(&pod)
	podExp, exists, err = manager.expectations.GetExpectations(rcKey)
	if !exists || err != nil || podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}

	// Deleting the second pod should clear expectations.
	manager.deletePod(secondPod)

	queueRC, _ = manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}
	manager.queue.Done(rcKey)

	podExp, exists, err = manager.expectations.GetExpectations(rcKey)
	if !exists || err != nil || !podExp.Fulfilled() {
		t.Fatalf("Wrong expectations %#v", podExp)
	}
}

func BenchmarkGetPodControllerMultiNS(b *testing.B) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)

	const nsNum = 1000

	pods := []v1.Pod{}
	for i := 0; i < nsNum; i++ {
		ns := fmt.Sprintf("ns-%d", i)
		for j := 0; j < 10; j++ {
			rcName := fmt.Sprintf("rc-%d", j)
			for k := 0; k < 10; k++ {
				podName := fmt.Sprintf("pod-%d-%d", j, k)
				pods = append(pods, v1.Pod{
					ObjectMeta: v1.ObjectMeta{
						Name:      podName,
						Namespace: ns,
						Labels:    map[string]string{"rcName": rcName},
					},
				})
			}
		}
	}

	for i := 0; i < nsNum; i++ {
		ns := fmt.Sprintf("ns-%d", i)
		for j := 0; j < 10; j++ {
			rcName := fmt.Sprintf("rc-%d", j)
			manager.rcLister.Indexer.Add(&v1.ReplicationController{
				ObjectMeta: v1.ObjectMeta{Name: rcName, Namespace: ns},
				Spec: v1.ReplicationControllerSpec{
					Selector: map[string]string{"rcName": rcName},
				},
			})
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, pod := range pods {
			manager.getPodController(&pod)
		}
	}
}

func BenchmarkGetPodControllerSingleNS(b *testing.B) {
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(client, controller.NoResyncPeriodFunc, BurstReplicas, 0)

	const rcNum = 1000
	const replicaNum = 3

	pods := []v1.Pod{}
	for i := 0; i < rcNum; i++ {
		rcName := fmt.Sprintf("rc-%d", i)
		for j := 0; j < replicaNum; j++ {
			podName := fmt.Sprintf("pod-%d-%d", i, j)
			pods = append(pods, v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name:      podName,
					Namespace: "foo",
					Labels:    map[string]string{"rcName": rcName},
				},
			})
		}
	}

	for i := 0; i < rcNum; i++ {
		rcName := fmt.Sprintf("rc-%d", i)
		manager.rcLister.Indexer.Add(&v1.ReplicationController{
			ObjectMeta: v1.ObjectMeta{Name: rcName, Namespace: "foo"},
			Spec: v1.ReplicationControllerSpec{
				Selector: map[string]string{"rcName": rcName},
			},
		})
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, pod := range pods {
			manager.getPodController(&pod)
		}
	}
}

// setupManagerWithGCEnabled creates a RC manager with a fakePodControl and with garbageCollectorEnabled set to true
func setupManagerWithGCEnabled(objs ...runtime.Object) (manager *ReplicationManager, fakePodControl *controller.FakePodControl) {
	c := fakeclientset.NewSimpleClientset(objs...)
	fakePodControl = &controller.FakePodControl{}
	manager = NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.garbageCollectorEnabled = true
	manager.podListerSynced = alwaysReady
	manager.podControl = fakePodControl
	return manager, fakePodControl
}

func TestDoNotPatchPodWithOtherControlRef(t *testing.T) {
	manager, fakePodControl := setupManagerWithGCEnabled()
	rc := newReplicationController(2)
	manager.rcLister.Indexer.Add(rc)
	var trueVar = true
	otherControllerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "v1", Kind: "ReplicationController", Name: "AnotherRC", Controller: &trueVar}
	// add to podLister a matching Pod controlled by another controller. Expect no patch.
	pod := newPod("pod", rc, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{otherControllerReference}
	manager.podLister.Indexer.Add(pod)
	err := manager.syncReplicationController(getKey(rc, t))
	if err != nil {
		t.Fatal(err)
	}
	// because the matching pod already has a controller, so 2 pods should be created.
	validateSyncReplication(t, fakePodControl, 2, 0, 0)
}

func TestPatchPodWithOtherOwnerRef(t *testing.T) {
	rc := newReplicationController(2)
	manager, fakePodControl := setupManagerWithGCEnabled(rc)
	manager.rcLister.Indexer.Add(rc)
	// add to podLister one more matching pod that doesn't have a controller
	// ref, but has an owner ref pointing to other object. Expect a patch to
	// take control of it.
	unrelatedOwnerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "batch/v1", Kind: "Job", Name: "Job"}
	pod := newPod("pod", rc, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{unrelatedOwnerReference}
	manager.podLister.Indexer.Add(pod)

	err := manager.syncReplicationController(getKey(rc, t))
	if err != nil {
		t.Fatal(err)
	}
	// 1 patch to take control of pod, and 1 create of new pod.
	validateSyncReplication(t, fakePodControl, 1, 0, 1)
}

func TestPatchPodWithCorrectOwnerRef(t *testing.T) {
	rc := newReplicationController(2)
	manager, fakePodControl := setupManagerWithGCEnabled(rc)
	manager.rcLister.Indexer.Add(rc)
	// add to podLister a matching pod that has an ownerRef pointing to the rc,
	// but ownerRef.Controller is false. Expect a patch to take control it.
	rcOwnerReference := metav1.OwnerReference{UID: rc.UID, APIVersion: "v1", Kind: "ReplicationController", Name: rc.Name}
	pod := newPod("pod", rc, v1.PodRunning, nil)
	pod.OwnerReferences = []metav1.OwnerReference{rcOwnerReference}
	manager.podLister.Indexer.Add(pod)

	err := manager.syncReplicationController(getKey(rc, t))
	if err != nil {
		t.Fatal(err)
	}
	// 1 patch to take control of pod, and 1 create of new pod.
	validateSyncReplication(t, fakePodControl, 1, 0, 1)
}

func TestPatchPodFails(t *testing.T) {
	rc := newReplicationController(2)
	manager, fakePodControl := setupManagerWithGCEnabled(rc)
	manager.rcLister.Indexer.Add(rc)
	// add to podLister two matching pods. Expect two patches to take control
	// them.
	manager.podLister.Indexer.Add(newPod("pod1", rc, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod2", rc, v1.PodRunning, nil))
	// let both patches fail. The rc manager will assume it fails to take
	// control of the pods and create new ones.
	fakePodControl.Err = fmt.Errorf("Fake Error")
	err := manager.syncReplicationController(getKey(rc, t))
	if err == nil || err.Error() != "Fake Error" {
		t.Fatalf("expected Fake Error, got %v", err)
	}
	// 2 patches to take control of pod1 and pod2 (both fail), 2 creates.
	validateSyncReplication(t, fakePodControl, 2, 0, 2)
}

func TestPatchExtraPodsThenDelete(t *testing.T) {
	rc := newReplicationController(2)
	manager, fakePodControl := setupManagerWithGCEnabled(rc)
	manager.rcLister.Indexer.Add(rc)
	// add to podLister three matching pods. Expect three patches to take control
	// them, and later delete one of them.
	manager.podLister.Indexer.Add(newPod("pod1", rc, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod2", rc, v1.PodRunning, nil))
	manager.podLister.Indexer.Add(newPod("pod3", rc, v1.PodRunning, nil))
	err := manager.syncReplicationController(getKey(rc, t))
	if err != nil {
		t.Fatal(err)
	}
	// 3 patches to take control of the pods, and 1 deletion because there is an extra pod.
	validateSyncReplication(t, fakePodControl, 0, 1, 3)
}

func TestUpdateLabelsRemoveControllerRef(t *testing.T) {
	manager, fakePodControl := setupManagerWithGCEnabled()
	rc := newReplicationController(2)
	manager.rcLister.Indexer.Add(rc)
	// put one pod in the podLister
	pod := newPod("pod", rc, v1.PodRunning, nil)
	pod.ResourceVersion = "1"
	var trueVar = true
	rcOwnerReference := metav1.OwnerReference{UID: rc.UID, APIVersion: "v1", Kind: "ReplicationController", Name: rc.Name, Controller: &trueVar}
	pod.OwnerReferences = []metav1.OwnerReference{rcOwnerReference}
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
	// verifies that rc is added to the queue
	rcKey := getKey(rc, t)
	queueRC, _ := manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}
	manager.queue.Done(queueRC)
	err := manager.syncReplicationController(rcKey)
	if err != nil {
		t.Fatal(err)
	}
	// expect 1 patch to be sent to remove the controllerRef for the pod.
	// expect 2 creates because the *(rc.Spec.Replicas)=2 and there exists no
	// matching pod.
	validateSyncReplication(t, fakePodControl, 2, 0, 1)
	fakePodControl.Clear()
}

func TestUpdateSelectorControllerRef(t *testing.T) {
	manager, fakePodControl := setupManagerWithGCEnabled()
	rc := newReplicationController(2)
	// put 2 pods in the podLister
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, rc, "pod")
	// update the RC so that its selector no longer matches the pods
	updatedRC := *rc
	updatedRC.Spec.Selector = map[string]string{"foo": "baz"}
	// put the updatedRC into the store. This is consistent with the behavior of
	// the Informer: Informer updates the store before call the handler
	// (updateRC() in this case).
	manager.rcLister.Indexer.Add(&updatedRC)
	manager.updateRC(rc, &updatedRC)
	// verifies that the rc is added to the queue
	rcKey := getKey(rc, t)
	queueRC, _ := manager.queue.Get()
	if queueRC != rcKey {
		t.Fatalf("Expected to find key %v in queue, found %v", rcKey, queueRC)
	}
	manager.queue.Done(queueRC)
	err := manager.syncReplicationController(rcKey)
	if err != nil {
		t.Fatal(err)
	}
	// expect 2 patches to be sent to remove the controllerRef for the pods.
	// expect 2 creates because the *(rc.Spec.Replicas)=2 and there exists no
	// matching pod.
	validateSyncReplication(t, fakePodControl, 2, 0, 2)
	fakePodControl.Clear()
}

// RC manager shouldn't adopt or create more pods if the rc is about to be
// deleted.
func TestDoNotAdoptOrCreateIfBeingDeleted(t *testing.T) {
	manager, fakePodControl := setupManagerWithGCEnabled()
	rc := newReplicationController(2)
	now := metav1.Now()
	rc.DeletionTimestamp = &now
	manager.rcLister.Indexer.Add(rc)
	pod1 := newPod("pod1", rc, v1.PodRunning, nil)
	manager.podLister.Indexer.Add(pod1)

	// no patch, no create
	err := manager.syncReplicationController(getKey(rc, t))
	if err != nil {
		t.Fatal(err)
	}
	validateSyncReplication(t, fakePodControl, 0, 0, 0)
}

func TestReadyReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "{}",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	rc := newReplicationController(2)
	rc.Status = v1.ReplicationControllerStatus{Replicas: 2, ReadyReplicas: 0, AvailableReplicas: 0, ObservedGeneration: 1}
	rc.Generation = 1
	manager.rcLister.Indexer.Add(rc)

	newPodList(manager.podLister.Indexer, 2, v1.PodPending, rc, "pod")
	newPodList(manager.podLister.Indexer, 2, v1.PodRunning, rc, "pod")

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.ReplicationController{})
	fakeHandler.ResponseBody = response

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	manager.syncReplicationController(getKey(rc, t))

	// ReadyReplicas should go from 0 to 2.
	rc.Status = v1.ReplicationControllerStatus{Replicas: 2, ReadyReplicas: 2, AvailableReplicas: 2, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Default.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name)+"/status", "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
}

func TestAvailableReplicas(t *testing.T) {
	// This is a happy server just to record the PUT request we expect for status.Replicas
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "{}",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	c := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	manager := NewReplicationManagerFromClient(c, controller.NoResyncPeriodFunc, BurstReplicas, 0)
	manager.podListerSynced = alwaysReady

	// Status.Replica should update to match number of pods in system, 1 new pod should be created.
	rc := newReplicationController(2)
	rc.Status = v1.ReplicationControllerStatus{Replicas: 2, ReadyReplicas: 0, ObservedGeneration: 1}
	rc.Generation = 1
	// minReadySeconds set to 15s
	rc.Spec.MinReadySeconds = 15
	manager.rcLister.Indexer.Add(rc)

	// First pod becomes ready 20s ago
	moment := metav1.Time{Time: time.Now().Add(-2e10)}
	pod := newPod("pod", rc, v1.PodRunning, &moment)
	manager.podLister.Indexer.Add(pod)

	// Second pod becomes ready now
	otherMoment := metav1.Now()
	otherPod := newPod("otherPod", rc, v1.PodRunning, &otherMoment)
	manager.podLister.Indexer.Add(otherPod)

	// This response body is just so we don't err out decoding the http response
	response := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.ReplicationController{})
	fakeHandler.ResponseBody = response

	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl

	// The controller should see only one available pod.
	manager.syncReplicationController(getKey(rc, t))

	rc.Status = v1.ReplicationControllerStatus{Replicas: 2, ReadyReplicas: 2, AvailableReplicas: 1, ObservedGeneration: 1}

	decRc := runtime.EncodeOrDie(testapi.Default.Codec(), rc)
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath(replicationControllerResourceName(), rc.Namespace, rc.Name)+"/status", "PUT", &decRc)
	validateSyncReplication(t, &fakePodControl, 0, 0, 0)
}

var (
	imagePullBackOff v1.ReplicationControllerConditionType = "ImagePullBackOff"

	condImagePullBackOff = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   imagePullBackOff,
			Status: v1.ConditionTrue,
			Reason: "NonExistentImage",
		}
	}

	condReplicaFailure = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   v1.ReplicationControllerReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "OtherFailure",
		}
	}

	condReplicaFailure2 = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   v1.ReplicationControllerReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "AnotherFailure",
		}
	}

	status = func() *v1.ReplicationControllerStatus {
		return &v1.ReplicationControllerStatus{
			Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()},
		}
	}
)

func TestGetCondition(t *testing.T) {
	exampleStatus := status()

	tests := []struct {
		name string

		status     v1.ReplicationControllerStatus
		condType   v1.ReplicationControllerConditionType
		condStatus v1.ConditionStatus
		condReason string

		expected bool
	}{
		{
			name: "condition exists",

			status:   *exampleStatus,
			condType: v1.ReplicationControllerReplicaFailure,

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

		status *v1.ReplicationControllerStatus
		cond   v1.ReplicationControllerCondition

		expectedStatus *v1.ReplicationControllerStatus
	}{
		{
			name: "set for the first time",

			status: &v1.ReplicationControllerStatus{},
			cond:   condReplicaFailure(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
		},
		{
			name: "simple set",

			status: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condImagePullBackOff()}},
			cond:   condReplicaFailure(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condImagePullBackOff(), condReplicaFailure()}},
		},
		{
			name: "overwrite",

			status: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
			cond:   condReplicaFailure2(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure2()}},
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

		status   *v1.ReplicationControllerStatus
		condType v1.ReplicationControllerConditionType

		expectedStatus *v1.ReplicationControllerStatus
	}{
		{
			name: "remove from empty status",

			status:   &v1.ReplicationControllerStatus{},
			condType: v1.ReplicationControllerReplicaFailure,

			expectedStatus: &v1.ReplicationControllerStatus{},
		},
		{
			name: "simple remove",

			status:   &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
			condType: v1.ReplicationControllerReplicaFailure,

			expectedStatus: &v1.ReplicationControllerStatus{},
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
