/*
Copyright 2015 The Kubernetes Authors.

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

package controller

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http/httptest"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/util/clock"
	"k8s.io/kubernetes/pkg/util/sets"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/util/uuid"
)

// NewFakeControllerExpectationsLookup creates a fake store for PodExpectations.
func NewFakeControllerExpectationsLookup(ttl time.Duration) (*ControllerExpectations, *clock.FakeClock) {
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := clock.NewFakeClock(fakeTime)
	ttlPolicy := &cache.TTLPolicy{Ttl: ttl, Clock: fakeClock}
	ttlStore := cache.NewFakeExpirationStore(
		ExpKeyFunc, nil, ttlPolicy, fakeClock)
	return &ControllerExpectations{ttlStore}, fakeClock
}

func newReplicationController(replicas int) *v1.ReplicationController {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{APIVersion: registered.GroupOrDie(v1.GroupName).GroupVersion.String()},
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

// create count pods with the given phase for the given rc (same selectors and namespace), and add them to the store.
func newPodList(store cache.Store, count int, status v1.PodPhase, rc *v1.ReplicationController) *v1.PodList {
	pods := []v1.Pod{}
	for i := 0; i < count; i++ {
		newPod := v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    rc.Spec.Selector,
				Namespace: rc.Namespace,
			},
			Status: v1.PodStatus{Phase: status},
		}
		if store != nil {
			store.Add(&newPod)
		}
		pods = append(pods, newPod)
	}
	return &v1.PodList{
		Items: pods,
	}
}

func TestControllerExpectations(t *testing.T) {
	ttl := 30 * time.Second
	e, fakeClock := NewFakeControllerExpectationsLookup(ttl)
	// In practice we can't really have add and delete expectations since we only either create or
	// delete replicas in one rc pass, and the rc goes to sleep soon after until the expectations are
	// either fulfilled or timeout.
	adds, dels := 10, 30
	rc := newReplicationController(1)

	// RC fires off adds and deletes at apiserver, then sets expectations
	rcKey, err := KeyFunc(rc)
	if err != nil {
		t.Errorf("Couldn't get key for object %#v: %v", rc, err)
	}
	e.SetExpectations(rcKey, adds, dels)
	var wg sync.WaitGroup
	for i := 0; i < adds+1; i++ {
		wg.Add(1)
		go func() {
			// In prod this can happen either because of a failed create by the rc
			// or after having observed a create via informer
			e.CreationObserved(rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	// There are still delete expectations
	if e.SatisfiedExpectations(rcKey) {
		t.Errorf("Rc will sync before expectations are met")
	}
	for i := 0; i < dels+1; i++ {
		wg.Add(1)
		go func() {
			e.DeletionObserved(rcKey)
			wg.Done()
		}()
	}
	wg.Wait()

	// Expectations have been surpassed
	if podExp, exists, err := e.GetExpectations(rcKey); err == nil && exists {
		add, del := podExp.GetExpectations()
		if add != -1 || del != -1 {
			t.Errorf("Unexpected pod expectations %#v", podExp)
		}
	} else {
		t.Errorf("Could not get expectations for rc, exists %v and err %v", exists, err)
	}
	if !e.SatisfiedExpectations(rcKey) {
		t.Errorf("Expectations are met but the rc will not sync")
	}

	// Next round of rc sync, old expectations are cleared
	e.SetExpectations(rcKey, 1, 2)
	if podExp, exists, err := e.GetExpectations(rcKey); err == nil && exists {
		add, del := podExp.GetExpectations()
		if add != 1 || del != 2 {
			t.Errorf("Unexpected pod expectations %#v", podExp)
		}
	} else {
		t.Errorf("Could not get expectations for rc, exists %v and err %v", exists, err)
	}

	// Expectations have expired because of ttl
	fakeClock.Step(ttl + 1)
	if !e.SatisfiedExpectations(rcKey) {
		t.Errorf("Expectations should have expired but didn't")
	}
}

func TestUIDExpectations(t *testing.T) {
	uidExp := NewUIDTrackingControllerExpectations(NewControllerExpectations())
	rcList := []*v1.ReplicationController{
		newReplicationController(2),
		newReplicationController(1),
		newReplicationController(0),
		newReplicationController(5),
	}
	rcToPods := map[string][]string{}
	rcKeys := []string{}
	for i := range rcList {
		rc := rcList[i]
		rcName := fmt.Sprintf("rc-%v", i)
		rc.Name = rcName
		rc.Spec.Selector[rcName] = rcName
		podList := newPodList(nil, 5, v1.PodRunning, rc)
		rcKey, err := KeyFunc(rc)
		if err != nil {
			t.Fatalf("Couldn't get key for object %#v: %v", rc, err)
		}
		rcKeys = append(rcKeys, rcKey)
		rcPodNames := []string{}
		for i := range podList.Items {
			p := &podList.Items[i]
			p.Name = fmt.Sprintf("%v-%v", p.Name, rc.Name)
			rcPodNames = append(rcPodNames, PodKey(p))
		}
		rcToPods[rcKey] = rcPodNames
		uidExp.ExpectDeletions(rcKey, rcPodNames)
	}
	for i := range rcKeys {
		j := rand.Intn(i + 1)
		rcKeys[i], rcKeys[j] = rcKeys[j], rcKeys[i]
	}
	for _, rcKey := range rcKeys {
		if uidExp.SatisfiedExpectations(rcKey) {
			t.Errorf("Controller %v satisfied expectations before deletion", rcKey)
		}
		for _, p := range rcToPods[rcKey] {
			uidExp.DeletionObserved(rcKey, p)
		}
		if !uidExp.SatisfiedExpectations(rcKey) {
			t.Errorf("Controller %v didn't satisfy expectations after deletion", rcKey)
		}
		uidExp.DeleteExpectations(rcKey)
		if uidExp.GetUIDs(rcKey) != nil {
			t.Errorf("Failed to delete uid expectations for %v", rcKey)
		}
	}
}

func TestCreatePods(t *testing.T) {
	ns := v1.NamespaceDefault
	body := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "empty_pod"}})
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})

	podControl := RealPodControl{
		KubeClient: clientset,
		Recorder:   &record.FakeRecorder{},
	}

	controllerSpec := newReplicationController(1)

	// Make sure createReplica sends a POST to the apiserver with a pod from the controllers pod template
	if err := podControl.CreatePods(ns, controllerSpec.Spec.Template, controllerSpec); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedPod := v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Labels:       controllerSpec.Spec.Template.Labels,
			GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
		},
		Spec: controllerSpec.Spec.Template.Spec,
	}
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath("pods", v1.NamespaceDefault, ""), "POST", nil)
	var actualPod = &v1.Pod{}
	err := json.Unmarshal([]byte(fakeHandler.RequestBody), actualPod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepDerivative(&expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", &expectedPod, actualPod)
	}
}

func TestActivePodFiltering(t *testing.T) {
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, 5, v1.PodRunning, rc)
	podList.Items[0].Status.Phase = v1.PodSucceeded
	podList.Items[1].Status.Phase = v1.PodFailed
	expectedNames := sets.NewString()
	for _, pod := range podList.Items[2:] {
		expectedNames.Insert(pod.Name)
	}

	var podPointers []*v1.Pod
	for i := range podList.Items {
		podPointers = append(podPointers, &podList.Items[i])
	}
	got := FilterActivePods(podPointers)
	gotNames := sets.NewString()
	for _, pod := range got {
		gotNames.Insert(pod.Name)
	}
	if expectedNames.Difference(gotNames).Len() != 0 || gotNames.Difference(expectedNames).Len() != 0 {
		t.Errorf("expected %v, got %v", expectedNames.List(), gotNames.List())
	}
}

func TestSortingActivePods(t *testing.T) {
	numPods := 9
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, numPods, v1.PodRunning, rc)

	pods := make([]*v1.Pod, len(podList.Items))
	for i := range podList.Items {
		pods[i] = &podList.Items[i]
	}
	// pods[0] is not scheduled yet.
	pods[0].Spec.NodeName = ""
	pods[0].Status.Phase = v1.PodPending
	// pods[1] is scheduled but pending.
	pods[1].Spec.NodeName = "bar"
	pods[1].Status.Phase = v1.PodPending
	// pods[2] is unknown.
	pods[2].Spec.NodeName = "foo"
	pods[2].Status.Phase = v1.PodUnknown
	// pods[3] is running but not ready.
	pods[3].Spec.NodeName = "foo"
	pods[3].Status.Phase = v1.PodRunning
	// pods[4] is running and ready but without LastTransitionTime.
	now := metav1.Now()
	pods[4].Spec.NodeName = "foo"
	pods[4].Status.Phase = v1.PodRunning
	pods[4].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue}}
	pods[4].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[5] is running and ready and with LastTransitionTime.
	pods[5].Spec.NodeName = "foo"
	pods[5].Status.Phase = v1.PodRunning
	pods[5].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: now}}
	pods[5].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[6] is running ready for a longer time than pods[5].
	then := metav1.Time{Time: now.AddDate(0, -1, 0)}
	pods[6].Spec.NodeName = "foo"
	pods[6].Status.Phase = v1.PodRunning
	pods[6].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[6].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 3}, {RestartCount: 0}}
	// pods[7] has lower container restart count than pods[6].
	pods[7].Spec.NodeName = "foo"
	pods[7].Status.Phase = v1.PodRunning
	pods[7].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[7].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}}
	pods[7].CreationTimestamp = now
	// pods[8] is older than pods[7].
	pods[8].Spec.NodeName = "foo"
	pods[8].Status.Phase = v1.PodRunning
	pods[8].Status.Conditions = []v1.PodCondition{{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: then}}
	pods[8].Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: 2}, {RestartCount: 1}}
	pods[8].CreationTimestamp = then

	getOrder := func(pods []*v1.Pod) []string {
		names := make([]string, len(pods))
		for i := range pods {
			names[i] = pods[i].Name
		}
		return names
	}

	expected := getOrder(pods)

	for i := 0; i < 20; i++ {
		idx := rand.Perm(numPods)
		randomizedPods := make([]*v1.Pod, numPods)
		for j := 0; j < numPods; j++ {
			randomizedPods[j] = pods[idx[j]]
		}
		sort.Sort(ActivePods(randomizedPods))
		actual := getOrder(randomizedPods)

		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("expected %v, got %v", expected, actual)
		}
	}
}
