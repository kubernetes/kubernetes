/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// NewFakeControllerExpectationsLookup creates a fake store for PodExpectations.
func NewFakeControllerExpectationsLookup(ttl time.Duration) (*ControllerExpectations, *util.FakeClock) {
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := &util.FakeClock{Time: fakeTime}
	ttlPolicy := &cache.TTLPolicy{Ttl: ttl, Clock: fakeClock}
	ttlStore := cache.NewFakeExpirationStore(
		ExpKeyFunc, nil, ttlPolicy, fakeClock)
	return &ControllerExpectations{ttlStore}, fakeClock
}

func newReplicationController(replicas int) *api.ReplicationController {
	rc := &api.ReplicationController{
		TypeMeta: api.TypeMeta{APIVersion: testapi.Default.Version()},
		ObjectMeta: api.ObjectMeta{
			UID:             util.NewUUID(),
			Name:            "foobar",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "18",
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
						"type": "production",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo/bar",
							TerminationMessagePath: api.TerminationMessagePathDefault,
							ImagePullPolicy:        api.PullIfNotPresent,
							SecurityContext:        securitycontext.ValidSecurityContextWithContainerDefaults(),
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
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
func newPodList(store cache.Store, count int, status api.PodPhase, rc *api.ReplicationController) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod%d", i),
				Labels:    rc.Spec.Selector,
				Namespace: rc.Namespace,
			},
			Status: api.PodStatus{Phase: status},
		}
		if store != nil {
			store.Add(&newPod)
		}
		pods = append(pods, newPod)
	}
	return &api.PodList{
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
		t.Errorf("Couldn't get key for object %+v: %v", rc, err)
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
	fakeClock.Time = fakeClock.Time.Add(ttl + 1)
	if !e.SatisfiedExpectations(rcKey) {
		t.Errorf("Expectations should have expired but didn't")
	}
}

func TestCreateReplica(t *testing.T) {
	ns := api.NamespaceDefault
	body := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Name: "empty_pod"}})
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Default.Version()})

	podControl := RealPodControl{
		KubeClient: client,
		Recorder:   &record.FakeRecorder{},
	}

	controllerSpec := newReplicationController(1)

	// Make sure createReplica sends a POST to the apiserver with a pod from the controllers pod template
	podControl.CreateReplica(ns, controllerSpec)

	expectedPod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels:       controllerSpec.Spec.Template.Labels,
			GenerateName: fmt.Sprintf("%s-", controllerSpec.Name),
		},
		Spec: controllerSpec.Spec.Template.Spec,
	}
	fakeHandler.ValidateRequest(t, testapi.Default.ResourcePath("pods", api.NamespaceDefault, ""), "POST", nil)
	actualPod, err := client.Codec.Decode([]byte(fakeHandler.RequestBody))
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !api.Semantic.DeepDerivative(&expectedPod, actualPod) {
		t.Logf("Body: %s", fakeHandler.RequestBody)
		t.Errorf("Unexpected mismatch.  Expected\n %#v,\n Got:\n %#v", &expectedPod, actualPod)
	}
}

func TestActivePodFiltering(t *testing.T) {
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, 5, api.PodRunning, rc)
	podList.Items[0].Status.Phase = api.PodSucceeded
	podList.Items[1].Status.Phase = api.PodFailed
	expectedNames := sets.NewString()
	for _, pod := range podList.Items[2:] {
		expectedNames.Insert(pod.Name)
	}

	got := FilterActivePods(podList.Items)
	gotNames := sets.NewString()
	for _, pod := range got {
		gotNames.Insert(pod.Name)
	}
	if expectedNames.Difference(gotNames).Len() != 0 || gotNames.Difference(expectedNames).Len() != 0 {
		t.Errorf("expected %v, got %v", expectedNames.List(), gotNames.List())
	}
}

func TestSortingActivePods(t *testing.T) {
	numPods := 5
	// This rc is not needed by the test, only the newPodList to give the pods labels/a namespace.
	rc := newReplicationController(0)
	podList := newPodList(nil, numPods, api.PodRunning, rc)

	pods := make([]*api.Pod, len(podList.Items))
	for i := range podList.Items {
		pods[i] = &podList.Items[i]
	}
	// pods[0] is not scheduled yet.
	pods[0].Spec.NodeName = ""
	pods[0].Status.Phase = api.PodPending
	// pods[1] is scheduled but pending.
	pods[1].Spec.NodeName = "bar"
	pods[1].Status.Phase = api.PodPending
	// pods[2] is unknown.
	pods[2].Spec.NodeName = "foo"
	pods[2].Status.Phase = api.PodUnknown
	// pods[3] is running but not ready.
	pods[3].Spec.NodeName = "foo"
	pods[3].Status.Phase = api.PodRunning
	// pods[4] is running and ready.
	pods[4].Spec.NodeName = "foo"
	pods[4].Status.Phase = api.PodRunning
	pods[4].Status.Conditions = []api.PodCondition{{Type: api.PodReady, Status: api.ConditionTrue}}

	getOrder := func(pods []*api.Pod) []string {
		names := make([]string, len(pods))
		for i := range pods {
			names[i] = pods[i].Name
		}
		return names
	}

	expected := getOrder(pods)

	for i := 0; i < 20; i++ {
		idx := rand.Perm(numPods)
		randomizedPods := make([]*api.Pod, numPods)
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
