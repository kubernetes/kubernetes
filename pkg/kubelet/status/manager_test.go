/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package status

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
)

var testPod *api.Pod = &api.Pod{
	ObjectMeta: api.ObjectMeta{
		UID:       "12345678",
		Name:      "foo",
		Namespace: "new",
	},
}

func newTestManager() *manager {
	return NewManager(&testclient.Fake{}).(*manager)
}

func generateRandomMessage() string {
	return strconv.Itoa(rand.Int())
}

func getRandomPodStatus() api.PodStatus {
	return api.PodStatus{
		Message: generateRandomMessage(),
	}
}

func verifyActions(t *testing.T, kubeClient client.Interface, expectedActions []testclient.Action) {
	actions := kubeClient.(*testclient.Fake).Actions()
	if len(actions) != len(expectedActions) {
		t.Errorf("unexpected actions, got: %s expected: %s", actions, expectedActions)
		return
	}
	for i := 0; i < len(actions); i++ {
		e := expectedActions[i]
		a := actions[i]
		if !a.Matches(e.GetVerb(), e.GetResource()) || a.GetSubresource() != e.GetSubresource() {
			t.Errorf("unexpected actions, got: %s expected: %s", actions, expectedActions)
		}
	}
}

func verifyUpdates(t *testing.T, manager *manager, expectedUpdates int) {
	// Consume all updates in the channel.
	numUpdates := 0
	for {
		hasUpdate := true
		select {
		case <-manager.podStatusChannel:
			numUpdates++
		default:
			hasUpdate = false
		}

		if !hasUpdate {
			break
		}
	}

	if numUpdates != expectedUpdates {
		t.Errorf("unexpected number of updates %d, expected %d", numUpdates, expectedUpdates)
	}
}

func TestNewStatus(t *testing.T) {
	syncer := newTestManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 1)

	status, _ := syncer.GetPodStatus(testPod.UID)
	if status.StartTime.IsZero() {
		t.Errorf("SetPodStatus did not set a proper start time value")
	}
}

func TestNewStatusPreservesPodStartTime(t *testing.T) {
	syncer := newTestManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: api.PodStatus{},
	}
	now := unversioned.Now()
	startTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	pod.Status.StartTime = &startTime
	syncer.SetPodStatus(pod, getRandomPodStatus())

	status, _ := syncer.GetPodStatus(pod.UID)
	if !status.StartTime.Time.Equal(startTime.Time) {
		t.Errorf("Unexpected start time, expected %v, actual %v", startTime, status.StartTime)
	}
}

func getReadyPodStatus() api.PodStatus {
	return api.PodStatus{
		Conditions: []api.PodCondition{
			{
				Type:   api.PodReady,
				Status: api.ConditionTrue,
			},
		},
	}
}

func TestNewStatusSetsReadyTransitionTime(t *testing.T) {
	syncer := newTestManager()
	podStatus := getReadyPodStatus()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: api.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	status, _ := syncer.GetPodStatus(pod.UID)
	readyCondition := api.GetPodReadyCondition(status)
	if readyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
}

func TestChangedStatus(t *testing.T) {
	syncer := newTestManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
}

func TestChangedStatusKeepsStartTime(t *testing.T) {
	syncer := newTestManager()
	now := unversioned.Now()
	firstStatus := getRandomPodStatus()
	firstStatus.StartTime = &now
	syncer.SetPodStatus(testPod, firstStatus)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
	finalStatus, _ := syncer.GetPodStatus(testPod.UID)
	if finalStatus.StartTime.IsZero() {
		t.Errorf("StartTime should not be zero")
	}
	if !finalStatus.StartTime.Time.Equal(now.Time) {
		t.Errorf("Expected %v, but got %v", now.Time, finalStatus.StartTime.Time)
	}
}

func TestChangedStatusUpdatesLastTransitionTime(t *testing.T) {
	syncer := newTestManager()
	podStatus := getReadyPodStatus()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: api.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	oldStatus, _ := syncer.GetPodStatus(pod.UID)
	anotherStatus := getReadyPodStatus()
	anotherStatus.Conditions[0].Status = api.ConditionFalse
	syncer.SetPodStatus(pod, anotherStatus)
	verifyUpdates(t, syncer, 1)
	newStatus, _ := syncer.GetPodStatus(pod.UID)

	oldReadyCondition := api.GetPodReadyCondition(oldStatus)
	newReadyCondition := api.GetPodReadyCondition(newStatus)
	if newReadyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
	if !oldReadyCondition.LastTransitionTime.Before(newReadyCondition.LastTransitionTime) {
		t.Errorf("Unexpected: new transition time %s, is not after old transition time %s", newReadyCondition.LastTransitionTime, oldReadyCondition.LastTransitionTime)
	}
}

func TestUnchangedStatus(t *testing.T) {
	syncer := newTestManager()
	podStatus := getRandomPodStatus()
	syncer.SetPodStatus(testPod, podStatus)
	syncer.SetPodStatus(testPod, podStatus)
	verifyUpdates(t, syncer, 1)
}

func TestUnchangedStatusPreservesLastTransitionTime(t *testing.T) {
	syncer := newTestManager()
	podStatus := getReadyPodStatus()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: api.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	oldStatus, _ := syncer.GetPodStatus(pod.UID)
	anotherStatus := getReadyPodStatus()
	syncer.SetPodStatus(pod, anotherStatus)
	// No update.
	verifyUpdates(t, syncer, 0)
	newStatus, _ := syncer.GetPodStatus(pod.UID)

	oldReadyCondition := api.GetPodReadyCondition(oldStatus)
	newReadyCondition := api.GetPodReadyCondition(newStatus)
	if newReadyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
	if !oldReadyCondition.LastTransitionTime.Equal(newReadyCondition.LastTransitionTime) {
		t.Errorf("Unexpected: new transition time %s, is not equal to old transition time %s", newReadyCondition.LastTransitionTime, oldReadyCondition.LastTransitionTime)
	}
}

func TestSyncBatchIgnoresNotFound(t *testing.T) {
	syncer := newTestManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	err := syncer.syncBatch()
	if err != nil {
		t.Errorf("unexpected syncing error: %v", err)
	}
	verifyActions(t, syncer.kubeClient, []testclient.Action{
		testclient.GetActionImpl{ActionImpl: testclient.ActionImpl{Verb: "get", Resource: "pods"}},
	})
}

func TestSyncBatch(t *testing.T) {
	syncer := newTestManager()
	syncer.kubeClient = testclient.NewSimpleFake(testPod)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	err := syncer.syncBatch()
	if err != nil {
		t.Errorf("unexpected syncing error: %v", err)
	}
	verifyActions(t, syncer.kubeClient, []testclient.Action{
		testclient.GetActionImpl{ActionImpl: testclient.ActionImpl{Verb: "get", Resource: "pods"}},
		testclient.UpdateActionImpl{ActionImpl: testclient.ActionImpl{Verb: "update", Resource: "pods", Subresource: "status"}},
	},
	)
}

func TestSyncBatchChecksMismatchedUID(t *testing.T) {
	syncer := newTestManager()
	testPod.UID = "first"
	differentPod := *testPod
	differentPod.UID = "second"
	syncer.kubeClient = testclient.NewSimpleFake(testPod)
	syncer.SetPodStatus(&differentPod, getRandomPodStatus())
	err := syncer.syncBatch()
	if err != nil {
		t.Errorf("unexpected syncing error: %v", err)
	}
	verifyActions(t, syncer.kubeClient, []testclient.Action{
		testclient.GetActionImpl{ActionImpl: testclient.ActionImpl{Verb: "get", Resource: "pods"}},
	})
}

// shuffle returns a new shuffled list of container statuses.
func shuffle(statuses []api.ContainerStatus) []api.ContainerStatus {
	numStatuses := len(statuses)
	randIndexes := rand.Perm(numStatuses)
	shuffled := make([]api.ContainerStatus, numStatuses)
	for i := 0; i < numStatuses; i++ {
		shuffled[i] = statuses[randIndexes[i]]
	}
	return shuffled
}

func TestStatusEquality(t *testing.T) {
	containerStatus := []api.ContainerStatus{}
	for i := 0; i < 10; i++ {
		s := api.ContainerStatus{
			Name: fmt.Sprintf("container%d", i),
		}
		containerStatus = append(containerStatus, s)
	}
	podStatus := api.PodStatus{
		ContainerStatuses: containerStatus,
	}
	for i := 0; i < 10; i++ {
		oldPodStatus := api.PodStatus{
			ContainerStatuses: shuffle(podStatus.ContainerStatuses),
		}
		if !isStatusEqual(&oldPodStatus, &podStatus) {
			t.Fatalf("Order of container statuses should not affect equality.")
		}
	}
}
