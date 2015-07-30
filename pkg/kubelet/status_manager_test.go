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

package kubelet

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var testPod *api.Pod = &api.Pod{
	ObjectMeta: api.ObjectMeta{
		UID:       "12345678",
		Name:      "foo",
		Namespace: "new",
	},
}

func newTestStatusManager() *statusManager {
	return newStatusManager(&testclient.Fake{})
}

func generateRandomMessage() string {
	return strconv.Itoa(rand.Int())
}

func getRandomPodStatus() api.PodStatus {
	return api.PodStatus{
		Message: generateRandomMessage(),
	}
}

func verifyActions(t *testing.T, kubeClient client.Interface, expectedActions []string) {
	actions := kubeClient.(*testclient.Fake).Actions()
	if len(actions) != len(expectedActions) {
		t.Errorf("unexpected actions, got: %s expected: %s", actions, expectedActions)
		return
	}
	for i := 0; i < len(actions); i++ {
		if actions[i].Action != expectedActions[i] {
			t.Errorf("unexpected actions, got: %s expected: %s", actions, expectedActions)
		}
	}
}

func verifyUpdates(t *testing.T, manager *statusManager, expectedUpdates int) {
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
	syncer := newTestStatusManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 1)

	status, _ := syncer.GetPodStatus(kubecontainer.GetPodFullName(testPod))
	if status.StartTime.IsZero() {
		t.Errorf("SetPodStatus did not set a proper start time value")
	}
}

func TestNewStatusPreservesPodStartTime(t *testing.T) {
	syncer := newTestStatusManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: api.PodStatus{},
	}
	now := util.Now()
	startTime := util.NewTime(now.Time.Add(-1 * time.Minute))
	pod.Status.StartTime = &startTime
	syncer.SetPodStatus(pod, getRandomPodStatus())

	status, _ := syncer.GetPodStatus(kubecontainer.GetPodFullName(pod))
	if !status.StartTime.Time.Equal(startTime.Time) {
		t.Errorf("Unexpected start time, expected %v, actual %v", startTime, status.StartTime)
	}
}

func TestChangedStatus(t *testing.T) {
	syncer := newTestStatusManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
}

func TestChangedStatusKeepsStartTime(t *testing.T) {
	syncer := newTestStatusManager()
	now := util.Now()
	firstStatus := getRandomPodStatus()
	firstStatus.StartTime = &now
	syncer.SetPodStatus(testPod, firstStatus)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
	finalStatus, _ := syncer.GetPodStatus(kubecontainer.GetPodFullName(testPod))
	if finalStatus.StartTime.IsZero() {
		t.Errorf("StartTime should not be zero")
	}
	if !finalStatus.StartTime.Time.Equal(now.Time) {
		t.Errorf("Expected %v, but got %v", now.Time, finalStatus.StartTime.Time)
	}
}

func TestUnchangedStatus(t *testing.T) {
	syncer := newTestStatusManager()
	podStatus := getRandomPodStatus()
	syncer.SetPodStatus(testPod, podStatus)
	syncer.SetPodStatus(testPod, podStatus)
	verifyUpdates(t, syncer, 1)
}

func TestSyncBatch(t *testing.T) {
	syncer := newTestStatusManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	err := syncer.syncBatch()
	if err != nil {
		t.Errorf("unexpected syncing error: %v", err)
	}
	verifyActions(t, syncer.kubeClient, []string{"get-pod", "update-status-pod"})
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
