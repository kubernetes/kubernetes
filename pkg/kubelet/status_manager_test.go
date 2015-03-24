/*
Copyright 2014 Google Inc. All rights reserved.

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
	"math/rand"
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

var testPod *api.Pod = &api.Pod{
	ObjectMeta: api.ObjectMeta{
		UID:       "12345678",
		Name:      "foo",
		Namespace: "new",
	},
}

func newTestStatusManager() *statusManager {
	return newStatusManager(&client.Fake{})
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
	actions := kubeClient.(*client.Fake).Actions
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

func TestNewStatus(t *testing.T) {
	syncer := newTestStatusManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SyncBatch()
	verifyActions(t, syncer.kubeClient, []string{"update-status-pod"})
}

func TestChangedStatus(t *testing.T) {
	syncer := newTestStatusManager()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SyncBatch()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SyncBatch()
	verifyActions(t, syncer.kubeClient, []string{"update-status-pod", "update-status-pod"})
}

func TestUnchangedStatus(t *testing.T) {
	syncer := newTestStatusManager()
	podStatus := getRandomPodStatus()
	syncer.SetPodStatus(testPod, podStatus)
	syncer.SyncBatch()
	syncer.SetPodStatus(testPod, podStatus)
	syncer.SyncBatch()
	verifyActions(t, syncer.kubeClient, []string{"update-status-pod"})
}
