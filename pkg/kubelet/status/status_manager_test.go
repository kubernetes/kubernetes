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

package status

import (
	"fmt"
	"math/rand"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeconfigmap "k8s.io/kubernetes/pkg/kubelet/configmap"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	kubesecret "k8s.io/kubernetes/pkg/kubelet/secret"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Generate new instance of test pod with the same initial value.
func getTestPod() *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
	}
}

// After adding reconciliation, if status in pod manager is different from the cached status, a reconciliation
// will be triggered, which will mess up all the old unit test.
// To simplify the implementation of unit test, we add testSyncBatch() here, it will make sure the statuses in
// pod manager the same with cached ones before syncBatch() so as to avoid reconciling.
func (m *manager) testSyncBatch() {
	for uid, status := range m.podStatuses {
		pod, ok := m.podManager.GetPodByUID(uid)
		if ok {
			pod.Status = status.status
		}
		pod, ok = m.podManager.GetMirrorPodByPod(pod)
		if ok {
			pod.Status = status.status
		}
	}
	m.syncBatch()
}

func newTestManager(kubeClient clientset.Interface) *manager {
	podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient(), kubesecret.NewFakeManager(), kubeconfigmap.NewFakeManager())
	podManager.AddPod(getTestPod())
	return NewManager(kubeClient, podManager, &statustest.FakePodDeletionSafetyProvider{}).(*manager)
}

func generateRandomMessage() string {
	return strconv.Itoa(rand.Int())
}

func getRandomPodStatus() v1.PodStatus {
	return v1.PodStatus{
		Message: generateRandomMessage(),
	}
}

func verifyActions(t *testing.T, manager *manager, expectedActions []core.Action) {
	t.Helper()
	manager.consumeUpdates()
	actions := manager.kubeClient.(*fake.Clientset).Actions()
	defer manager.kubeClient.(*fake.Clientset).ClearActions()
	if len(actions) != len(expectedActions) {
		t.Fatalf("unexpected actions, got: %+v expected: %+v", actions, expectedActions)
		return
	}
	for i := 0; i < len(actions); i++ {
		e := expectedActions[i]
		a := actions[i]
		if !a.Matches(e.GetVerb(), e.GetResource().Resource) || a.GetSubresource() != e.GetSubresource() {
			t.Errorf("unexpected actions, got: %+v expected: %+v", actions, expectedActions)
		}
	}
}

func verifyUpdates(t *testing.T, manager *manager, expectedUpdates int) {
	// Consume all updates in the channel.
	numUpdates := manager.consumeUpdates()
	if numUpdates != expectedUpdates {
		t.Errorf("unexpected number of updates %d, expected %d", numUpdates, expectedUpdates)
	}
}

func (m *manager) consumeUpdates() int {
	updates := 0
	for {
		select {
		case syncRequest := <-m.podStatusChannel:
			m.syncPod(syncRequest.podUID, syncRequest.status)
			updates++
		default:
			return updates
		}
	}
}

func TestNewStatus(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 1)

	status := expectPodStatus(t, syncer, testPod)
	if status.StartTime.IsZero() {
		t.Errorf("SetPodStatus did not set a proper start time value")
	}
}

func TestNewStatusPreservesPodStartTime(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: v1.PodStatus{},
	}
	now := metav1.Now()
	startTime := metav1.NewTime(now.Time.Add(-1 * time.Minute))
	pod.Status.StartTime = &startTime
	syncer.SetPodStatus(pod, getRandomPodStatus())

	status := expectPodStatus(t, syncer, pod)
	if !status.StartTime.Time.Equal(startTime.Time) {
		t.Errorf("Unexpected start time, expected %v, actual %v", startTime, status.StartTime)
	}
}

func getReadyPodStatus() v1.PodStatus {
	return v1.PodStatus{
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
		},
	}
}

func TestNewStatusSetsReadyTransitionTime(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	podStatus := getReadyPodStatus()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: v1.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	status := expectPodStatus(t, syncer, pod)
	readyCondition := podutil.GetPodReadyCondition(status)
	if readyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
}

func TestChangedStatus(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
}

func TestChangedStatusKeepsStartTime(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	now := metav1.Now()
	firstStatus := getRandomPodStatus()
	firstStatus.StartTime = &now
	syncer.SetPodStatus(testPod, firstStatus)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyUpdates(t, syncer, 2)
	finalStatus := expectPodStatus(t, syncer, testPod)
	if finalStatus.StartTime.IsZero() {
		t.Errorf("StartTime should not be zero")
	}
	expected := now.Rfc3339Copy()
	if !finalStatus.StartTime.Equal(&expected) {
		t.Errorf("Expected %v, but got %v", expected, finalStatus.StartTime)
	}
}

func TestChangedStatusUpdatesLastTransitionTime(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	podStatus := getReadyPodStatus()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: v1.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	oldStatus := expectPodStatus(t, syncer, pod)
	anotherStatus := getReadyPodStatus()
	anotherStatus.Conditions[0].Status = v1.ConditionFalse
	syncer.SetPodStatus(pod, anotherStatus)
	verifyUpdates(t, syncer, 1)
	newStatus := expectPodStatus(t, syncer, pod)

	oldReadyCondition := podutil.GetPodReadyCondition(oldStatus)
	newReadyCondition := podutil.GetPodReadyCondition(newStatus)
	if newReadyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
	if newReadyCondition.LastTransitionTime.Before(&oldReadyCondition.LastTransitionTime) {
		t.Errorf("Unexpected: new transition time %s, is before old transition time %s", newReadyCondition.LastTransitionTime, oldReadyCondition.LastTransitionTime)
	}
}

func TestUnchangedStatus(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	podStatus := getRandomPodStatus()
	syncer.SetPodStatus(testPod, podStatus)
	syncer.SetPodStatus(testPod, podStatus)
	verifyUpdates(t, syncer, 1)
}

func TestUnchangedStatusPreservesLastTransitionTime(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	podStatus := getReadyPodStatus()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Status: v1.PodStatus{},
	}
	syncer.SetPodStatus(pod, podStatus)
	verifyUpdates(t, syncer, 1)
	oldStatus := expectPodStatus(t, syncer, pod)
	anotherStatus := getReadyPodStatus()
	syncer.SetPodStatus(pod, anotherStatus)
	// No update.
	verifyUpdates(t, syncer, 0)
	newStatus := expectPodStatus(t, syncer, pod)

	oldReadyCondition := podutil.GetPodReadyCondition(oldStatus)
	newReadyCondition := podutil.GetPodReadyCondition(newStatus)
	if newReadyCondition.LastTransitionTime.IsZero() {
		t.Errorf("Unexpected: last transition time not set")
	}
	if !oldReadyCondition.LastTransitionTime.Equal(&newReadyCondition.LastTransitionTime) {
		t.Errorf("Unexpected: new transition time %s, is not equal to old transition time %s", newReadyCondition.LastTransitionTime, oldReadyCondition.LastTransitionTime)
	}
}

func TestSyncPodIgnoresNotFound(t *testing.T) {
	client := fake.Clientset{}
	syncer := newTestManager(&client)
	client.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, errors.NewNotFound(api.Resource("pods"), "test-pod")
	})
	syncer.SetPodStatus(getTestPod(), getRandomPodStatus())
	verifyActions(t, syncer, []core.Action{getAction()})
}

func TestSyncPod(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	syncer.kubeClient = fake.NewSimpleClientset(testPod)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	verifyActions(t, syncer, []core.Action{getAction(), patchAction()})
}

func TestSyncPodChecksMismatchedUID(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	pod := getTestPod()
	pod.UID = "first"
	syncer.podManager.AddPod(pod)
	differentPod := getTestPod()
	differentPod.UID = "second"
	syncer.podManager.AddPod(differentPod)
	syncer.kubeClient = fake.NewSimpleClientset(pod)
	syncer.SetPodStatus(differentPod, getRandomPodStatus())
	verifyActions(t, syncer, []core.Action{getAction()})
}

func TestSyncPodNoDeadlock(t *testing.T) {
	client := &fake.Clientset{}
	m := newTestManager(client)
	pod := getTestPod()

	// Setup fake client.
	var ret *v1.Pod
	var err error
	client.AddReactor("*", "pods", func(action core.Action) (bool, runtime.Object, error) {
		switch action := action.(type) {
		case core.GetAction:
			assert.Equal(t, pod.Name, action.GetName(), "Unexpected GetAction: %+v", action)
		case core.UpdateAction:
			assert.Equal(t, pod.Name, action.GetObject().(*v1.Pod).Name, "Unexpected UpdateAction: %+v", action)
		default:
			assert.Fail(t, "Unexpected Action: %+v", action)
		}
		return true, ret, err
	})

	pod.Status.ContainerStatuses = []v1.ContainerStatus{{State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}}}

	t.Logf("Pod not found.")
	ret = nil
	err = errors.NewNotFound(api.Resource("pods"), pod.Name)
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction()})

	t.Logf("Pod was recreated.")
	ret = getTestPod()
	ret.UID = "other_pod"
	err = nil
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction()})

	t.Logf("Pod not deleted (success case).")
	ret = getTestPod()
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction(), patchAction()})

	t.Logf("Pod is terminated, but still running.")
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction(), patchAction()})

	t.Logf("Pod is terminated successfully.")
	pod.Status.ContainerStatuses[0].State.Running = nil
	pod.Status.ContainerStatuses[0].State.Terminated = &v1.ContainerStateTerminated{}
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction(), patchAction()})

	t.Logf("Error case.")
	ret = nil
	err = fmt.Errorf("intentional test error")
	m.SetPodStatus(pod, getRandomPodStatus())
	verifyActions(t, m, []core.Action{getAction()})
}

func TestStaleUpdates(t *testing.T) {
	pod := getTestPod()
	client := fake.NewSimpleClientset(pod)
	m := newTestManager(client)

	status := v1.PodStatus{Message: "initial status"}
	m.SetPodStatus(pod, status)
	status.Message = "first version bump"
	m.SetPodStatus(pod, status)
	status.Message = "second version bump"
	m.SetPodStatus(pod, status)

	t.Logf("sync batch before syncPods pushes latest status, so we should see three statuses in the channel, but only one update")
	m.syncBatch()
	verifyUpdates(t, m, 3)
	verifyActions(t, m, []core.Action{getAction(), patchAction()})
	t.Logf("Nothing left in the channel to sync")
	verifyActions(t, m, []core.Action{})

	t.Log("Unchanged status should not send an update")
	m.SetPodStatus(pod, status)
	verifyUpdates(t, m, 0)

	t.Log("... even if it's stale as long as nothing changes")
	mirrorPodUID := kubetypes.MirrorPodUID(pod.UID)
	m.apiStatusVersions[mirrorPodUID] = m.apiStatusVersions[mirrorPodUID] - 1

	m.SetPodStatus(pod, status)
	m.syncBatch()
	verifyActions(t, m, []core.Action{getAction()})

	t.Logf("Nothing stuck in the pipe.")
	verifyUpdates(t, m, 0)
}

// shuffle returns a new shuffled list of container statuses.
func shuffle(statuses []v1.ContainerStatus) []v1.ContainerStatus {
	numStatuses := len(statuses)
	randIndexes := rand.Perm(numStatuses)
	shuffled := make([]v1.ContainerStatus, numStatuses)
	for i := 0; i < numStatuses; i++ {
		shuffled[i] = statuses[randIndexes[i]]
	}
	return shuffled
}

func TestStatusEquality(t *testing.T) {
	pod := v1.Pod{
		Spec: v1.PodSpec{},
	}
	containerStatus := []v1.ContainerStatus{}
	for i := 0; i < 10; i++ {
		s := v1.ContainerStatus{
			Name: fmt.Sprintf("container%d", i),
		}
		containerStatus = append(containerStatus, s)
	}
	podStatus := v1.PodStatus{
		ContainerStatuses: containerStatus,
	}
	for i := 0; i < 10; i++ {
		oldPodStatus := v1.PodStatus{
			ContainerStatuses: shuffle(podStatus.ContainerStatuses),
		}
		normalizeStatus(&pod, &oldPodStatus)
		normalizeStatus(&pod, &podStatus)
		if !isPodStatusByKubeletEqual(&oldPodStatus, &podStatus) {
			t.Fatalf("Order of container statuses should not affect normalized equality.")
		}
	}

	oldPodStatus := podStatus
	podStatus.Conditions = append(podStatus.Conditions, v1.PodCondition{
		Type:   v1.PodConditionType("www.example.com/feature"),
		Status: v1.ConditionTrue,
	})

	oldPodStatus.Conditions = append(podStatus.Conditions, v1.PodCondition{
		Type:   v1.PodConditionType("www.example.com/feature"),
		Status: v1.ConditionFalse,
	})

	normalizeStatus(&pod, &oldPodStatus)
	normalizeStatus(&pod, &podStatus)
	if !isPodStatusByKubeletEqual(&oldPodStatus, &podStatus) {
		t.Fatalf("Differences in pod condition not owned by kubelet should not affect normalized equality.")
	}
}

func TestStatusNormalizationEnforcesMaxBytes(t *testing.T) {
	pod := v1.Pod{
		Spec: v1.PodSpec{},
	}
	containerStatus := []v1.ContainerStatus{}
	for i := 0; i < 48; i++ {
		s := v1.ContainerStatus{
			Name: fmt.Sprintf("container%d", i),
			LastTerminationState: v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{
					Message: strings.Repeat("abcdefgh", int(24+i%3)),
				},
			},
		}
		containerStatus = append(containerStatus, s)
	}
	podStatus := v1.PodStatus{
		InitContainerStatuses: containerStatus[:24],
		ContainerStatuses:     containerStatus[24:],
	}
	result := normalizeStatus(&pod, &podStatus)
	count := 0
	for _, s := range result.InitContainerStatuses {
		l := len(s.LastTerminationState.Terminated.Message)
		if l < 192 || l > 256 {
			t.Errorf("container message had length %d", l)
		}
		count += l
	}
	if count > kubecontainer.MaxPodTerminationMessageLogLength {
		t.Errorf("message length not truncated")
	}
}

func TestStaticPod(t *testing.T) {
	staticPod := getTestPod()
	staticPod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"}
	mirrorPod := getTestPod()
	mirrorPod.UID = "mirror-12345678"
	mirrorPod.Annotations = map[string]string{
		kubetypes.ConfigSourceAnnotationKey: "api",
		kubetypes.ConfigMirrorAnnotationKey: "mirror",
	}
	client := fake.NewSimpleClientset(mirrorPod)
	m := newTestManager(client)

	t.Logf("Create the static pod")
	m.podManager.AddPod(staticPod)
	assert.True(t, kubetypes.IsStaticPod(staticPod), "SetUp error: staticPod")

	status := getRandomPodStatus()
	now := metav1.Now()
	status.StartTime = &now
	m.SetPodStatus(staticPod, status)

	t.Logf("Should be able to get the static pod status from status manager")
	retrievedStatus := expectPodStatus(t, m, staticPod)
	normalizeStatus(staticPod, &status)
	assert.True(t, isPodStatusByKubeletEqual(&status, &retrievedStatus), "Expected: %+v, Got: %+v", status, retrievedStatus)

	t.Logf("Should not sync pod in syncBatch because there is no corresponding mirror pod for the static pod.")
	m.syncBatch()
	assert.Equal(t, len(m.kubeClient.(*fake.Clientset).Actions()), 0, "Expected no updates after syncBatch, got %+v", m.kubeClient.(*fake.Clientset).Actions())

	t.Logf("Create the mirror pod")
	m.podManager.AddPod(mirrorPod)
	assert.True(t, kubetypes.IsMirrorPod(mirrorPod), "SetUp error: mirrorPod")
	assert.Equal(t, m.podManager.TranslatePodUID(mirrorPod.UID), kubetypes.ResolvedPodUID(staticPod.UID))

	t.Logf("Should be able to get the mirror pod status from status manager")
	retrievedStatus, _ = m.GetPodStatus(mirrorPod.UID)
	assert.True(t, isPodStatusByKubeletEqual(&status, &retrievedStatus), "Expected: %+v, Got: %+v", status, retrievedStatus)

	t.Logf("Should sync pod because the corresponding mirror pod is created")
	verifyActions(t, m, []core.Action{getAction(), patchAction()})

	t.Logf("syncBatch should not sync any pods because nothing is changed.")
	m.testSyncBatch()
	verifyActions(t, m, []core.Action{})

	t.Logf("Change mirror pod identity.")
	m.podManager.DeletePod(mirrorPod)
	mirrorPod.UID = "new-mirror-pod"
	mirrorPod.Status = v1.PodStatus{}
	m.podManager.AddPod(mirrorPod)

	t.Logf("Should not update to mirror pod, because UID has changed.")
	m.syncBatch()
	verifyActions(t, m, []core.Action{getAction()})
}

func TestTerminatePod(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	t.Logf("update the pod's status to Failed.  TerminatePod should preserve this status update.")
	firstStatus := getRandomPodStatus()
	firstStatus.Phase = v1.PodFailed
	firstStatus.InitContainerStatuses = []v1.ContainerStatus{
		{Name: "init-test-1"},
		{Name: "init-test-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "InitTest", ExitCode: 0}}},
		{Name: "init-test-3", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "InitTest", ExitCode: 3}}},
	}
	firstStatus.ContainerStatuses = []v1.ContainerStatus{
		{Name: "test-1"},
		{Name: "test-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test", ExitCode: 2}}},
		{Name: "test-3", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test", ExitCode: 0}}},
	}
	syncer.SetPodStatus(testPod, firstStatus)

	t.Logf("set the testPod to a pod with Phase running, to simulate a stale pod")
	testPod.Status = getRandomPodStatus()
	testPod.Status.Phase = v1.PodRunning

	syncer.TerminatePod(testPod)

	t.Logf("we expect the container statuses to have changed to terminated")
	newStatus := expectPodStatus(t, syncer, testPod)
	for i := range newStatus.ContainerStatuses {
		assert.False(t, newStatus.ContainerStatuses[i].State.Terminated == nil, "expected containers to be terminated")
	}
	for i := range newStatus.InitContainerStatuses {
		assert.False(t, newStatus.InitContainerStatuses[i].State.Terminated == nil, "expected init containers to be terminated")
	}

	expectUnknownState := v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "ContainerStatusUnknown", Message: "The container could not be located when the pod was terminated", ExitCode: 137}}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[0].State, expectUnknownState) {
		t.Errorf("terminated container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.InitContainerStatuses[0].State, expectUnknownState))
	}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[1].State, firstStatus.InitContainerStatuses[1].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[2].State, firstStatus.InitContainerStatuses[2].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[0].State, expectUnknownState) {
		t.Errorf("terminated container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.ContainerStatuses[0].State, expectUnknownState))
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[1].State, firstStatus.ContainerStatuses[1].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[2].State, firstStatus.ContainerStatuses[2].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}

	t.Logf("we expect the previous status update to be preserved.")
	assert.Equal(t, newStatus.Phase, firstStatus.Phase)
	assert.Equal(t, newStatus.Message, firstStatus.Message)
}

func TestTerminatePodWaiting(t *testing.T) {
	syncer := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	t.Logf("update the pod's status to Failed.  TerminatePod should preserve this status update.")
	firstStatus := getRandomPodStatus()
	firstStatus.Phase = v1.PodFailed
	firstStatus.InitContainerStatuses = []v1.ContainerStatus{
		{Name: "init-test-1"},
		{Name: "init-test-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "InitTest", ExitCode: 0}}},
		{Name: "init-test-3", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: "InitTest"}}},
	}
	firstStatus.ContainerStatuses = []v1.ContainerStatus{
		{Name: "test-1"},
		{Name: "test-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test", ExitCode: 2}}},
		{Name: "test-3", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: "Test"}}},
	}
	syncer.SetPodStatus(testPod, firstStatus)

	t.Logf("set the testPod to a pod with Phase running, to simulate a stale pod")
	testPod.Status = getRandomPodStatus()
	testPod.Status.Phase = v1.PodRunning

	syncer.TerminatePod(testPod)

	t.Logf("we expect the container statuses to have changed to terminated")
	newStatus := expectPodStatus(t, syncer, testPod)
	for _, container := range newStatus.ContainerStatuses {
		assert.False(t, container.State.Terminated == nil, "expected containers to be terminated")
	}
	for _, container := range newStatus.InitContainerStatuses[:2] {
		assert.False(t, container.State.Terminated == nil, "expected init containers to be terminated")
	}
	for _, container := range newStatus.InitContainerStatuses[2:] {
		assert.False(t, container.State.Waiting == nil, "expected init containers to be waiting")
	}

	expectUnknownState := v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "ContainerStatusUnknown", Message: "The container could not be located when the pod was terminated", ExitCode: 137}}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[0].State, expectUnknownState) {
		t.Errorf("terminated container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.InitContainerStatuses[0].State, expectUnknownState))
	}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[1].State, firstStatus.InitContainerStatuses[1].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}
	if !reflect.DeepEqual(newStatus.InitContainerStatuses[2].State, firstStatus.InitContainerStatuses[2].State) {
		t.Errorf("waiting container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.InitContainerStatuses[2].State, firstStatus.InitContainerStatuses[2].State))
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[0].State, expectUnknownState) {
		t.Errorf("terminated container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.ContainerStatuses[0].State, expectUnknownState))
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[1].State, firstStatus.ContainerStatuses[1].State) {
		t.Errorf("existing terminated container state not preserved: %#v", newStatus.ContainerStatuses)
	}
	if !reflect.DeepEqual(newStatus.ContainerStatuses[2].State, expectUnknownState) {
		t.Errorf("waiting container state not defaulted: %s", diff.ObjectReflectDiff(newStatus.ContainerStatuses[2].State, expectUnknownState))
	}

	t.Logf("we expect the previous status update to be preserved.")
	assert.Equal(t, newStatus.Phase, firstStatus.Phase)
	assert.Equal(t, newStatus.Message, firstStatus.Message)
}

func TestTerminatePod_DefaultUnknownStatus(t *testing.T) {
	newPod := func(initContainers, containers int, fns ...func(*v1.Pod)) *v1.Pod {
		pod := getTestPod()
		for i := 0; i < initContainers; i++ {
			pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
				Name: fmt.Sprintf("init-%d", i),
			})
		}
		for i := 0; i < containers; i++ {
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name: fmt.Sprintf("%d", i),
			})
		}
		pod.Status.StartTime = &metav1.Time{Time: time.Unix(1, 0).UTC()}
		for _, fn := range fns {
			fn(pod)
		}
		return pod
	}
	expectTerminatedUnknown := func(t *testing.T, state v1.ContainerState) {
		t.Helper()
		if state.Terminated == nil || state.Running != nil || state.Waiting != nil {
			t.Fatalf("unexpected state: %#v", state)
		}
		if state.Terminated.ExitCode != 137 || state.Terminated.Reason != "ContainerStatusUnknown" || len(state.Terminated.Message) == 0 {
			t.Fatalf("unexpected terminated state: %#v", state.Terminated)
		}
	}
	expectTerminated := func(t *testing.T, state v1.ContainerState, exitCode int32) {
		t.Helper()
		if state.Terminated == nil || state.Running != nil || state.Waiting != nil {
			t.Fatalf("unexpected state: %#v", state)
		}
		if state.Terminated.ExitCode != exitCode {
			t.Fatalf("unexpected terminated state: %#v", state.Terminated)
		}
	}
	expectWaiting := func(t *testing.T, state v1.ContainerState) {
		t.Helper()
		if state.Terminated != nil || state.Running != nil || state.Waiting == nil {
			t.Fatalf("unexpected state: %#v", state)
		}
	}

	testCases := []struct {
		name     string
		pod      *v1.Pod
		updateFn func(*v1.Pod)
		expectFn func(t *testing.T, status v1.PodStatus)
	}{
		{pod: newPod(0, 1, func(pod *v1.Pod) { pod.Status.Phase = v1.PodFailed })},
		{pod: newPod(0, 1, func(pod *v1.Pod) { pod.Status.Phase = v1.PodRunning })},
		{pod: newPod(0, 1, func(pod *v1.Pod) {
			pod.Status.Phase = v1.PodRunning
			pod.Status.ContainerStatuses = []v1.ContainerStatus{
				{Name: "0", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test", ExitCode: 2}}},
			}
		})},
		{
			name: "last termination state set",
			pod: newPod(0, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{
						Name:                 "0",
						LastTerminationState: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{Reason: "Test", ExitCode: 2}},
						State:                v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}},
					},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				container := status.ContainerStatuses[0]
				if container.LastTerminationState.Terminated.ExitCode != 2 {
					t.Fatalf("unexpected last state: %#v", container.LastTerminationState)
				}
				expectTerminatedUnknown(t, container.State)
			},
		},
		{
			name: "no previous state",
			pod: newPod(0, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "uninitialized pod defaults the first init container",
			pod: newPod(1, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectWaiting(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "uninitialized pod defaults only the first init container",
			pod: newPod(2, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectWaiting(t, status.InitContainerStatuses[1].State)
				expectWaiting(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "uninitialized pod defaults gaps",
			pod: newPod(4, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 1}}},
					{Name: "init-3", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.InitContainerStatuses[1].State)
				expectTerminated(t, status.InitContainerStatuses[2].State, 1)
				expectWaiting(t, status.InitContainerStatuses[3].State)
				expectWaiting(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "failed last container is uninitialized",
			pod: newPod(3, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 1}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.InitContainerStatuses[1].State)
				expectTerminated(t, status.InitContainerStatuses[2].State, 1)
				expectWaiting(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "successful last container is initialized",
			pod: newPod(3, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-2", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.InitContainerStatuses[1].State)
				expectTerminated(t, status.InitContainerStatuses[2].State, 0)
				expectTerminatedUnknown(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "successful last previous container is initialized, and container state is overwritten",
			pod: newPod(3, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{Name: "init-1", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
					{
						Name:                 "init-2",
						LastTerminationState: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}},
						State:                v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}},
					},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.InitContainerStatuses[1].State)
				expectTerminatedUnknown(t, status.InitContainerStatuses[2].State)
				expectTerminatedUnknown(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "running container proves initialization",
			pod: newPod(1, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Running: &v1.ContainerStateRunning{}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.ContainerStatuses[0].State)
			},
		},
		{
			name: "evidence of terminated container proves initialization",
			pod: newPod(1, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", State: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminated(t, status.ContainerStatuses[0].State, 0)
			},
		},
		{
			name: "evidence of previously terminated container proves initialization",
			pod: newPod(1, 1, func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
				pod.Status.Phase = v1.PodRunning
				pod.Status.InitContainerStatuses = []v1.ContainerStatus{
					{Name: "init-0", State: v1.ContainerState{Waiting: &v1.ContainerStateWaiting{}}},
				}
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{Name: "0", LastTerminationState: v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0}}},
				}
			}),
			expectFn: func(t *testing.T, status v1.PodStatus) {
				expectTerminatedUnknown(t, status.InitContainerStatuses[0].State)
				expectTerminatedUnknown(t, status.ContainerStatuses[0].State)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient(), kubesecret.NewFakeManager(), kubeconfigmap.NewFakeManager())
			syncer := NewManager(&fake.Clientset{}, podManager, &statustest.FakePodDeletionSafetyProvider{}).(*manager)

			original := tc.pod.DeepCopy()
			syncer.SetPodStatus(original, original.Status)

			copied := tc.pod.DeepCopy()
			if tc.updateFn != nil {
				tc.updateFn(copied)
			}
			expected := copied.DeepCopy()

			syncer.TerminatePod(copied)
			status := expectPodStatus(t, syncer, tc.pod.DeepCopy())
			if tc.expectFn != nil {
				tc.expectFn(t, status)
				return
			}
			if !reflect.DeepEqual(expected.Status, status) {
				diff := cmp.Diff(expected.Status, status)
				if len(diff) == 0 {
					t.Fatalf("diff returned no results for failed DeepEqual: %#v != %#v", expected.Status, status)
				}
				t.Fatalf("unexpected status: %s", diff)
			}
		})
	}
}

func TestSetContainerReadiness(t *testing.T) {
	cID1 := kubecontainer.ContainerID{Type: "test", ID: "1"}
	cID2 := kubecontainer.ContainerID{Type: "test", ID: "2"}
	containerStatuses := []v1.ContainerStatus{
		{
			Name:        "c1",
			ContainerID: cID1.String(),
			Ready:       false,
		}, {
			Name:        "c2",
			ContainerID: cID2.String(),
			Ready:       false,
		},
	}
	status := v1.PodStatus{
		ContainerStatuses: containerStatuses,
		Conditions: []v1.PodCondition{{
			Type:   v1.PodReady,
			Status: v1.ConditionFalse,
		}},
	}
	pod := getTestPod()
	pod.Spec.Containers = []v1.Container{{Name: "c1"}, {Name: "c2"}}

	// Verify expected readiness of containers & pod.
	verifyReadiness := func(step string, status *v1.PodStatus, c1Ready, c2Ready, podReady bool) {
		for _, c := range status.ContainerStatuses {
			switch c.ContainerID {
			case cID1.String():
				if c.Ready != c1Ready {
					t.Errorf("[%s] Expected readiness of c1 to be %v but was %v", step, c1Ready, c.Ready)
				}
			case cID2.String():
				if c.Ready != c2Ready {
					t.Errorf("[%s] Expected readiness of c2 to be %v but was %v", step, c2Ready, c.Ready)
				}
			default:
				t.Fatalf("[%s] Unexpected container: %+v", step, c)
			}
		}
		if status.Conditions[0].Type != v1.PodReady {
			t.Fatalf("[%s] Unexpected condition: %+v", step, status.Conditions[0])
		} else if ready := (status.Conditions[0].Status == v1.ConditionTrue); ready != podReady {
			t.Errorf("[%s] Expected readiness of pod to be %v but was %v", step, podReady, ready)
		}
	}

	m := newTestManager(&fake.Clientset{})
	// Add test pod because the container spec has been changed.
	m.podManager.AddPod(pod)

	t.Log("Setting readiness before status should fail.")
	m.SetContainerReadiness(pod.UID, cID1, true)
	verifyUpdates(t, m, 0)
	if status, ok := m.GetPodStatus(pod.UID); ok {
		t.Errorf("Unexpected PodStatus: %+v", status)
	}

	t.Log("Setting initial status.")
	m.SetPodStatus(pod, status)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyReadiness("initial", &status, false, false, false)

	t.Log("Setting unchanged readiness should do nothing.")
	m.SetContainerReadiness(pod.UID, cID1, false)
	verifyUpdates(t, m, 0)
	status = expectPodStatus(t, m, pod)
	verifyReadiness("unchanged", &status, false, false, false)

	t.Log("Setting container readiness should generate update but not pod readiness.")
	m.SetContainerReadiness(pod.UID, cID1, true)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyReadiness("c1 ready", &status, true, false, false)

	t.Log("Setting both containers to ready should update pod readiness.")
	m.SetContainerReadiness(pod.UID, cID2, true)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyReadiness("all ready", &status, true, true, true)

	t.Log("Setting non-existent container readiness should fail.")
	m.SetContainerReadiness(pod.UID, kubecontainer.ContainerID{Type: "test", ID: "foo"}, true)
	verifyUpdates(t, m, 0)
	status = expectPodStatus(t, m, pod)
	verifyReadiness("ignore non-existent", &status, true, true, true)
}

func TestSetContainerStartup(t *testing.T) {
	cID1 := kubecontainer.ContainerID{Type: "test", ID: "1"}
	cID2 := kubecontainer.ContainerID{Type: "test", ID: "2"}
	containerStatuses := []v1.ContainerStatus{
		{
			Name:        "c1",
			ContainerID: cID1.String(),
			Ready:       false,
		}, {
			Name:        "c2",
			ContainerID: cID2.String(),
			Ready:       false,
		},
	}
	status := v1.PodStatus{
		ContainerStatuses: containerStatuses,
		Conditions: []v1.PodCondition{{
			Type:   v1.PodReady,
			Status: v1.ConditionFalse,
		}},
	}
	pod := getTestPod()
	pod.Spec.Containers = []v1.Container{{Name: "c1"}, {Name: "c2"}}

	// Verify expected startup of containers & pod.
	verifyStartup := func(step string, status *v1.PodStatus, c1Started, c2Started, podStarted bool) {
		for _, c := range status.ContainerStatuses {
			switch c.ContainerID {
			case cID1.String():
				if (c.Started != nil && *c.Started) != c1Started {
					t.Errorf("[%s] Expected startup of c1 to be %v but was %v", step, c1Started, c.Started)
				}
			case cID2.String():
				if (c.Started != nil && *c.Started) != c2Started {
					t.Errorf("[%s] Expected startup of c2 to be %v but was %v", step, c2Started, c.Started)
				}
			default:
				t.Fatalf("[%s] Unexpected container: %+v", step, c)
			}
		}
	}

	m := newTestManager(&fake.Clientset{})
	// Add test pod because the container spec has been changed.
	m.podManager.AddPod(pod)

	t.Log("Setting startup before status should fail.")
	m.SetContainerStartup(pod.UID, cID1, true)
	verifyUpdates(t, m, 0)
	if status, ok := m.GetPodStatus(pod.UID); ok {
		t.Errorf("Unexpected PodStatus: %+v", status)
	}

	t.Log("Setting initial status.")
	m.SetPodStatus(pod, status)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyStartup("initial", &status, false, false, false)

	t.Log("Setting unchanged startup should do nothing.")
	m.SetContainerStartup(pod.UID, cID1, false)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyStartup("unchanged", &status, false, false, false)

	t.Log("Setting container startup should generate update but not pod startup.")
	m.SetContainerStartup(pod.UID, cID1, true)
	verifyUpdates(t, m, 1) // Started = nil to false
	status = expectPodStatus(t, m, pod)
	verifyStartup("c1 ready", &status, true, false, false)

	t.Log("Setting both containers to ready should update pod startup.")
	m.SetContainerStartup(pod.UID, cID2, true)
	verifyUpdates(t, m, 1)
	status = expectPodStatus(t, m, pod)
	verifyStartup("all ready", &status, true, true, true)

	t.Log("Setting non-existent container startup should fail.")
	m.SetContainerStartup(pod.UID, kubecontainer.ContainerID{Type: "test", ID: "foo"}, true)
	verifyUpdates(t, m, 0)
	status = expectPodStatus(t, m, pod)
	verifyStartup("ignore non-existent", &status, true, true, true)
}

func TestSyncBatchCleanupVersions(t *testing.T) {
	m := newTestManager(&fake.Clientset{})
	testPod := getTestPod()
	mirrorPod := getTestPod()
	mirrorPod.UID = "mirror-uid"
	mirrorPod.Name = "mirror_pod"
	mirrorPod.Annotations = map[string]string{
		kubetypes.ConfigSourceAnnotationKey: "api",
		kubetypes.ConfigMirrorAnnotationKey: "mirror",
	}

	t.Logf("Orphaned pods should be removed.")
	m.apiStatusVersions[kubetypes.MirrorPodUID(testPod.UID)] = 100
	m.apiStatusVersions[kubetypes.MirrorPodUID(mirrorPod.UID)] = 200
	m.syncBatch()
	if _, ok := m.apiStatusVersions[kubetypes.MirrorPodUID(testPod.UID)]; ok {
		t.Errorf("Should have cleared status for testPod")
	}
	if _, ok := m.apiStatusVersions[kubetypes.MirrorPodUID(mirrorPod.UID)]; ok {
		t.Errorf("Should have cleared status for mirrorPod")
	}

	t.Logf("Non-orphaned pods should not be removed.")
	m.SetPodStatus(testPod, getRandomPodStatus())
	m.podManager.AddPod(mirrorPod)
	staticPod := mirrorPod
	staticPod.UID = "static-uid"
	staticPod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"}
	m.podManager.AddPod(staticPod)
	m.apiStatusVersions[kubetypes.MirrorPodUID(testPod.UID)] = 100
	m.apiStatusVersions[kubetypes.MirrorPodUID(mirrorPod.UID)] = 200
	m.testSyncBatch()
	if _, ok := m.apiStatusVersions[kubetypes.MirrorPodUID(testPod.UID)]; !ok {
		t.Errorf("Should not have cleared status for testPod")
	}
	if _, ok := m.apiStatusVersions[kubetypes.MirrorPodUID(mirrorPod.UID)]; !ok {
		t.Errorf("Should not have cleared status for mirrorPod")
	}
}

func TestReconcilePodStatus(t *testing.T) {
	testPod := getTestPod()
	client := fake.NewSimpleClientset(testPod)
	syncer := newTestManager(client)
	syncer.SetPodStatus(testPod, getRandomPodStatus())
	t.Logf("Call syncBatch directly to test reconcile")
	syncer.syncBatch() // The apiStatusVersions should be set now
	client.ClearActions()

	podStatus, ok := syncer.GetPodStatus(testPod.UID)
	if !ok {
		t.Fatalf("Should find pod status for pod: %#v", testPod)
	}
	testPod.Status = podStatus

	t.Logf("If the pod status is the same, a reconciliation is not needed and syncBatch should do nothing")
	syncer.podManager.UpdatePod(testPod)
	if syncer.needsReconcile(testPod.UID, podStatus) {
		t.Fatalf("Pod status is the same, a reconciliation is not needed")
	}
	syncer.SetPodStatus(testPod, podStatus)
	syncer.syncBatch()
	verifyActions(t, syncer, []core.Action{})

	// If the pod status is the same, only the timestamp is in Rfc3339 format (lower precision without nanosecond),
	// a reconciliation is not needed, syncBatch should do nothing.
	// The StartTime should have been set in SetPodStatus().
	// This test is done because the related issue #15262/PR #15263 to move apiserver to RFC339NANO is closed.
	t.Logf("Syncbatch should do nothing, as a reconciliation is not required")
	normalizedStartTime := testPod.Status.StartTime.Rfc3339Copy()
	testPod.Status.StartTime = &normalizedStartTime
	syncer.podManager.UpdatePod(testPod)
	if syncer.needsReconcile(testPod.UID, podStatus) {
		t.Fatalf("Pod status only differs for timestamp format, a reconciliation is not needed")
	}
	syncer.SetPodStatus(testPod, podStatus)
	syncer.syncBatch()
	verifyActions(t, syncer, []core.Action{})

	t.Logf("If the pod status is different, a reconciliation is needed, syncBatch should trigger an update")
	changedPodStatus := getRandomPodStatus()
	syncer.podManager.UpdatePod(testPod)
	if !syncer.needsReconcile(testPod.UID, changedPodStatus) {
		t.Fatalf("Pod status is different, a reconciliation is needed")
	}
	syncer.SetPodStatus(testPod, changedPodStatus)
	syncer.syncBatch()
	verifyActions(t, syncer, []core.Action{getAction(), patchAction()})
}

func expectPodStatus(t *testing.T, m *manager, pod *v1.Pod) v1.PodStatus {
	status, ok := m.GetPodStatus(pod.UID)
	if !ok {
		t.Fatalf("Expected PodStatus for %q not found", pod.UID)
	}
	return status
}

func TestDeletePods(t *testing.T) {
	pod := getTestPod()
	t.Logf("Set the deletion timestamp.")
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	client := fake.NewSimpleClientset(pod)
	m := newTestManager(client)
	m.podDeletionSafety.(*statustest.FakePodDeletionSafetyProvider).Reclaimed = true
	m.podManager.AddPod(pod)
	status := getRandomPodStatus()
	now := metav1.Now()
	status.StartTime = &now
	m.SetPodStatus(pod, status)
	t.Logf("Expect to see a delete action.")
	verifyActions(t, m, []core.Action{getAction(), patchAction(), deleteAction()})
}

func TestDeletePodWhileReclaiming(t *testing.T) {
	pod := getTestPod()
	t.Logf("Set the deletion timestamp.")
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	client := fake.NewSimpleClientset(pod)
	m := newTestManager(client)
	m.podDeletionSafety.(*statustest.FakePodDeletionSafetyProvider).Reclaimed = false
	m.podManager.AddPod(pod)
	status := getRandomPodStatus()
	now := metav1.Now()
	status.StartTime = &now
	m.SetPodStatus(pod, status)
	t.Logf("Expect to see a delete action.")
	verifyActions(t, m, []core.Action{getAction(), patchAction()})
}

func TestDoNotDeleteMirrorPods(t *testing.T) {
	staticPod := getTestPod()
	staticPod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: "file"}
	mirrorPod := getTestPod()
	mirrorPod.UID = "mirror-12345678"
	mirrorPod.Annotations = map[string]string{
		kubetypes.ConfigSourceAnnotationKey: "api",
		kubetypes.ConfigMirrorAnnotationKey: "mirror",
	}
	t.Logf("Set the deletion timestamp.")
	mirrorPod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	client := fake.NewSimpleClientset(mirrorPod)
	m := newTestManager(client)
	m.podManager.AddPod(staticPod)
	m.podManager.AddPod(mirrorPod)
	t.Logf("Verify setup.")
	assert.True(t, kubetypes.IsStaticPod(staticPod), "SetUp error: staticPod")
	assert.True(t, kubetypes.IsMirrorPod(mirrorPod), "SetUp error: mirrorPod")
	assert.Equal(t, m.podManager.TranslatePodUID(mirrorPod.UID), kubetypes.ResolvedPodUID(staticPod.UID))

	status := getRandomPodStatus()
	now := metav1.Now()
	status.StartTime = &now
	m.SetPodStatus(staticPod, status)

	t.Logf("Expect not to see a delete action.")
	verifyActions(t, m, []core.Action{getAction(), patchAction()})
}

func TestUpdateLastTransitionTime(t *testing.T) {
	old := metav1.Now()
	for desc, test := range map[string]struct {
		condition    *v1.PodCondition
		oldCondition *v1.PodCondition
		expectUpdate bool
	}{
		"should do nothing if no corresponding condition": {
			expectUpdate: false,
		},
		"should update last transition time if no old condition": {
			condition: &v1.PodCondition{
				Type:   "test-type",
				Status: v1.ConditionTrue,
			},
			oldCondition: nil,
			expectUpdate: true,
		},
		"should update last transition time if condition is changed": {
			condition: &v1.PodCondition{
				Type:   "test-type",
				Status: v1.ConditionTrue,
			},
			oldCondition: &v1.PodCondition{
				Type:               "test-type",
				Status:             v1.ConditionFalse,
				LastTransitionTime: old,
			},
			expectUpdate: true,
		},
		"should keep last transition time if condition is not changed": {
			condition: &v1.PodCondition{
				Type:   "test-type",
				Status: v1.ConditionFalse,
			},
			oldCondition: &v1.PodCondition{
				Type:               "test-type",
				Status:             v1.ConditionFalse,
				LastTransitionTime: old,
			},
			expectUpdate: false,
		},
	} {
		t.Logf("TestCase %q", desc)
		status := &v1.PodStatus{}
		oldStatus := &v1.PodStatus{}
		if test.condition != nil {
			status.Conditions = []v1.PodCondition{*test.condition}
		}
		if test.oldCondition != nil {
			oldStatus.Conditions = []v1.PodCondition{*test.oldCondition}
		}
		updateLastTransitionTime(status, oldStatus, "test-type")
		if test.expectUpdate {
			assert.True(t, status.Conditions[0].LastTransitionTime.After(old.Time))
		} else if test.condition != nil {
			assert.Equal(t, old, status.Conditions[0].LastTransitionTime)
		}
	}
}

func getAction() core.GetAction {
	return core.GetActionImpl{ActionImpl: core.ActionImpl{Verb: "get", Resource: schema.GroupVersionResource{Resource: "pods"}}}
}

func patchAction() core.PatchAction {
	return core.PatchActionImpl{ActionImpl: core.ActionImpl{Verb: "patch", Resource: schema.GroupVersionResource{Resource: "pods"}, Subresource: "status"}}
}

func deleteAction() core.DeleteAction {
	return core.DeleteActionImpl{ActionImpl: core.ActionImpl{Verb: "delete", Resource: schema.GroupVersionResource{Resource: "pods"}}}
}

func TestMergePodStatus(t *testing.T) {
	useCases := []struct {
		desc                 string
		hasRunningContainers bool
		oldPodStatus         func(input v1.PodStatus) v1.PodStatus
		newPodStatus         func(input v1.PodStatus) v1.PodStatus
		expectPodStatus      v1.PodStatus
	}{
		{
			"no change",
			false,
			func(input v1.PodStatus) v1.PodStatus { return input },
			func(input v1.PodStatus) v1.PodStatus { return input },
			getPodStatus(),
		},
		{
			"readiness changes",
			false,
			func(input v1.PodStatus) v1.PodStatus { return input },
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions[0].Status = v1.ConditionFalse
				return input
			},
			v1.PodStatus{
				Phase: v1.PodRunning,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionFalse,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
				},
				Message: "Message",
			},
		},
		{
			"additional pod condition",
			false,
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions = append(input.Conditions, v1.PodCondition{
					Type:   v1.PodConditionType("example.com/feature"),
					Status: v1.ConditionTrue,
				})
				return input
			},
			func(input v1.PodStatus) v1.PodStatus { return input },
			v1.PodStatus{
				Phase: v1.PodRunning,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodConditionType("example.com/feature"),
						Status: v1.ConditionTrue,
					},
				},
				Message: "Message",
			},
		},
		{
			"additional pod condition and readiness changes",
			false,
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions = append(input.Conditions, v1.PodCondition{
					Type:   v1.PodConditionType("example.com/feature"),
					Status: v1.ConditionTrue,
				})
				return input
			},
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions[0].Status = v1.ConditionFalse
				return input
			},
			v1.PodStatus{
				Phase: v1.PodRunning,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionFalse,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodConditionType("example.com/feature"),
						Status: v1.ConditionTrue,
					},
				},
				Message: "Message",
			},
		},
		{
			"additional pod condition changes",
			false,
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions = append(input.Conditions, v1.PodCondition{
					Type:   v1.PodConditionType("example.com/feature"),
					Status: v1.ConditionTrue,
				})
				return input
			},
			func(input v1.PodStatus) v1.PodStatus {
				input.Conditions = append(input.Conditions, v1.PodCondition{
					Type:   v1.PodConditionType("example.com/feature"),
					Status: v1.ConditionFalse,
				})
				return input
			},
			v1.PodStatus{
				Phase: v1.PodRunning,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodConditionType("example.com/feature"),
						Status: v1.ConditionTrue,
					},
				},
				Message: "Message",
			},
		},
		{
			"phase is transitioning to failed and no containers running",
			false,
			func(input v1.PodStatus) v1.PodStatus {
				input.Phase = v1.PodRunning
				input.Reason = "Unknown"
				input.Message = "Message"
				return input
			},
			func(input v1.PodStatus) v1.PodStatus {
				input.Phase = v1.PodFailed
				input.Reason = "Evicted"
				input.Message = "Was Evicted"
				return input
			},
			v1.PodStatus{
				Phase: v1.PodFailed,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionFalse,
					},
					{
						Type:   v1.ContainersReady,
						Status: v1.ConditionFalse,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
				},
				Reason:  "Evicted",
				Message: "Was Evicted",
			},
		},
		{
			"phase is transitioning to failed and containers running",
			true,
			func(input v1.PodStatus) v1.PodStatus {
				input.Phase = v1.PodRunning
				input.Reason = "Unknown"
				input.Message = "Message"
				return input
			},
			func(input v1.PodStatus) v1.PodStatus {
				input.Phase = v1.PodFailed
				input.Reason = "Evicted"
				input.Message = "Was Evicted"
				return input
			},
			v1.PodStatus{
				Phase: v1.PodRunning,
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodReady,
						Status: v1.ConditionTrue,
					},
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionTrue,
					},
				},
				Reason:  "Unknown",
				Message: "Message",
			},
		},
	}

	for _, tc := range useCases {
		t.Run(tc.desc, func(t *testing.T) {
			output := mergePodStatus(tc.oldPodStatus(getPodStatus()), tc.newPodStatus(getPodStatus()), tc.hasRunningContainers)
			if !conditionsEqual(output.Conditions, tc.expectPodStatus.Conditions) || !statusEqual(output, tc.expectPodStatus) {
				t.Fatalf("unexpected output: %s", cmp.Diff(tc.expectPodStatus, output))
			}
		})
	}

}

func statusEqual(left, right v1.PodStatus) bool {
	left.Conditions = nil
	right.Conditions = nil
	return reflect.DeepEqual(left, right)
}

func conditionsEqual(left, right []v1.PodCondition) bool {
	if len(left) != len(right) {
		return false
	}

	for _, l := range left {
		found := false
		for _, r := range right {
			if l.Type == r.Type {
				found = true
				if l.Status != r.Status {
					return false
				}
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func getPodStatus() v1.PodStatus {
	return v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
			{
				Type:   v1.PodScheduled,
				Status: v1.ConditionTrue,
			},
		},
		Message: "Message",
	}
}
