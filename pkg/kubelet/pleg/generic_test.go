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

package pleg

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util/clock"
)

const (
	testContainerRuntimeType = "fooRuntime"
)

type TestGenericPLEG struct {
	pleg    *GenericPLEG
	runtime *containertest.FakeRuntime
	clock   *clock.FakeClock
}

func newTestGenericPLEG() *TestGenericPLEG {
	fakeRuntime := &containertest.FakeRuntime{}
	clock := clock.NewFakeClock(time.Time{})
	// The channel capacity should be large enough to hold all events in a
	// single test.
	pleg := &GenericPLEG{
		relistPeriod: time.Hour,
		runtime:      fakeRuntime,
		eventChannel: make(chan *PodLifecycleEvent, 100),
		podRecords:   make(podRecords),
		clock:        clock,
	}
	return &TestGenericPLEG{pleg: pleg, runtime: fakeRuntime, clock: clock}
}

func getEventsFromChannel(ch <-chan *PodLifecycleEvent) []*PodLifecycleEvent {
	events := []*PodLifecycleEvent{}
	for len(ch) > 0 {
		e := <-ch
		events = append(events, e)
	}
	return events
}

func createTestContainer(ID string, state kubecontainer.ContainerState) *kubecontainer.Container {
	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: testContainerRuntimeType, ID: ID},
		State: state,
	}
}

type sortableEvents []*PodLifecycleEvent

func (a sortableEvents) Len() int      { return len(a) }
func (a sortableEvents) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a sortableEvents) Less(i, j int) bool {
	if a[i].ID != a[j].ID {
		return a[i].ID < a[j].ID
	}
	return a[i].Data.(string) < a[j].Data.(string)
}

func verifyEvents(t *testing.T, expected, actual []*PodLifecycleEvent) {
	sort.Sort(sortableEvents(expected))
	sort.Sort(sortableEvents(actual))
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Actual events differ from the expected; diff:\n %v", diff.ObjectDiff(expected, actual))
	}
}

func TestRelisting(t *testing.T) {
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime
	ch := pleg.Watch()
	// The first relist should send a PodSync event to each pod.
	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
				createTestContainer("c2", kubecontainer.ContainerStateRunning),
				createTestContainer("c3", kubecontainer.ContainerStateUnknown),
			},
		}},
		{Pod: &kubecontainer.Pod{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
			},
		}},
	}
	pleg.relist()
	// Report every running/exited container if we see them for the first time.
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerStarted, Data: "c2"},
		{ID: "4567", Type: ContainerDied, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c1"},
	}
	actual := getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)

	// The second relist should not send out any event because no container has
	// changed.
	pleg.relist()
	verifyEvents(t, expected, actual)

	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c2", kubecontainer.ContainerStateExited),
				createTestContainer("c3", kubecontainer.ContainerStateRunning),
			},
		}},
		{Pod: &kubecontainer.Pod{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				createTestContainer("c4", kubecontainer.ContainerStateRunning),
			},
		}},
	}
	pleg.relist()
	// Only report containers that transitioned to running or exited status.
	expected = []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerRemoved, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerStarted, Data: "c3"},
		{ID: "4567", Type: ContainerRemoved, Data: "c1"},
		{ID: "4567", Type: ContainerStarted, Data: "c4"},
	}

	actual = getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}

func TestDetectingContainerDeaths(t *testing.T) {
	// Vary the number of relists after the container started and before the
	// container died to account for the changes in pleg's internal states.
	testReportMissingContainers(t, 1)
	testReportMissingPods(t, 1)

	testReportMissingContainers(t, 3)
	testReportMissingPods(t, 3)
}

func testReportMissingContainers(t *testing.T, numRelists int) {
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime
	ch := pleg.Watch()
	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateRunning),
				createTestContainer("c2", kubecontainer.ContainerStateRunning),
				createTestContainer("c3", kubecontainer.ContainerStateExited),
			},
		}},
	}
	// Relist and drain the events from the channel.
	for i := 0; i < numRelists; i++ {
		pleg.relist()
		getEventsFromChannel(ch)
	}

	// Container c2 was stopped and removed between relists. We should report
	// the event. The exited container c3 was garbage collected (i.e., removed)
	// between relists. We should ignore that event.
	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateRunning),
			},
		}},
	}
	pleg.relist()
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerRemoved, Data: "c2"},
		{ID: "1234", Type: ContainerRemoved, Data: "c3"},
	}
	actual := getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}

func testReportMissingPods(t *testing.T, numRelists int) {
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime
	ch := pleg.Watch()
	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c2", kubecontainer.ContainerStateRunning),
			},
		}},
	}
	// Relist and drain the events from the channel.
	for i := 0; i < numRelists; i++ {
		pleg.relist()
		getEventsFromChannel(ch)
	}

	// Container c2 was stopped and removed between relists. We should report
	// the event.
	runtime.AllPodList = []*containertest.FakePod{}
	pleg.relist()
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerRemoved, Data: "c2"},
	}
	actual := getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}

func newTestGenericPLEGWithRuntimeMock() (*GenericPLEG, *containertest.Mock) {
	runtimeMock := &containertest.Mock{}
	pleg := &GenericPLEG{
		relistPeriod: time.Hour,
		runtime:      runtimeMock,
		eventChannel: make(chan *PodLifecycleEvent, 100),
		podRecords:   make(podRecords),
		cache:        kubecontainer.NewCache(),
		clock:        clock.RealClock{},
	}
	return pleg, runtimeMock
}

func createTestPodsStatusesAndEvents(num int) ([]*kubecontainer.Pod, []*kubecontainer.PodStatus, []*PodLifecycleEvent) {
	var pods []*kubecontainer.Pod
	var statuses []*kubecontainer.PodStatus
	var events []*PodLifecycleEvent
	for i := 0; i < num; i++ {
		id := types.UID(fmt.Sprintf("test-pod-%d", i))
		cState := kubecontainer.ContainerStateRunning
		container := createTestContainer(fmt.Sprintf("c%d", i), cState)
		pod := &kubecontainer.Pod{
			ID:         id,
			Containers: []*kubecontainer.Container{container},
		}
		status := &kubecontainer.PodStatus{
			ID:                id,
			ContainerStatuses: []*kubecontainer.ContainerStatus{{ID: container.ID, State: cState}},
		}
		event := &PodLifecycleEvent{ID: pod.ID, Type: ContainerStarted, Data: container.ID.ID}
		pods = append(pods, pod)
		statuses = append(statuses, status)
		events = append(events, event)

	}
	return pods, statuses, events
}

func TestRelistWithCache(t *testing.T) {
	pleg, runtimeMock := newTestGenericPLEGWithRuntimeMock()
	ch := pleg.Watch()

	pods, statuses, events := createTestPodsStatusesAndEvents(2)
	runtimeMock.On("GetPods", true).Return(pods, nil)
	runtimeMock.On("GetPodStatus", pods[0].ID, "", "").Return(statuses[0], nil).Once()
	// Inject an error when querying runtime for the pod status for pods[1].
	statusErr := fmt.Errorf("unable to get status")
	runtimeMock.On("GetPodStatus", pods[1].ID, "", "").Return(&kubecontainer.PodStatus{}, statusErr).Once()

	pleg.relist()
	actualEvents := getEventsFromChannel(ch)
	cases := []struct {
		pod    *kubecontainer.Pod
		status *kubecontainer.PodStatus
		error  error
	}{
		{pod: pods[0], status: statuses[0], error: nil},
		{pod: pods[1], status: &kubecontainer.PodStatus{}, error: statusErr},
	}
	for i, c := range cases {
		testStr := fmt.Sprintf("test[%d]", i)
		actualStatus, actualErr := pleg.cache.Get(c.pod.ID)
		assert.Equal(t, c.status, actualStatus, testStr)
		assert.Equal(t, c.error, actualErr, testStr)
	}
	// pleg should not generate any event for pods[1] because of the error.
	assert.Exactly(t, []*PodLifecycleEvent{events[0]}, actualEvents)

	// Return normal status for pods[1].
	runtimeMock.On("GetPodStatus", pods[1].ID, "", "").Return(statuses[1], nil).Once()
	pleg.relist()
	actualEvents = getEventsFromChannel(ch)
	cases = []struct {
		pod    *kubecontainer.Pod
		status *kubecontainer.PodStatus
		error  error
	}{
		{pod: pods[0], status: statuses[0], error: nil},
		{pod: pods[1], status: statuses[1], error: nil},
	}
	for i, c := range cases {
		testStr := fmt.Sprintf("test[%d]", i)
		actualStatus, actualErr := pleg.cache.Get(c.pod.ID)
		assert.Equal(t, c.status, actualStatus, testStr)
		assert.Equal(t, c.error, actualErr, testStr)
	}
	// Now that we are able to query status for pods[1], pleg should generate an event.
	assert.Exactly(t, []*PodLifecycleEvent{events[1]}, actualEvents)
}

func TestRemoveCacheEntry(t *testing.T) {
	pleg, runtimeMock := newTestGenericPLEGWithRuntimeMock()
	pods, statuses, _ := createTestPodsStatusesAndEvents(1)
	runtimeMock.On("GetPods", true).Return(pods, nil).Once()
	runtimeMock.On("GetPodStatus", pods[0].ID, "", "").Return(statuses[0], nil).Once()
	// Does a relist to populate the cache.
	pleg.relist()
	// Delete the pod from runtime. Verify that the cache entry has been
	// removed after relisting.
	runtimeMock.On("GetPods", true).Return([]*kubecontainer.Pod{}, nil).Once()
	pleg.relist()
	actualStatus, actualErr := pleg.cache.Get(pods[0].ID)
	assert.Equal(t, &kubecontainer.PodStatus{ID: pods[0].ID}, actualStatus)
	assert.Equal(t, nil, actualErr)
}

func TestHealthy(t *testing.T) {
	testPleg := newTestGenericPLEG()
	pleg, _, clock := testPleg.pleg, testPleg.runtime, testPleg.clock
	ok, _ := pleg.Healthy()
	assert.True(t, ok, "pleg should be healthy")

	// Advance the clock without any relisting.
	clock.Step(time.Minute * 10)
	ok, _ = pleg.Healthy()
	assert.False(t, ok, "pleg should be unhealthy")

	// Relist and than advance the time by 1 minute. pleg should be healthy
	// because this is within the allowed limit.
	pleg.relist()
	clock.Step(time.Minute * 1)
	ok, _ = pleg.Healthy()
	assert.True(t, ok, "pleg should be healthy")
}

func TestRelistWithReinspection(t *testing.T) {
	pleg, runtimeMock := newTestGenericPLEGWithRuntimeMock()
	ch := pleg.Watch()

	infraContainer := createTestContainer("infra", kubecontainer.ContainerStateRunning)

	podID := types.UID("test-pod")
	pods := []*kubecontainer.Pod{{
		ID:         podID,
		Containers: []*kubecontainer.Container{infraContainer},
	}}
	runtimeMock.On("GetPods", true).Return(pods, nil).Once()

	goodStatus := &kubecontainer.PodStatus{
		ID:                podID,
		ContainerStatuses: []*kubecontainer.ContainerStatus{{ID: infraContainer.ID, State: infraContainer.State}},
	}
	runtimeMock.On("GetPodStatus", podID, "", "").Return(goodStatus, nil).Once()

	goodEvent := &PodLifecycleEvent{ID: podID, Type: ContainerStarted, Data: infraContainer.ID.ID}

	// listing 1 - everything ok, infra container set up for pod
	pleg.relist()
	actualEvents := getEventsFromChannel(ch)
	actualStatus, actualErr := pleg.cache.Get(podID)
	assert.Equal(t, goodStatus, actualStatus)
	assert.Equal(t, nil, actualErr)
	assert.Exactly(t, []*PodLifecycleEvent{goodEvent}, actualEvents)

	// listing 2 - pretend runtime was in the middle of creating the non-infra container for the pod
	// and return an error during inspection
	transientContainer := createTestContainer("transient", kubecontainer.ContainerStateUnknown)
	podsWithTransientContainer := []*kubecontainer.Pod{{
		ID:         podID,
		Containers: []*kubecontainer.Container{infraContainer, transientContainer},
	}}
	runtimeMock.On("GetPods", true).Return(podsWithTransientContainer, nil).Once()

	badStatus := &kubecontainer.PodStatus{
		ID:                podID,
		ContainerStatuses: []*kubecontainer.ContainerStatus{},
	}
	runtimeMock.On("GetPodStatus", podID, "", "").Return(badStatus, errors.New("inspection error")).Once()

	pleg.relist()
	actualEvents = getEventsFromChannel(ch)
	actualStatus, actualErr = pleg.cache.Get(podID)
	assert.Equal(t, badStatus, actualStatus)
	assert.Equal(t, errors.New("inspection error"), actualErr)
	assert.Exactly(t, []*PodLifecycleEvent{}, actualEvents)

	// listing 3 - pretend the transient container has now disappeared, leaving just the infra
	// container. Make sure the pod is reinspected for its status and the cache is updated.
	runtimeMock.On("GetPods", true).Return(pods, nil).Once()
	runtimeMock.On("GetPodStatus", podID, "", "").Return(goodStatus, nil).Once()

	pleg.relist()
	actualEvents = getEventsFromChannel(ch)
	actualStatus, actualErr = pleg.cache.Get(podID)
	assert.Equal(t, goodStatus, actualStatus)
	assert.Equal(t, nil, actualErr)
	// no events are expected because relist #1 set the old pod record which has the infra container
	// running. relist #2 had the inspection error and therefore didn't modify either old or new.
	// relist #3 forced the reinspection of the pod to retrieve its status, but because the list of
	// containers was the same as relist #1, nothing "changed", so there are no new events.
	assert.Exactly(t, []*PodLifecycleEvent{}, actualEvents)
}

// Test detecting sandbox state changes.
func TestRelistingWithSandboxes(t *testing.T) {
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime
	ch := pleg.Watch()
	// The first relist should send a PodSync event to each pod.
	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
				createTestContainer("c2", kubecontainer.ContainerStateRunning),
				createTestContainer("c3", kubecontainer.ContainerStateUnknown),
			},
		}},
		{Pod: &kubecontainer.Pod{
			ID: "4567",
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
			},
		}},
	}
	pleg.relist()
	// Report every running/exited container if we see them for the first time.
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerStarted, Data: "c2"},
		{ID: "4567", Type: ContainerDied, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c1"},
	}
	actual := getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)

	// The second relist should not send out any event because no container has
	// changed.
	pleg.relist()
	verifyEvents(t, expected, actual)

	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("c2", kubecontainer.ContainerStateExited),
				createTestContainer("c3", kubecontainer.ContainerStateRunning),
			},
		}},
		{Pod: &kubecontainer.Pod{
			ID: "4567",
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("c4", kubecontainer.ContainerStateRunning),
			},
		}},
	}
	pleg.relist()
	// Only report containers that transitioned to running or exited status.
	expected = []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerRemoved, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerStarted, Data: "c3"},
		{ID: "4567", Type: ContainerRemoved, Data: "c1"},
		{ID: "4567", Type: ContainerStarted, Data: "c4"},
	}

	actual = getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}
