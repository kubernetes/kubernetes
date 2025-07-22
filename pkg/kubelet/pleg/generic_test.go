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
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

const (
	testContainerRuntimeType = "fooRuntime"
	// largeChannelCap is a large enough capacity to hold all events in a single test.
	largeChannelCap = 100
)

type TestGenericPLEG struct {
	pleg    *GenericPLEG
	runtime *containertest.FakeRuntime
	clock   *testingclock.FakeClock
}

func newTestGenericPLEG() *TestGenericPLEG {
	return newTestGenericPLEGWithChannelSize(largeChannelCap)
}

func newTestGenericPLEGWithChannelSize(eventChannelCap int) *TestGenericPLEG {
	fakeRuntime := &containertest.FakeRuntime{}
	fakeCache := containertest.NewFakeCache(fakeRuntime)
	clock := testingclock.NewFakeClock(time.Time{})
	// The channel capacity should be large enough to hold all events in a
	// single test.
	pleg := NewGenericPLEG(
		klog.Logger{},
		fakeRuntime,
		make(chan *PodLifecycleEvent, eventChannelCap),
		&RelistDuration{RelistPeriod: time.Hour, RelistThreshold: 3 * time.Minute},
		fakeCache,
		clock,
	).(*GenericPLEG)
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

func createTestContainer(id string, state kubecontainer.State) *kubecontainer.Container {
	return &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: testContainerRuntimeType, ID: id},
		Name:  id,
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
		t.Errorf("Actual events differ from the expected; diff:\n %v", cmp.Diff(expected, actual))
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
	pleg.Relist()
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
	pleg.Relist()
	actual = getEventsFromChannel(ch)
	assert.Empty(t, actual, "no container has changed, event length should be 0")

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
	pleg.Relist()
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

// TestEventChannelFull test when channel is full, the events will be discard.
func TestEventChannelFull(t *testing.T) {
	testPleg := newTestGenericPLEGWithChannelSize(4)
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
	pleg.Relist()
	// Report every running/exited container if we see them for the first time.
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerStarted, Data: "c2"},
		{ID: "4567", Type: ContainerDied, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c1"},
	}
	actual := getEventsFromChannel(ch)
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
	pleg.Relist()
	allEvents := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerRemoved, Data: "c1"},
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerStarted, Data: "c3"},
		{ID: "4567", Type: ContainerRemoved, Data: "c1"},
		{ID: "4567", Type: ContainerStarted, Data: "c4"},
	}
	// event channel is full, discard events
	actual = getEventsFromChannel(ch)
	assert.Len(t, actual, 4, "channel length should be 4")
	assert.Subsetf(t, allEvents, actual, "actual events should in all events")
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
		pleg.Relist()
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
	pleg.Relist()
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
		pleg.Relist()
		getEventsFromChannel(ch)
	}

	// Container c2 was stopped and removed between relists. We should report
	// the event.
	runtime.AllPodList = []*containertest.FakePod{}
	pleg.Relist()
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerRemoved, Data: "c2"},
	}
	actual := getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}

func newTestGenericPLEGWithRuntimeMock(runtimeMock kubecontainer.Runtime) *GenericPLEG {
	pleg := NewGenericPLEG(
		klog.Logger{},
		runtimeMock,
		make(chan *PodLifecycleEvent, 1000),
		&RelistDuration{RelistPeriod: time.Hour, RelistThreshold: 2 * time.Hour},
		kubecontainer.NewCache(),
		clock.RealClock{},
	).(*GenericPLEG)
	return pleg
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
			ContainerStatuses: []*kubecontainer.Status{{ID: container.ID, State: cState}},
		}
		event := &PodLifecycleEvent{ID: pod.ID, Type: ContainerStarted, Data: container.ID.ID}
		pods = append(pods, pod)
		statuses = append(statuses, status)
		events = append(events, event)

	}
	return pods, statuses, events
}

func TestRelistWithCache(t *testing.T) {
	ctx := context.Background()
	runtimeMock := containertest.NewMockRuntime(t)

	pleg := newTestGenericPLEGWithRuntimeMock(runtimeMock)
	ch := pleg.Watch()

	pods, statuses, events := createTestPodsStatusesAndEvents(2)
	runtimeMock.EXPECT().GetPods(ctx, true).Return(pods, nil).Maybe()
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[0].ID, "", "").Return(statuses[0], nil).Times(1)
	// Inject an error when querying runtime for the pod status for pods[1].
	statusErr := fmt.Errorf("unable to get status")
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[1].ID, "", "").Return(&kubecontainer.PodStatus{}, statusErr).Times(1)

	pleg.Relist()
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
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[1].ID, "", "").Return(statuses[1], nil).Times(1)
	pleg.Relist()
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
	ctx := context.Background()
	runtimeMock := containertest.NewMockRuntime(t)
	pleg := newTestGenericPLEGWithRuntimeMock(runtimeMock)

	pods, statuses, _ := createTestPodsStatusesAndEvents(1)
	runtimeMock.EXPECT().GetPods(ctx, true).Return(pods, nil).Times(1)
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[0].ID, "", "").Return(statuses[0], nil).Times(1)
	// Does a relist to populate the cache.
	pleg.Relist()
	// Delete the pod from runtime. Verify that the cache entry has been
	// removed after relisting.
	runtimeMock.EXPECT().GetPods(ctx, true).Return([]*kubecontainer.Pod{}, nil).Times(1)
	pleg.Relist()
	actualStatus, actualErr := pleg.cache.Get(pods[0].ID)
	assert.Equal(t, &kubecontainer.PodStatus{ID: pods[0].ID}, actualStatus)
	assert.NoError(t, actualErr)
}

func TestHealthy(t *testing.T) {
	testPleg := newTestGenericPLEG()

	// pleg should initially be unhealthy
	pleg, _, clock := testPleg.pleg, testPleg.runtime, testPleg.clock
	ok, _ := pleg.Healthy()
	assert.False(t, ok, "pleg should be unhealthy")

	// Advance the clock without any relisting.
	clock.Step(time.Minute * 10)
	ok, _ = pleg.Healthy()
	assert.False(t, ok, "pleg should be unhealthy")

	// Relist and than advance the time by 1 minute. pleg should be healthy
	// because this is within the allowed limit.
	pleg.Relist()
	clock.Step(time.Minute * 1)
	ok, _ = pleg.Healthy()
	assert.True(t, ok, "pleg should be healthy")

	// Advance by relistThreshold without any relisting. pleg should be unhealthy
	// because it has been longer than relistThreshold since a relist occurred.
	clock.Step(pleg.relistDuration.RelistThreshold)
	ok, _ = pleg.Healthy()
	assert.False(t, ok, "pleg should be unhealthy")
}

func TestRelistWithReinspection(t *testing.T) {
	ctx := context.Background()
	runtimeMock := containertest.NewMockRuntime(t)

	pleg := newTestGenericPLEGWithRuntimeMock(runtimeMock)
	ch := pleg.Watch()

	infraContainer := createTestContainer("infra", kubecontainer.ContainerStateRunning)

	podID := types.UID("test-pod")
	pods := []*kubecontainer.Pod{{
		ID:         podID,
		Containers: []*kubecontainer.Container{infraContainer},
	}}
	runtimeMock.EXPECT().GetPods(ctx, true).Return(pods, nil).Times(1)

	goodStatus := &kubecontainer.PodStatus{
		ID:                podID,
		ContainerStatuses: []*kubecontainer.Status{{ID: infraContainer.ID, State: infraContainer.State}},
	}
	runtimeMock.EXPECT().GetPodStatus(ctx, podID, "", "").Return(goodStatus, nil).Times(1)

	goodEvent := &PodLifecycleEvent{ID: podID, Type: ContainerStarted, Data: infraContainer.ID.ID}

	// listing 1 - everything ok, infra container set up for pod
	pleg.Relist()
	actualEvents := getEventsFromChannel(ch)
	actualStatus, actualErr := pleg.cache.Get(podID)
	assert.Equal(t, goodStatus, actualStatus)
	assert.NoError(t, actualErr)
	assert.Exactly(t, []*PodLifecycleEvent{goodEvent}, actualEvents)

	// listing 2 - pretend runtime was in the middle of creating the non-infra container for the pod
	// and return an error during inspection
	transientContainer := createTestContainer("transient", kubecontainer.ContainerStateUnknown)
	podsWithTransientContainer := []*kubecontainer.Pod{{
		ID:         podID,
		Containers: []*kubecontainer.Container{infraContainer, transientContainer},
	}}
	runtimeMock.EXPECT().GetPods(ctx, true).Return(podsWithTransientContainer, nil).Times(1)

	badStatus := &kubecontainer.PodStatus{
		ID:                podID,
		ContainerStatuses: []*kubecontainer.Status{},
	}
	runtimeMock.EXPECT().GetPodStatus(ctx, podID, "", "").Return(badStatus, errors.New("inspection error")).Times(1)

	pleg.Relist()
	actualEvents = getEventsFromChannel(ch)
	actualStatus, actualErr = pleg.cache.Get(podID)
	assert.Equal(t, badStatus, actualStatus)
	assert.Equal(t, errors.New("inspection error"), actualErr)
	assert.Exactly(t, []*PodLifecycleEvent{}, actualEvents)

	// listing 3 - pretend the transient container has now disappeared, leaving just the infra
	// container. Make sure the pod is reinspected for its status and the cache is updated.
	runtimeMock.EXPECT().GetPods(ctx, true).Return(pods, nil).Times(1)
	runtimeMock.EXPECT().GetPodStatus(ctx, podID, "", "").Return(goodStatus, nil).Times(1)

	pleg.Relist()
	actualEvents = getEventsFromChannel(ch)
	actualStatus, actualErr = pleg.cache.Get(podID)
	assert.Equal(t, goodStatus, actualStatus)
	assert.NoError(t, actualErr)
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
	pleg.Relist()
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
	pleg.Relist()
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
	pleg.Relist()
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

func TestRelistIPChange(t *testing.T) {
	ctx := context.Background()
	testCases := []struct {
		name   string
		podID  string
		podIPs []string
	}{
		{
			name:   "test-0",
			podID:  "test-pod-0",
			podIPs: []string{"192.168.1.5"},
		},
		{
			name:   "tets-1",
			podID:  "test-pod-1",
			podIPs: []string{"192.168.1.5/24", "2000::"},
		},
	}

	for _, tc := range testCases {
		runtimeMock := containertest.NewMockRuntime(t)

		pleg := newTestGenericPLEGWithRuntimeMock(runtimeMock)
		ch := pleg.Watch()

		id := types.UID(tc.podID)
		cState := kubecontainer.ContainerStateRunning
		container := createTestContainer("c0", cState)
		pod := &kubecontainer.Pod{
			ID:         id,
			Containers: []*kubecontainer.Container{container},
		}
		status := &kubecontainer.PodStatus{
			ID:                id,
			IPs:               tc.podIPs,
			ContainerStatuses: []*kubecontainer.Status{{ID: container.ID, State: cState}},
		}
		event := &PodLifecycleEvent{ID: pod.ID, Type: ContainerStarted, Data: container.ID.ID}

		runtimeMock.EXPECT().GetPods(ctx, true).Return([]*kubecontainer.Pod{pod}, nil).Times(1)
		runtimeMock.EXPECT().GetPodStatus(ctx, pod.ID, "", "").Return(status, nil).Times(1)

		pleg.Relist()
		actualEvents := getEventsFromChannel(ch)
		actualStatus, actualErr := pleg.cache.Get(pod.ID)
		assert.Equal(t, status, actualStatus, tc.name)
		assert.NoError(t, actualErr, tc.name)
		assert.Exactly(t, []*PodLifecycleEvent{event}, actualEvents)

		// Clear the IP address and mark the container terminated
		container = createTestContainer("c0", kubecontainer.ContainerStateExited)
		pod = &kubecontainer.Pod{
			ID:         id,
			Containers: []*kubecontainer.Container{container},
		}
		status = &kubecontainer.PodStatus{
			ID:                id,
			ContainerStatuses: []*kubecontainer.Status{{ID: container.ID, State: kubecontainer.ContainerStateExited}},
		}
		event = &PodLifecycleEvent{ID: pod.ID, Type: ContainerDied, Data: container.ID.ID}
		runtimeMock.EXPECT().GetPods(ctx, true).Return([]*kubecontainer.Pod{pod}, nil).Times(1)
		runtimeMock.EXPECT().GetPodStatus(ctx, pod.ID, "", "").Return(status, nil).Times(1)

		pleg.Relist()
		actualEvents = getEventsFromChannel(ch)
		actualStatus, actualErr = pleg.cache.Get(pod.ID)
		// Must copy status to compare since its pointer gets passed through all
		// the way to the event
		statusCopy := *status
		statusCopy.IPs = tc.podIPs
		assert.Equal(t, &statusCopy, actualStatus, tc.name)
		require.NoError(t, actualErr, tc.name)
		assert.Exactly(t, []*PodLifecycleEvent{event}, actualEvents)
	}
}

func TestRunningPodAndContainerCount(t *testing.T) {
	metrics.Register()
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime

	runtime.AllPodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateRunning),
				createTestContainer("c2", kubecontainer.ContainerStateUnknown),
				createTestContainer("c3", kubecontainer.ContainerStateUnknown),
			},
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("s1", kubecontainer.ContainerStateRunning),
				createTestContainer("s2", kubecontainer.ContainerStateRunning),
				createTestContainer("s3", kubecontainer.ContainerStateUnknown),
			},
		}},
		{Pod: &kubecontainer.Pod{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
			},
			Sandboxes: []*kubecontainer.Container{
				createTestContainer("s1", kubecontainer.ContainerStateRunning),
				createTestContainer("s2", kubecontainer.ContainerStateExited),
			},
		}},
	}

	pleg.Relist()

	tests := []struct {
		name        string
		metricsName string
		wants       string
	}{
		{
			name:        "test container count",
			metricsName: "kubelet_running_containers",
			wants: `
# HELP kubelet_running_containers [ALPHA] Number of containers currently running
# TYPE kubelet_running_containers gauge
kubelet_running_containers{container_state="exited"} 1
kubelet_running_containers{container_state="running"} 1
kubelet_running_containers{container_state="unknown"} 2
`,
		},
		{
			name:        "test pod count",
			metricsName: "kubelet_running_pods",
			wants: `
# HELP kubelet_running_pods [ALPHA] Number of pods that have a running pod sandbox
# TYPE kubelet_running_pods gauge
kubelet_running_pods 2
`,
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(tc.wants), tc.metricsName); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestWatchConditions(t *testing.T) {
	pods := []*kubecontainer.Pod{{
		Name: "running-pod",
		ID:   "running",
		Sandboxes: []*kubecontainer.Container{
			createTestContainer("s", kubecontainer.ContainerStateRunning),
		},
		Containers: []*kubecontainer.Container{
			createTestContainer("c", kubecontainer.ContainerStateRunning),
		},
	}, {
		Name: "running-pod-2",
		ID:   "running-2",
		Sandboxes: []*kubecontainer.Container{
			createTestContainer("s", kubecontainer.ContainerStateRunning),
		},
		Containers: []*kubecontainer.Container{
			createTestContainer("c-exited", kubecontainer.ContainerStateExited),
			createTestContainer("c-running", kubecontainer.ContainerStateRunning),
		},
	}, {
		Name: "terminating-pod",
		ID:   "terminating",
		Sandboxes: []*kubecontainer.Container{
			createTestContainer("s", kubecontainer.ContainerStateExited),
		},
	}, {
		Name: "reinspect-pod",
		ID:   "reinspect",
		Sandboxes: []*kubecontainer.Container{
			createTestContainer("s", kubecontainer.ContainerStateRunning),
		},
	}}
	initialPods := pods
	initialPods = append(initialPods, &kubecontainer.Pod{
		Name: "terminated-pod",
		ID:   "terminated",
		Sandboxes: []*kubecontainer.Container{
			createTestContainer("s", kubecontainer.ContainerStateExited),
		},
	})

	alwaysComplete := func(_ *kubecontainer.PodStatus) bool {
		return true
	}
	neverComplete := func(_ *kubecontainer.PodStatus) bool {
		return false
	}

	var pleg *GenericPLEG
	var updatingCond WatchCondition
	// updatingCond always completes, but updates the condition first.
	updatingCond = func(_ *kubecontainer.PodStatus) bool {
		pleg.SetPodWatchCondition("running", "updating", updatingCond)
		return true
	}

	// resettingCond decrements the version before it completes.
	var resettingCond = func(_ *kubecontainer.PodStatus) bool {
		versioned := pleg.watchConditions["running"]["resetting"]
		versioned.version = 0
		pleg.watchConditions["running"]["resetting"] = versioned
		return true
	}

	// makeContainerCond returns a RunningContainerWatchCondition that asserts the expected container status
	makeContainerCond := func(expectedContainerName string, complete bool) WatchCondition {
		return RunningContainerWatchCondition(expectedContainerName, func(status *kubecontainer.Status) bool {
			if status.Name != expectedContainerName {
				panic(fmt.Sprintf("unexpected container name: got %q, want %q", status.Name, expectedContainerName))
			}
			return complete
		})
	}

	testCases := []struct {
		name                    string
		podUID                  types.UID
		watchConditions         map[string]WatchCondition
		incrementInitialVersion bool                               // Whether to call SetPodWatchCondition multiple times to increment the version
		expectEvaluated         bool                               // Whether the watch conditions should be evaluated
		expectRemoved           bool                               // Whether podUID should be present in the watch conditions map
		expectWatchConditions   map[string]versionedWatchCondition // The expected watch conditions for the podUID (only key & version checked)
	}{{
		name:   "no watch conditions",
		podUID: "running",
	}, {
		name:   "running pod with conditions",
		podUID: "running",
		watchConditions: map[string]WatchCondition{
			"completing": alwaysComplete,
			"watching":   neverComplete,
			"updating":   updatingCond,
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"watching": {version: 0},
			"updating": {version: 1},
		},
	}, {
		name:                    "conditions with incremented versions",
		podUID:                  "running",
		incrementInitialVersion: true,
		watchConditions: map[string]WatchCondition{
			"completing": alwaysComplete,
			"watching":   neverComplete,
			"updating":   updatingCond,
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"watching": {version: 1},
			"updating": {version: 2},
		},
	}, {
		name:                    "completed watch condition with older version",
		podUID:                  "running",
		incrementInitialVersion: true,
		watchConditions: map[string]WatchCondition{
			"resetting": resettingCond,
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"resetting": {version: 0},
		},
	}, {
		name:   "non-existent pod",
		podUID: "non-existent",
		watchConditions: map[string]WatchCondition{
			"watching": neverComplete,
		},
		expectEvaluated: false,
		expectRemoved:   true,
	}, {
		name:   "terminated pod",
		podUID: "terminated",
		watchConditions: map[string]WatchCondition{
			"watching": neverComplete,
		},
		expectEvaluated: false,
		expectRemoved:   true,
	}, {
		name:   "reinspecting pod",
		podUID: "reinspect",
		watchConditions: map[string]WatchCondition{
			"watching": neverComplete,
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"watching": {version: 0},
		},
	}, {
		name:   "single container conditions",
		podUID: "running",
		watchConditions: map[string]WatchCondition{
			"completing": makeContainerCond("c", true),
			"watching":   makeContainerCond("c", false),
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"watching": {version: 0},
		},
	}, {
		name:   "multi-container conditions",
		podUID: "running-2",
		watchConditions: map[string]WatchCondition{
			"completing:exited":  makeContainerCond("c-exited", true),
			"watching:exited":    makeContainerCond("c-exited", false),
			"completing:running": makeContainerCond("c-running", true),
			"watching:running":   makeContainerCond("c-running", false),
			"completing:dne":     makeContainerCond("c-dne", true),
			"watching:dne":       makeContainerCond("c-dne", false),
		},
		expectEvaluated: true,
		expectWatchConditions: map[string]versionedWatchCondition{
			"watching:running": {version: 0},
		},
	}}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			runtimeMock := containertest.NewMockRuntime(t)
			pleg = newTestGenericPLEGWithRuntimeMock(runtimeMock)

			// Mock pod statuses
			for _, pod := range initialPods {
				podStatus := &kubecontainer.PodStatus{
					ID:        pod.ID,
					Name:      pod.Name,
					Namespace: pod.Namespace,
				}
				for _, c := range pod.Containers {
					podStatus.ContainerStatuses = append(podStatus.ContainerStatuses, &kubecontainer.Status{
						ID:    c.ID,
						Name:  c.Name,
						State: c.State,
					})
				}
				runtimeMock.EXPECT().
					GetPodStatus(mock.Anything, pod.ID, pod.Name, pod.Namespace).
					Return(podStatus, nil).Maybe()
			}

			// Setup initial pod records.
			runtimeMock.EXPECT().GetPods(mock.Anything, true).Return(initialPods, nil).Once()
			pleg.Relist()
			pleg.podsToReinspect["reinspect"] = nil

			// Remove "terminated" pod.
			runtimeMock.EXPECT().GetPods(mock.Anything, true).Return(pods, nil).Once()

			var evaluatedConditions []string
			for key, condition := range test.watchConditions {
				wrappedCondition := func(status *kubecontainer.PodStatus) bool {
					defer func() {
						if r := recover(); r != nil {
							require.Fail(t, "condition error", r)
						}
					}()
					assert.Equal(t, test.podUID, status.ID, "podUID")
					if !test.expectEvaluated {
						assert.Fail(t, "conditions should not be evaluated")
					} else {
						evaluatedConditions = append(evaluatedConditions, key)
					}
					return condition(status)
				}
				pleg.SetPodWatchCondition(test.podUID, key, wrappedCondition)
				if test.incrementInitialVersion {
					// Set the watch condition a second time to increment the version.
					pleg.SetPodWatchCondition(test.podUID, key, wrappedCondition)
				}
			}
			pleg.Relist()

			if test.expectEvaluated {
				assert.Len(t, evaluatedConditions, len(test.watchConditions), "all conditions should be evaluated")
			}

			if test.expectRemoved {
				assert.NotContains(t, pleg.watchConditions, test.podUID, "Pod should be removed from watch conditions")
			} else {
				actualConditions := pleg.watchConditions[test.podUID]
				assert.Len(t, actualConditions, len(test.expectWatchConditions), "expected number of conditions")
				for key, expected := range test.expectWatchConditions {
					if !assert.Contains(t, actualConditions, key) {
						continue
					}
					actual := actualConditions[key]
					assert.Equal(t, key, actual.key)
					assert.Equal(t, expected.version, actual.version)
				}
			}

		})
	}
}
