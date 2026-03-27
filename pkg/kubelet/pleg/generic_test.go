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
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
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
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[0]).Return(statuses[0], nil).Times(1)
	// Inject an error when querying runtime for the pod status for pods[1].
	statusErr := fmt.Errorf("unable to get status")
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[1]).Return(&kubecontainer.PodStatus{}, statusErr).Times(1)

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
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[1]).Return(statuses[1], nil).Times(1)
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
	runtimeMock.EXPECT().GetPodStatus(ctx, pods[0]).Return(statuses[0], nil).Times(1)
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

func TestReinspect(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name              string
		requestReinspect  bool
		alreadyReinspect  bool
		updateCacheError  error
		podDeleted        bool
		expectUpdateCache bool
		expectReinspect   bool // value in podsToReinspect AFTER relist
		expectEvent       bool
		expectStatus      bool
	}{
		{
			name:              "RequestReinspect a pod not previously listed, success",
			requestReinspect:  true,
			alreadyReinspect:  false,
			updateCacheError:  nil,
			expectUpdateCache: true,
			expectReinspect:   false,
			expectEvent:       true,
			expectStatus:      true,
		},
		{
			name:              "RequestReinspect a pod not previously listed, failure",
			requestReinspect:  true,
			alreadyReinspect:  false,
			updateCacheError:  errors.New("fail"),
			expectUpdateCache: true,
			expectReinspect:   true,
			expectEvent:       false,
			expectStatus:      true,
		},
		{
			name:              "RequestReinspect of a pod already listed for reinspection, success",
			requestReinspect:  true,
			alreadyReinspect:  true,
			updateCacheError:  nil,
			expectUpdateCache: true,
			expectReinspect:   false,
			expectEvent:       true,
			expectStatus:      true,
		},
		{
			name:              "RequestReinspect of a pod already listed for reinspection, failure",
			requestReinspect:  true,
			alreadyReinspect:  true,
			updateCacheError:  errors.New("fail"),
			expectUpdateCache: true,
			expectReinspect:   true,
			expectEvent:       false,
			expectStatus:      true,
		},
		{
			name:              "Don't request reinspection",
			requestReinspect:  false,
			alreadyReinspect:  false,
			updateCacheError:  nil,
			expectUpdateCache: false,
			expectReinspect:   false,
			expectEvent:       false,
			expectStatus:      false,
		},
		{
			name:              "Don't request reinspection, but already listed for reinspection, success",
			requestReinspect:  false,
			alreadyReinspect:  true,
			updateCacheError:  nil,
			expectUpdateCache: true,
			expectReinspect:   false,
			expectEvent:       true,
			expectStatus:      true,
		},
		{
			name:              "Don't request reinspection, but already listed for reinspection, failure",
			requestReinspect:  false,
			alreadyReinspect:  true,
			updateCacheError:  errors.New("fail"),
			expectUpdateCache: true,
			expectReinspect:   true,
			expectEvent:       false,
			expectStatus:      true,
		},
		{
			name:              "Pod deleted, should clear reinspect",
			requestReinspect:  false,
			alreadyReinspect:  true,
			podDeleted:        true,
			updateCacheError:  nil,
			expectUpdateCache: true,
			expectReinspect:   false,
			expectEvent:       true,
			expectStatus:      false, // Deleted from cache
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runtimeMock := containertest.NewMockRuntime(t)
			pleg := newTestGenericPLEGWithRuntimeMock(runtimeMock)
			ch := pleg.Watch()

			podID := types.UID("test-pod")
			if tc.alreadyReinspect || tc.requestReinspect {
				pleg.RequestReinspect(podID)
			}

			pod := &kubecontainer.Pod{ID: podID, Name: "name", Namespace: "ns"}
			// Populate podRecords.
			pleg.podRecords[podID] = &podRecord{old: pod, current: pod}

			if tc.podDeleted {
				runtimeMock.EXPECT().GetPods(ctx, true).Return([]*kubecontainer.Pod{}, nil)
			} else {
				runtimeMock.EXPECT().GetPods(ctx, true).Return([]*kubecontainer.Pod{pod}, nil)
			}

			var expectedStatus *kubecontainer.PodStatus
			if tc.updateCacheError == nil {
				expectedStatus = &kubecontainer.PodStatus{ID: podID, TimeStamp: time.Now()}
			}
			if tc.expectUpdateCache {
				if tc.podDeleted {
					// updateCache(ctx, nil, podID) will be called, it doesn't call GetPodStatus
				} else {
					runtimeMock.EXPECT().GetPodStatus(ctx, pod).Return(expectedStatus, tc.updateCacheError)
				}
			}

			pleg.Relist()

			_, actualReinspect := pleg.podsToReinspect.Load(podID)
			assert.Equal(t, tc.expectReinspect, actualReinspect)

			actualEvents := getEventsFromChannel(ch)
			if tc.expectEvent {
				assert.NotEmpty(t, actualEvents, "Expected events to be emitted")
				// We expect at least a PodSync event since pod state didn't change in test setup
				hasPodSync := false
				for _, e := range actualEvents {
					if e.ID == podID && e.Type == PodSync {
						hasPodSync = true
						break
					}
				}
				assert.True(t, hasPodSync, "Expected PodSync event for pod")
			} else {
				assert.Empty(t, actualEvents, "Expected no events to be emitted")
			}

			if tc.expectStatus {
				actualStatus, actualErr := pleg.cache.Get(podID)
				assert.Equal(t, expectedStatus, actualStatus)
				assert.Equal(t, tc.updateCacheError, actualErr)
			} else if tc.podDeleted {
				actualStatus, _ := pleg.cache.Get(podID)
				// If deleted, Get returns an empty status with the ID
				assert.Equal(t, &kubecontainer.PodStatus{ID: podID}, actualStatus)
			}
		})
	}
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
		runtimeMock.EXPECT().GetPodStatus(ctx, pod).Return(status, nil).Times(1)

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
		runtimeMock.EXPECT().GetPodStatus(ctx, pod).Return(status, nil).Times(1)

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

func TestWorkerLoop(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.PLEGOnDemandRelist, true)
	synctest.Test(t, func(t *testing.T) {
		runtimeMock := containertest.NewMockRuntime(t)
		cache := kubecontainer.NewCache()
		pleg := NewGenericPLEG(
			ktesting.NewLogger(t, ktesting.DefaultConfig),
			runtimeMock,
			make(chan *PodLifecycleEvent, 100),
			&RelistDuration{RelistPeriod: 2 * time.Second},
			cache,
			clock.RealClock{},
		).(*GenericPLEG)

		pod1 := &kubecontainer.Pod{ID: "pod1", Name: "pod1", Containers: []*kubecontainer.Container{{ID: kubecontainer.ContainerID{Type: "test", ID: "c1"}, State: kubecontainer.ContainerStateRunning}}}
		pod2 := &kubecontainer.Pod{ID: "pod2", Name: "pod2", Containers: []*kubecontainer.Container{{ID: kubecontainer.ContainerID{Type: "test", ID: "c2"}, State: kubecontainer.ContainerStateRunning}}}

		var call *mock.Call // Used to assert order of mock calls.

		startTime := time.Now()
		pleg.globalRelistTimer = pleg.clock.NewTimer(2 * time.Second)

		// pod1 and pod2 requests should initially be blocked.
		p1res := getNewerThanAsync(t, cache, pod1.ID, startTime)
		requireBlocked(t, p1res)
		p2res := getNewerThanAsync(t, cache, pod2.ID, startTime)
		requireBlocked(t, p2res)

		t.Log("Test single-pod relist (no reinspect)")
		pleg.RequestRelist(pod1.ID)

		mctx := context.Background()
		call = runtimeMock.EXPECT().GetPod(mctx, pod1.ID).RunAndReturn(func(ctx context.Context, uid types.UID) (*kubecontainer.Pod, error) {
			assert.Equal(t, pod1.ID, uid)
			pod1.Timestamp = time.Now()
			return pod1, nil
		}).Once()
		call = runtimeMock.EXPECT().GetPodStatus(mctx, pod1).RunAndReturn(func(_ context.Context, pod *kubecontainer.Pod) (*kubecontainer.PodStatus, error) {
			assert.Equal(t, pod1, pod)
			return &kubecontainer.PodStatus{ID: pod1.ID, TimeStamp: time.Now()}, nil
		}).NotBefore(call).Once()

		pleg.workerLoopIteration()

		p1Status := requireUnblocked(t, p1res) // pod1 is now unblocked
		assert.Equal(t, pod1.ID, p1Status.ID)
		assert.Equal(t, time.Now(), p1Status.TimeStamp)
		requireBlocked(t, p2res) // pod2 is still blocked

		p1NewRes := getNewerThanAsync(t, cache, pod1.ID, startTime.Add(2*time.Second))
		requireBlocked(t, p1NewRes)

		t.Log("Test triggering both global relist and pod2 relist to ensure the global relist gets priority.")
		pleg.RequestRelist(pod2.ID)
		time.Sleep(2 * time.Second)

		call = runtimeMock.EXPECT().GetPods(mctx, true).RunAndReturn(func(_ context.Context, _ bool) ([]*kubecontainer.Pod, error) {
			pod1.Timestamp = time.Now()
			pod2.Timestamp = time.Now()
			return []*kubecontainer.Pod{pod1, pod2}, nil
		}).NotBefore(call).Once()
		call = runtimeMock.EXPECT().GetPodStatus(mctx, pod2).RunAndReturn(func(_ context.Context, pod *kubecontainer.Pod) (*kubecontainer.PodStatus, error) {
			assert.Equal(t, pod2, pod)
			return &kubecontainer.PodStatus{ID: pod2.ID, TimeStamp: time.Now()}, nil
		}).NotBefore(call).Once()

		pleg.workerLoopIteration()

		// The global relist should have unblocked p2res and p1NewRes.
		p2Status := requireUnblocked(t, p2res)
		assert.Equal(t, pod2.ID, p2Status.ID)
		p1NewStatus := requireUnblocked(t, p1NewRes)
		assert.Equal(t, p1Status, p1NewStatus) // Status timestamp should be unchanged

		// The pod2 relist request should NOT trigger a relist, since it was made before the global
		// relist occurred. Drain it from the channel to verify (the mock will trigger an error if GetPod is called for it).
		pleg.workerLoopIteration()

		t.Log("Test reinspection")
		p1ReinpsectRes := getNewerThanAsync(t, cache, pod1.ID, time.Now())
		requireBlocked(t, p1ReinpsectRes)

		// Queue up the next test case: reinspection of pod1.
		pleg.RequestReinspect(pod1.ID)
		pleg.RequestRelist(pod1.ID)
		time.Sleep(2 * time.Second)

		call = runtimeMock.EXPECT().GetPods(mctx, true).RunAndReturn(func(_ context.Context, _ bool) ([]*kubecontainer.Pod, error) {
			pod1.Timestamp = time.Now()
			pod2.Timestamp = time.Now()
			return []*kubecontainer.Pod{pod1, pod2}, nil
		}).NotBefore(call).Once()
		runtimeMock.EXPECT().GetPodStatus(mctx, pod1).RunAndReturn(func(_ context.Context, pod *kubecontainer.Pod) (*kubecontainer.PodStatus, error) {
			assert.Equal(t, pod1, pod)
			return &kubecontainer.PodStatus{ID: pod1.ID, TimeStamp: time.Now()}, nil
		}).NotBefore(call).Once()

		pleg.workerLoopIteration()

		p1ReinspectStatus := requireUnblocked(t, p1ReinpsectRes)
		assert.Equal(t, pod1.ID, p1ReinspectStatus.ID)
		assert.Equal(t, time.Now(), p1ReinspectStatus.TimeStamp) // Status timestamp should be updated.

		// The pod1 relist request should NOT trigger a relist, since it was made before the global
		// relist occurred. Drain it from the channel to verify (the mock will trigger an error if GetPod is called for it).
		pleg.workerLoopIteration()
	})
}

func getNewerThanAsync(t *testing.T, cache kubecontainer.ROCache, podID types.UID, minTime time.Time) <-chan *kubecontainer.PodStatus {
	resCh := make(chan *kubecontainer.PodStatus, 1)
	go func() {
		s, err := cache.GetNewerThan(podID, minTime)
		assert.NoError(t, err)
		resCh <- s
	}()
	return resCh
}

func requireBlocked[T any](t *testing.T, ch <-chan T) {
	t.Helper()
	synctest.Wait()
	select {
	case r := <-ch:
		t.Fatalf("Receive should have blocked, but got: %v", r)
	default:
		// OK.
	}
}

func requireUnblocked[T any](t *testing.T, ch <-chan T) (received T) {
	t.Helper()
	synctest.Wait()
	select {
	case r := <-ch:
		return r
	default:
		t.Fatal("Receive should not have been blocked")
		return
	}
}
