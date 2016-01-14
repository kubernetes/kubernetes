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

package pleg

import (
	"reflect"
	"sort"
	"testing"
	"time"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util"
)

const (
	testContainerRuntimeType = "fooRuntime"
)

type TestGenericPLEG struct {
	pleg    *GenericPLEG
	runtime *kubecontainer.FakeRuntime
}

func newTestGenericPLEG() *TestGenericPLEG {
	fakeRuntime := &kubecontainer.FakeRuntime{}
	// The channel capacity should be large enough to hold all events in a
	// single test.
	pleg := &GenericPLEG{
		relistPeriod: time.Hour,
		runtime:      fakeRuntime,
		eventChannel: make(chan *PodLifecycleEvent, 100),
		containers:   make(map[string]containerInfo),
	}
	return &TestGenericPLEG{pleg: pleg, runtime: fakeRuntime}
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
		t.Errorf("Actual events differ from the expected; diff: %v", util.ObjectDiff(expected, actual))
	}
}

func TestRelisting(t *testing.T) {
	testPleg := newTestGenericPLEG()
	pleg, runtime := testPleg.pleg, testPleg.runtime
	ch := pleg.Watch()

	// The first relist should send a PodSync event to each pod.
	runtime.AllPodList = []*kubecontainer.Pod{
		{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
				createTestContainer("c2", kubecontainer.ContainerStateRunning),
				createTestContainer("c3", kubecontainer.ContainerStateUnknown),
			},
		},
		{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				createTestContainer("c1", kubecontainer.ContainerStateExited),
			},
		},
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

	// The second relist should not send out any event because no container
	// changed.
	pleg.relist()
	verifyEvents(t, expected, actual)

	runtime.AllPodList = []*kubecontainer.Pod{
		{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				createTestContainer("c2", kubecontainer.ContainerStateExited),
				createTestContainer("c3", kubecontainer.ContainerStateRunning),
			},
		},
		{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				createTestContainer("c4", kubecontainer.ContainerStateRunning),
			},
		},
	}
	pleg.relist()
	// Only report containers that transitioned to running or exited status.
	expected = []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerDied, Data: "c2"},
		{ID: "1234", Type: ContainerStarted, Data: "c3"},
		{ID: "4567", Type: ContainerStarted, Data: "c4"},
	}

	actual = getEventsFromChannel(ch)
	verifyEvents(t, expected, actual)
}
