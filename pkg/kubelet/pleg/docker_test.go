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
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/types"
)

const (
	testWaitForEventsTimeout = time.Minute * 1
)

type fakeContainerGetter struct {
	lock     sync.RWMutex
	running  []*kubecontainer.Container
	dead     []*kubecontainer.Container
	cResults map[string]*dockertools.ContainerExaminationResult
}

func newFakeContainerGetter() *fakeContainerGetter {
	return &fakeContainerGetter{cResults: make(map[string]*dockertools.ContainerExaminationResult)}
}

func (f *fakeContainerGetter) GetRunningContainers() ([]*kubecontainer.Container, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.running, nil
}

func (f *fakeContainerGetter) GetTerminatedContainers() ([]*kubecontainer.Container, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.dead, nil
}

func (f *fakeContainerGetter) ExamineContainer(dockerID string) (*dockertools.ContainerExaminationResult, error) {
	f.lock.RLock()
	defer f.lock.RUnlock()
	if c, ok := f.cResults[dockerID]; ok {
		return c, nil
	}
	return nil, fmt.Errorf("cannot examine container %q", dockerID)
}

func (f *fakeContainerGetter) SetContainers(running, dead []*kubecontainer.Pod) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.running = []*kubecontainer.Container{}
	f.dead = []*kubecontainer.Container{}
	for _, p := range running {
		for _, c := range p.Containers {
			f.running = append(f.running, c)
			f.cResults[string(c.ID)] = &dockertools.ContainerExaminationResult{Pod: p}
		}
	}
	for _, p := range dead {
		for _, c := range p.Containers {
			f.dead = append(f.dead, c)
			f.cResults[string(c.ID)] = &dockertools.ContainerExaminationResult{Pod: p}
		}
	}
}

func (f *fakeContainerGetter) SetContainerResults(pods []*kubecontainer.Pod) {
	f.lock.Lock()
	defer f.lock.Unlock()
	for _, p := range pods {
		for _, c := range p.Containers {
			f.cResults[string(c.ID)] = &dockertools.ContainerExaminationResult{Pod: p}
		}
	}
}

type TestDockerPLEG struct {
	pleg                  *DockerPLEG
	containerGetter       *fakeContainerGetter
	containerEventWatcher *fakeContainerEventWatcher
}

func newTestDockerPLEG(relistPeriod time.Duration) *TestDockerPLEG {
	containerGetter := newFakeContainerGetter()
	containerEventWatcher := &fakeContainerEventWatcher{}
	// The channel capacity should be large enough to hold all events in a
	// single test.
	pleg := NewDockerPLEG(containerEventWatcher, containerGetter, 100, relistPeriod)
	return &TestDockerPLEG{pleg: pleg, containerGetter: containerGetter, containerEventWatcher: containerEventWatcher}
}

func TestRelisting(t *testing.T) {
	// Set a high relist period because we want to manually trigger relisting.
	testPleg := newTestDockerPLEG(time.Hour)
	pleg, containerGetter := testPleg.pleg, testPleg.containerGetter
	ch := pleg.Watch()

	// Report containers that are newly running.
	running := []*kubecontainer.Pod{
		{
			ID: "1234",
			Containers: []*kubecontainer.Container{
				{ID: types.UID("c1"), Name: "foo"},
			},
		},
		{
			ID: "7890",
			Containers: []*kubecontainer.Container{
				{ID: types.UID("c2"), Name: "bar"},
			},
		},
	}
	containerGetter.SetContainers(running, []*kubecontainer.Pod{})
	pleg.relist()
	expected := []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerStarted, Data: "foo"},
		{ID: "7890", Type: ContainerStarted, Data: "bar"},
	}
	actual := getEventsFromChannel(ch)
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Expected: %#v, got: %#v", expected, actual)
	}

	// Report containers that are newly dead.
	containerGetter.SetContainers([]*kubecontainer.Pod{}, []*kubecontainer.Pod{})
	pleg.relist()
	expected = []*PodLifecycleEvent{
		{ID: "1234", Type: ContainerStopped, Data: "foo"},
		{ID: "7890", Type: ContainerStopped, Data: "bar"},
	}
	actual = getEventsFromChannel(ch)
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Expected: %#v, got: %#v", expected, actual)
	}

	// If a new dead container has not been seen in running/dead container set,
	// send both creation and deletion event for it.
	dead := []*kubecontainer.Pod{
		{
			ID: "4567",
			Containers: []*kubecontainer.Container{
				{ID: types.UID("c3"), Name: "tar"},
			},
		},
	}
	containerGetter.SetContainers([]*kubecontainer.Pod{}, dead)
	pleg.relist()
	expected = []*PodLifecycleEvent{
		{ID: "4567", Type: ContainerStarted, Data: "tar"},
		{ID: "4567", Type: ContainerStopped, Data: "tar"},
	}
	actual = getEventsFromChannel(ch)
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Expected: %#v, got: %#v", expected, actual)
	}
}

func getNumEventsFromChannel(ch <-chan *PodLifecycleEvent, numEvents int, timeout time.Duration) []*PodLifecycleEvent {
	events := []*PodLifecycleEvent{}
	start := time.Now()
	for len(events) < numEvents && time.Now().Before(start.Add(timeout)) {
		events = append(events, getEventsFromChannel(ch)...)
		time.Sleep(time.Second)
	}
	return events
}

func TestProcessContainerEvents(t *testing.T) {
	// Set a high relist period because we don't want to test relisting here.
	testPleg := newTestDockerPLEG(time.Hour)
	pleg, containerGetter, containerEventWatcher := testPleg.pleg, testPleg.containerGetter, testPleg.containerEventWatcher
	ch := pleg.Watch()

	pods := []*kubecontainer.Pod{
		{
			ID: "9999",
			Containers: []*kubecontainer.Container{
				{ID: types.UID("c1"), Name: "foo"},
			},
		},
		{
			ID: "7890",
			Containers: []*kubecontainer.Container{
				{ID: types.UID("c2"), Name: "bar"},
			},
		},
	}
	// This makes sure that pleg can inspect the container for pod ID, etc.
	containerGetter.SetContainerResults(pods)
	// Start a goroutine to watch container events.
	pleg.Start()
	// Send the container events. The timestamps are set slightly ahead of time
	// so that they are guaranteed to be greater than the last relist
	// timestamp. This ensures that they are not treated as outdated events.
	containerEventWatcher.SetEvents([]*ContainerEvent{
		{ID: "c1", Timestamp: time.Now().Add(time.Hour), Type: ContainerEventStarted},
		{ID: "c2", Timestamp: time.Now().Add(time.Hour), Type: ContainerEventStopped},
	})
	expected := []*PodLifecycleEvent{
		{ID: "9999", Type: ContainerStarted, Data: "foo"},
		{ID: "7890", Type: ContainerStopped, Data: "bar"},
	}
	actual := getNumEventsFromChannel(ch, len(expected), testWaitForEventsTimeout)
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("Expected: %#v, got: %#v", expected, actual)
	}
}
