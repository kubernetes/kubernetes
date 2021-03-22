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

package oom

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/cadvisor/utils/oomparser"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
)

type fakeStreamer struct {
	oomInstancesToStream []*oomparser.OomInstance
}

func (fs *fakeStreamer) StreamOoms(outStream chan<- *oomparser.OomInstance) {
	for _, oomInstance := range fs.oomInstancesToStream {
		outStream <- oomInstance
	}
}

// Mock function that is used to find Pods by UID where the Pod is found.
func GetPodByUIDReturnTrue(uid types.UID) (*v1.Pod, bool) {
	return &v1.Pod{}, true
}

// Mock function that is used to find Pods by UID where the Pod is not found.
func GetPodByUIDReturnFalse(uid types.UID) (*v1.Pod, bool) {
	return &v1.Pod{}, false
}

// TestWatcherRecordsEventsForLegacyOomEvents ensures that our OomInstances coming
// from `StreamOoms` are translated into events in our recorder.
func TestWatcherRecordsEventsForLegacyOomEvents(t *testing.T) {
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       legacyRecordEventContainerName,
			VictimContainerName: legacyRecordEventContainerName,
		},
	}
	numExpectedOomEvents := len(oomInstancesToStream)

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnFalse))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)
	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
}

// TestWatcherRecordsEventsForOomEvents ensures that our OomInstances coming
// from `StreamOoms` are translated into events in our recorder.
func TestWatcherRecordsEventsForOomEvents(t *testing.T) {
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
	}
	numExpectedOomEvents := len(oomInstancesToStream) * 2

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnTrue))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)
	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
}

func getRecordedEvents(t *testing.T, fakeRecorder *record.FakeRecorder, numExpectedOomEvents int) []string {
	done := false
	eventsRecorded := []string{}

	for !done {
		select {
		case event := <-fakeRecorder.Events:
			eventsRecorded = append(eventsRecorded, event)
			t.Log(event)
			if len(eventsRecorded) == numExpectedOomEvents {
				done = true
			}
		case <-time.After(10 * time.Second):
			done = true
		}
	}

	return eventsRecorded
}

// TestWatcherRecordsEventsForLegacyOomEventsCorrectContainerName verifies that we
// only record OOM events when the container name is the one for which we want
// to record events (i.e. /).
func TestWatcherRecordsEventsForLegacyOomEventsCorrectContainerName(t *testing.T) {
	// By "incorrect" container name, we mean a container name for which we
	// don't want to record an oom event.
	numOomEventsWithIncorrectContainerName := 1
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       legacyRecordEventContainerName,
			VictimContainerName: legacyRecordEventContainerName,
		},
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       legacyRecordEventContainerName + "kubepods/invalid-container-name",
			VictimContainerName: legacyRecordEventContainerName + "kubepods/invalid-container-name",
		},
	}
	numExpectedOomEvents := len(oomInstancesToStream) - numOomEventsWithIncorrectContainerName

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnFalse))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)
	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
}

// TestWatcherRecordsEventsForOomEventsCorrectContainerName verifies that we
// only record OOM events when the container name is the one for which we want
// to record events.
func TestWatcherRecordsEventsForOomEventsCorrectContainerName(t *testing.T) {
	// By "incorrect" container name, we mean a container name for which we
	// don't want to record an oom event.
	numOomEventsWithIncorrectContainerName := 1
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/invalid-container-name",
			VictimContainerName: "/kubepods/invalid-container-name",
		},
	}
	numExpectedOomEvents := len(oomInstancesToStream) - numOomEventsWithIncorrectContainerName

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnTrue))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)
	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
}

// TestWatcherRecordsEventsForLegacyOomEventsWithAdditionalInfo verifies that the
// emitted event has the proper pid/process data when appropriate.
func TestWatcherRecordsEventsForLegacyOomEventsWithAdditionalInfo(t *testing.T) {
	// The process and event info should appear in the event message.
	eventPid := 1000
	processName := "fakeProcess"

	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 eventPid,
			ProcessName:         processName,
			TimeOfDeath:         time.Now(),
			ContainerName:       legacyRecordEventContainerName,
			VictimContainerName: legacyRecordEventContainerName,
		},
	}
	numExpectedOomEvents := len(oomInstancesToStream)

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnFalse))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)

	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
	assert.Contains(t, eventsRecorded[0], systemOOMEvent)
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("pid: %d", eventPid))
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("victim process: %s", processName))
}

// TestWatcherRecordsEventsForOomEventsWithAdditionalInfo verifies that the
// emitted events have the proper pid/process data for both node and pod
// resources.
func TestWatcherRecordsEventsForOomEventsWithAdditionalInfo(t *testing.T) {
	// The process and event info should appear in the event message.
	eventPid1 := 1000
	processName1 := "fakeProcess"
	eventPid2 := 2000
	processName2 := "fakeProcess2"

	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 eventPid1,
			ProcessName:         processName1,
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
		{
			Pid:                 eventPid2,
			ProcessName:         processName2,
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
	}
	// 4 events are emitted, one each for both the node and pod resource
	numExpectedOomEvents := len(oomInstancesToStream) * 2

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnTrue))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)

	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
	assert.Contains(t, eventsRecorded[0], systemOOMEvent)
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("pid: %d", eventPid1))
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("victim process: %s", processName1))
	assert.Contains(t, eventsRecorded[1], systemOOMEvent)
	assert.Contains(t, eventsRecorded[1], fmt.Sprintf("pid: %d", eventPid1))
	assert.Contains(t, eventsRecorded[1], fmt.Sprintf("victim process: %s", processName1))
	assert.Contains(t, eventsRecorded[2], systemOOMEvent)
	assert.Contains(t, eventsRecorded[2], fmt.Sprintf("pid: %d", eventPid2))
	assert.Contains(t, eventsRecorded[2], fmt.Sprintf("victim process: %s", processName2))
	assert.Contains(t, eventsRecorded[3], systemOOMEvent)
	assert.Contains(t, eventsRecorded[3], fmt.Sprintf("pid: %d", eventPid2))
	assert.Contains(t, eventsRecorded[3], fmt.Sprintf("victim process: %s", processName2))
}

// TestWatcherRecordsEventsForOomEventsWithPodNotFound verifies that the
// emitted node resource events have the proper pid/process data when
// the pod cannot be found.
func TestWatcherRecordsEventsForOomEventsWithPodNotFound(t *testing.T) {
	eventPid1 := 1000
	processName1 := "fakeProcess"
	eventPid2 := 2000
	processName2 := "fakeProcess2"

	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 eventPid1,
			ProcessName:         processName1,
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
		{
			Pid:                 eventPid2,
			ProcessName:         processName2,
			TimeOfDeath:         time.Now(),
			ContainerName:       "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
			VictimContainerName: "/kubepods/burstable/podfa08b07e-b396-4cd5-bda3-f7ebbe2e9b88",
		},
	}
	// 2 events are emitted, only for the node resource when the pod resource is not found
	numExpectedOomEvents := len(oomInstancesToStream)

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
	}
	assert.NoError(t, oomWatcher.Start(node, GetPodByUIDReturnFalse))

	eventsRecorded := getRecordedEvents(t, fakeRecorder, numExpectedOomEvents)
	assert.Equal(t, numExpectedOomEvents, len(eventsRecorded))
	assert.Contains(t, eventsRecorded[0], systemOOMEvent)
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("pid: %d", eventPid1))
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("victim process: %s", processName1))
	assert.Contains(t, eventsRecorded[1], systemOOMEvent)
	assert.Contains(t, eventsRecorded[1], fmt.Sprintf("pid: %d", eventPid2))
	assert.Contains(t, eventsRecorded[1], fmt.Sprintf("victim process: %s", processName2))
}
