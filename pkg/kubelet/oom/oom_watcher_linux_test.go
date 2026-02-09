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
	"encoding/json"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/test/utils/ktesting"

	"github.com/google/cadvisor/utils/oomparser"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeStreamer struct {
	oomInstancesToStream []*oomparser.OomInstance
}

func (fs *fakeStreamer) StreamOoms(outStream chan<- *oomparser.OomInstance) {
	for _, oomInstance := range fs.oomInstancesToStream {
		outStream <- oomInstance
	}
}

type fakeCheckPointManager struct {
	timestamp time.Time
}

func (c fakeCheckPointManager) CreateCheckpoint(checkpointKey string, checkpoint checkpointmanager.Checkpoint) error {
	return nil
}

func (c fakeCheckPointManager) GetCheckpoint(checkpointKey string, checkpoint checkpointmanager.Checkpoint) error {
	s := Checkpoint{LastProcessedTimestamp: c.timestamp}
	res, err := json.Marshal(s)
	if err != nil {
		return err
	}
	return checkpoint.UnmarshalCheckpoint(res)
}

func (c fakeCheckPointManager) RemoveCheckpoint(checkpointKey string) error {
	return nil
}

func (c fakeCheckPointManager) ListCheckpoints() ([]string, error) {
	return []string{}, nil
}

// TestWatcherRecordsEventsForOomEvents ensures that our OomInstances coming
// from `StreamOoms` are translated into events in our recorder.
func TestWatcherRecordsEventsForOomEvents(t *testing.T) {
	tCtx := ktesting.Init(t)
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
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
		checkPoint: &Checkpoint{
			LastProcessedTimestamp: time.Now().Add(-1 * time.Hour),
		},
		checkPointManager: fakeCheckPointManager{},
	}
	require.NoError(t, oomWatcher.Start(tCtx, node))

	eventsRecorded := getRecordedEvents(fakeRecorder, numExpectedOomEvents)
	assert.Len(t, eventsRecorded, numExpectedOomEvents)
}

func TestWatcherIgnoreOutOfOrderEvents(t *testing.T) {
	tCtx := ktesting.Init(t)
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess-1",
			TimeOfDeath:         time.Now().Add(1 * time.Hour),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
		},
		{
			Pid:                 1001,
			ProcessName:         "fakeProcess-2",
			TimeOfDeath:         time.Now().Add(2 * time.Hour),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
		},
		{
			Pid:                 1002,
			ProcessName:         "fakeProcess-3",
			TimeOfDeath:         time.Now().Add(-2 * time.Hour),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
		},
	}
	numExpectedOomEvents := 1 // Since the current time is set +90 min from now, the first and third messages will be discarded

	fakeStreamer := &fakeStreamer{
		oomInstancesToStream: oomInstancesToStream,
	}

	fakeRecorder := record.NewFakeRecorder(numExpectedOomEvents)
	node := &v1.ObjectReference{}

	oomWatcher := &realWatcher{
		recorder:    fakeRecorder,
		oomStreamer: fakeStreamer,
		checkPoint:  &Checkpoint{},
		checkPointManager: fakeCheckPointManager{
			timestamp: time.Now().Add(90 * time.Minute),
		},
	}

	require.NoError(t, oomWatcher.Start(tCtx, node))

	eventsRecorded := getRecordedEvents(fakeRecorder, numExpectedOomEvents)
	assert.Len(t, eventsRecorded, 1)
}

func getRecordedEvents(fakeRecorder *record.FakeRecorder, numExpectedOomEvents int) []string {
	eventsRecorded := []string{}

loop:
	for {
		select {
		case event := <-fakeRecorder.Events:
			eventsRecorded = append(eventsRecorded, event)
		case <-time.After(500 * time.Millisecond):
			break loop
		}
	}
	return eventsRecorded

}

// TestWatcherRecordsEventsForOomEventsCorrectContainerName verifies that we
// only record OOM events when the container name is the one for which we want
// to record events (i.e. /).
func TestWatcherRecordsEventsForOomEventsCorrectContainerName(t *testing.T) {
	// By "incorrect" container name, we mean a container name for which we
	// don't want to record an oom event.
	tCtx := ktesting.Init(t)
	numOomEventsWithIncorrectContainerName := 1
	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
		},
		{
			Pid:                 1000,
			ProcessName:         "fakeProcess",
			TimeOfDeath:         time.Now(),
			ContainerName:       recordEventContainerName + "kubepods/some-container",
			VictimContainerName: recordEventContainerName + "kubepods",
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
		checkPoint: &Checkpoint{
			LastProcessedTimestamp: time.Now().Add(-1 * time.Hour),
		},
		checkPointManager: fakeCheckPointManager{},
	}
	require.NoError(t, oomWatcher.Start(tCtx, node))

	eventsRecorded := getRecordedEvents(fakeRecorder, numExpectedOomEvents)
	assert.Len(t, eventsRecorded, numExpectedOomEvents)
}

// TestWatcherRecordsEventsForOomEventsWithAdditionalInfo verifies that our the
// emitted event has the proper pid/process data when appropriate.
func TestWatcherRecordsEventsForOomEventsWithAdditionalInfo(t *testing.T) {
	// The process and event info should appear in the event message.
	eventPid := 1000
	processName := "fakeProcess"

	tCtx := ktesting.Init(t)

	oomInstancesToStream := []*oomparser.OomInstance{
		{
			Pid:                 eventPid,
			ProcessName:         processName,
			TimeOfDeath:         time.Now(),
			ContainerName:       recordEventContainerName + "some-container",
			VictimContainerName: recordEventContainerName,
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
		checkPoint: &Checkpoint{
			LastProcessedTimestamp: time.Now().Add(-1 * time.Hour),
		},
		checkPointManager: fakeCheckPointManager{},
	}
	require.NoError(t, oomWatcher.Start(tCtx, node))

	eventsRecorded := getRecordedEvents(fakeRecorder, numExpectedOomEvents)

	assert.Len(t, eventsRecorded, numExpectedOomEvents)
	assert.Contains(t, eventsRecorded[0], systemOOMEvent)
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("pid: %d", eventPid))
	assert.Contains(t, eventsRecorded[0], fmt.Sprintf("victim process: %s", processName))
}
