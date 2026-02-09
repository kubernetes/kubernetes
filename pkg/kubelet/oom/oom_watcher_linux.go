//go:build linux

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
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"

	"github.com/google/cadvisor/utils/oomparser"
)

type streamer interface {
	StreamOoms(chan<- *oomparser.OomInstance)
}

var _ streamer = &oomparser.OomParser{}

type Checkpoint struct {
	LastProcessedTimestamp time.Time
}

func (s *Checkpoint) MarshalCheckpoint() ([]byte, error) {
	return json.Marshal(s)
}

func (s *Checkpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, s)
}

func (s *Checkpoint) VerifyChecksum() error {
	if s.LastProcessedTimestamp.IsZero() {
		return fmt.Errorf("LastProcessedTimestamp is zero")
	}
	return nil
}

func (s *Checkpoint) isNewEvent(newTime time.Time) bool {
	return newTime.After(s.LastProcessedTimestamp)
}

var _ checkpointmanager.Checkpoint = &Checkpoint{}

type realWatcher struct {
	recorder          record.EventRecorder
	oomStreamer       streamer
	checkPoint        *Checkpoint
	checkPointManager checkpointmanager.CheckpointManager
}

var _ Watcher = &realWatcher{}

// NewWatcher creates and initializes a OOMWatcher backed by Cadvisor as
// the oom streamer.
func NewWatcher(recorder record.EventRecorder, rootDir string) (Watcher, error) {
	// for test purpose
	_, ok := recorder.(*record.FakeRecorder)
	if ok {
		return nil, nil
	}

	manager, err := checkpointmanager.NewCheckpointManager(rootDir)
	if err != nil {
		return nil, err
	}

	oomStreamer, err := oomparser.New()
	if err != nil {
		return nil, err
	}
	watcher := &realWatcher{
		recorder:          recorder,
		oomStreamer:       oomStreamer,
		checkPoint:        &Checkpoint{},
		checkPointManager: manager,
	}

	return watcher, nil
}

const (
	systemOOMEvent           = "SystemOOM"
	recordEventContainerName = "/"
	checkpointKey            = "oom_watcher"
)

// Start watches for system oom's and records an event for every system oom encountered.
func (ow *realWatcher) Start(ctx context.Context, ref *v1.ObjectReference) error {
	logger := klog.FromContext(ctx)

	if err := ow.checkPointManager.GetCheckpoint(checkpointKey, ow.checkPoint); err != nil {
		ow.checkPoint.LastProcessedTimestamp = time.Now()
		logger.Info("unable to restore OOMCheckpoint from file", "OOMCheckpoint", ow.checkPoint.LastProcessedTimestamp, "error", err)
	}

	outStream := make(chan *oomparser.OomInstance, 10)
	go ow.oomStreamer.StreamOoms(outStream)

	go func() {
		defer runtime.HandleCrashWithContext(ctx)

		for event := range outStream {
			if event.VictimContainerName == recordEventContainerName {
				logger.V(1).Info("Got sys oom event", "event", event)
				if !ow.checkPoint.isNewEvent(event.TimeOfDeath) {
					logger.V(1).Info("ignore out of the order event", "event", event, "LastProcessedTimestamp", ow.checkPoint.LastProcessedTimestamp)
					continue
				}

				eventMsg := "System OOM encountered"
				if event.ProcessName != "" && event.Pid != 0 {
					eventMsg = fmt.Sprintf("%s, victim process: %s, pid: %d", eventMsg, event.ProcessName, event.Pid)
				}
				ow.recorder.Eventf(ref, v1.EventTypeWarning, systemOOMEvent, eventMsg)
				ow.checkPoint.LastProcessedTimestamp = event.TimeOfDeath
				if err := ow.checkPointManager.CreateCheckpoint(checkpointKey, ow.checkPoint); err != nil {
					logger.Error(err, "Unable to store watcher OOMCheckpoint")
				}
			}
		}
		logger.Error(nil, "Unexpectedly stopped receiving OOM notifications")
	}()
	return nil
}
