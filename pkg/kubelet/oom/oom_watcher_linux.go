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
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/third_party/forked/cadvisor/oomparser"
)

type streamer interface {
	StreamOoms(ctx context.Context, outStream chan<- *oomparser.OomInstance)
}

var _ streamer = &oomparser.OomParser{}

type realWatcher struct {
	recorder    record.EventRecorderLogger
	oomStreamer streamer
}

var _ Watcher = &realWatcher{}

// NewWatcher creates and initializes an OOMWatcher backed by the kernel log
// (/dev/kmsg) oom streamer.
func NewWatcher(recorder record.EventRecorderLogger) (Watcher, error) {
	// for test purpose
	_, ok := recorder.(*record.FakeRecorder)
	if ok {
		return nil, nil
	}

	oomStreamer, err := oomparser.New()
	if err != nil {
		return nil, err
	}

	watcher := &realWatcher{
		recorder:    recorder,
		oomStreamer: oomStreamer,
	}

	return watcher, nil
}

const (
	systemOOMEvent           = "SystemOOM"
	recordEventContainerName = "/"
)

// Start watches for system oom's and records an event for every system oom encountered.
func (ow *realWatcher) Start(ctx context.Context, ref *v1.ObjectReference) error {
	outStream := make(chan *oomparser.OomInstance, 10)
	go ow.oomStreamer.StreamOoms(ctx, outStream)

	go func() {
		logger := klog.FromContext(ctx)
		defer runtime.HandleCrashWithContext(ctx)

		for event := range outStream {
			// Count every OOM kill per container to back the
			// container_oom_events_total metric.
			recordOOMKill(event.ContainerName)

			if event.VictimContainerName == recordEventContainerName {
				logger.V(1).Info("Got sys oom event", "event", event)
				eventMsg := "System OOM encountered"
				if event.ProcessName != "" && event.Pid != 0 {
					eventMsg = fmt.Sprintf("%s, victim process: %s, pid: %d", eventMsg, event.ProcessName, event.Pid)
				}
				ow.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, systemOOMEvent, "%s", eventMsg)
			}
		}
		logger.Error(nil, "Unexpectedly stopped receiving OOM notifications")
	}()
	return nil
}
