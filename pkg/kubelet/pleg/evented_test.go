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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	internalapi "k8s.io/cri-api/pkg/apis"
	v1 "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

var _ podRelister = (*fakePodRelister)(nil)

type fakePodRelister struct {
	relistRequests []types.UID
}

func newFakePodRelister() *fakePodRelister {
	return &fakePodRelister{}
}

func (f *fakePodRelister) RequestRelist(logger klog.Logger, podUID types.UID) {
	f.relistRequests = append(f.relistRequests, podUID)
}

func newTestEventedPLEG(podRelister *fakePodRelister) *EventedPLEG {
	return NewEventedPLEG(nil, podRelister, 5)
}

type erroringRuntimeService struct {
	internalapi.RuntimeService
	streamErrors []error
	calls        int
}

func (f *erroringRuntimeService) GetContainerEvents(context.Context, chan *v1.ContainerEventResponse, func(v1.RuntimeService_GetContainerEventsClient)) error {
	err := f.streamErrors[f.calls]
	f.calls++
	return err
}

func TestEventedPLEGRetryExhaustionClosesEventChannelAndLogsLastError(t *testing.T) {
	firstErr := errors.New("first stream failure")
	lastErr := errors.New("last stream failure")
	runtimeService := &erroringRuntimeService{streamErrors: []error{firstErr, lastErr}}
	pleg := newTestEventedPLEG(newFakePodRelister())
	pleg.runtimeService = runtimeService
	pleg.eventedPlegMaxStreamRetries = len(runtimeService.streamErrors)

	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(0)))
	ctx := klog.NewContext(t.Context(), logger)
	done := make(chan struct{})
	go func() {
		defer close(done)
		pleg.watchEventsChannel(ctx)
	}()

	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("EventedPLEG did not stop after exhausting stream retries")
	}

	assert.Equal(t, len(runtimeService.streamErrors), runtimeService.calls)

	var terminalLogs int
	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		if entry.Message != "Evented PLEG: disabling container-termination fast path after exhausting event stream retries" {
			continue
		}
		terminalLogs++
		require.ErrorIs(t, entry.Err, lastErr)
		assert.Equal(t, []any{"retries", len(runtimeService.streamErrors)}, entry.ParameterKVList)
	}
	assert.Equal(t, 1, terminalLogs)
}

func TestEventedPLEGWatcherRetriesWhenStreamClosesWithoutError(t *testing.T) {
	runtimeService := &erroringRuntimeService{streamErrors: []error{nil}}
	pleg := newTestEventedPLEG(newFakePodRelister())
	pleg.runtimeService = runtimeService
	pleg.eventedPlegMaxStreamRetries = len(runtimeService.streamErrors)

	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(0)))
	ctx := klog.NewContext(t.Context(), logger)
	done := make(chan struct{})
	go func() {
		defer close(done)
		pleg.watchEventsChannel(ctx)
	}()

	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("EventedPLEG did not stop after the event stream closed without an error")
	}

	assert.Equal(t, 1, runtimeService.calls)
	var terminalLogs int
	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		if entry.Message != "Evented PLEG: disabling container-termination fast path after exhausting event stream retries" {
			continue
		}
		terminalLogs++
		require.ErrorIs(t, entry.Err, errEventedPLEGStreamClosed)
	}
	assert.Equal(t, 1, terminalLogs)
}

type blockingRuntimeService struct {
	internalapi.RuntimeService
	started chan struct{}
}

func (f *blockingRuntimeService) GetContainerEvents(ctx context.Context, _ chan *v1.ContainerEventResponse, _ func(v1.RuntimeService_GetContainerEventsClient)) error {
	close(f.started)
	<-ctx.Done()
	return ctx.Err()
}

func TestEventedPLEGWatcherStopsWhenContextIsCanceled(t *testing.T) {
	runtimeService := &blockingRuntimeService{started: make(chan struct{})}
	pleg := newTestEventedPLEG(newFakePodRelister())
	pleg.runtimeService = runtimeService

	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(0)))
	ctx, cancel := context.WithCancel(klog.NewContext(t.Context(), logger))
	done := make(chan struct{})
	go func() {
		defer close(done)
		pleg.watchEventsChannel(ctx)
	}()

	select {
	case <-runtimeService.started:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("EventedPLEG did not start watching container events")
	}
	cancel()

	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("EventedPLEG did not stop after its context was canceled")
	}

	for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
		assert.NotEqual(t, "Evented PLEG: disabling container-termination fast path after exhausting event stream retries", entry.Message)
	}
}

func TestProcessCRIEventsRequestsRelistOnlyForStoppedEvents(t *testing.T) {
	tests := []struct {
		name       string
		event      *v1.ContainerEventResponse
		wantRelist []types.UID
	}{
		{
			name: "stopped non-zero exit requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 2,
			}),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "stopped OOMKilled requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 0,
				Reason:   "OOMKilled",
			}),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "stopped clean exit requests relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 0,
			}),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "started event waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STARTED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_RUNNING,
				ExitCode: 2,
			}),
		},
		{
			name: "deleted event waits for generic relist",
			event: newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_DELETED_EVENT, &v1.ContainerStatus{
				Id:       "container1",
				State:    v1.ContainerState_CONTAINER_EXITED,
				ExitCode: 2,
			}),
		},
		{
			name:       "stopped event without container status requests relist",
			event:      newContainerEvent("pod1", "container1", v1.ContainerEventType_CONTAINER_STOPPED_EVENT),
			wantRelist: []types.UID{"pod1"},
		},
		{
			name: "missing sandbox metadata waits for generic relist",
			event: &v1.ContainerEventResponse{
				ContainerId:        "container1",
				ContainerEventType: v1.ContainerEventType_CONTAINER_STOPPED_EVENT,
				CreatedAt:          time.Now().UnixNano(),
				ContainersStatuses: []*v1.ContainerStatus{{
					Id:       "container1",
					State:    v1.ContainerState_CONTAINER_EXITED,
					ExitCode: 2,
				}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podRelister := newFakePodRelister()
			eventedPleg := newTestEventedPLEG(podRelister)
			logger := ktesting.NewLogger(t, ktesting.DefaultConfig)
			eventsCh := make(chan *v1.ContainerEventResponse, 1)
			eventsCh <- test.event
			close(eventsCh)

			eventedPleg.processCRIEvents(logger, eventsCh)

			assert.Equal(t, test.wantRelist, podRelister.relistRequests)
		})
	}
}

func newContainerEvent(podUID types.UID, containerID string, eventType v1.ContainerEventType, statuses ...*v1.ContainerStatus) *v1.ContainerEventResponse {
	return &v1.ContainerEventResponse{
		ContainerId:        containerID,
		ContainerEventType: eventType,
		CreatedAt:          time.Now().UnixNano(),
		PodSandboxStatus: &v1.PodSandboxStatus{
			Metadata: &v1.PodSandboxMetadata{
				Uid: string(podUID),
			},
		},
		ContainersStatuses: statuses,
	}
}
