/*
Copyright 2022 The Kubernetes Authors.

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
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

const eventedPLEGContainerEventsChannelCapacity = 1000

var errEventedPLEGStreamClosed = errors.New("CRI container event stream closed without an error")

type podRelister interface {
	RequestRelist(logger klog.Logger, podUID types.UID)
}

// EventedPLEG watches CRI container events and requests low-latency pod relists.
// GenericPLEG remains responsible for observing runtime state, updating the cache,
// and producing pod lifecycle events.
type EventedPLEG struct {
	runtimeService internalapi.RuntimeService
	podRelister    podRelister
	startOnce      sync.Once

	eventedPlegMaxStreamRetries int
}

// NewEventedPLEG creates a CRI event watcher that requests low-latency pod relists.
func NewEventedPLEG(runtimeService internalapi.RuntimeService, relister podRelister,
	eventedPlegMaxStreamRetries int) *EventedPLEG {
	return &EventedPLEG{
		runtimeService:              runtimeService,
		podRelister:                 relister,
		eventedPlegMaxStreamRetries: eventedPlegMaxStreamRetries,
	}
}

// Start starts watching CRI container events until the context is canceled or
// stream retries are exhausted.
func (e *EventedPLEG) Start(ctx context.Context) {
	e.startOnce.Do(func() {
		go e.watchEventsChannel(ctx)
	})
}

func (e *EventedPLEG) watchEventsChannel(ctx context.Context) {
	logger := klog.FromContext(ctx)
	containerEventsResponseCh := make(chan *runtimeapi.ContainerEventResponse, eventedPLEGContainerEventsChannelCapacity)

	// Get the container events from the runtime.
	go func() {
		defer close(containerEventsResponseCh)

		numAttempts := 0
		var lastStreamErr error
		for {
			if numAttempts >= e.eventedPlegMaxStreamRetries {
				logger.Error(lastStreamErr, "Evented PLEG: disabling container-termination fast path after exhausting event stream retries", "retries", numAttempts)
				return
			}

			err := e.runtimeService.GetContainerEvents(ctx, containerEventsResponseCh, func(runtimeapi.RuntimeService_GetContainerEventsClient) {
				metrics.EventedPLEGConn.Inc()
			})
			if ctx.Err() != nil {
				return
			}
			if err == nil {
				err = errEventedPLEGStreamClosed
			}
			metrics.EventedPLEGConnErr.Inc()
			numAttempts++
			lastStreamErr = err
			logger.V(4).Info("Evented PLEG: failed to get container events, retrying", "err", err)
		}
	}()

	e.processCRIEvents(logger, containerEventsResponseCh)
}

// processCRIEvents processes the incoming CRI container events stream.
// It acts as a fast-path hint: container terminations request an
// immediate GenericPLEG relist. GenericPLEG remains the only component that
// updates the pod cache and emits pod lifecycle events.
func (e *EventedPLEG) processCRIEvents(logger klog.Logger, containerEventsResponseCh <-chan *runtimeapi.ContainerEventResponse) {
	for event := range containerEventsResponseCh {
		// Ignore the event if PodSandboxStatus is nil.
		// This might happen under some race condition where the podSandbox has
		// been deleted, and therefore container runtime couldn't find the
		// podSandbox for the container when generating the event.
		// It is safe to ignore because
		// a) a event would have been received for the sandbox deletion,
		// b) in worst case, a relist will eventually sync the pod status.
		// TODO(#114371): Figure out a way to handle this case instead of ignoring.
		if event.PodSandboxStatus == nil || event.PodSandboxStatus.Metadata == nil {
			logger.Error(nil, "Evented PLEG: received ContainerEventResponse with nil PodSandboxStatus or PodSandboxStatus.Metadata", "containerEventResponse", event)
			continue
		}

		e.updateLatencyMetric(event)

		if event.ContainerEventType != runtimeapi.ContainerEventType_CONTAINER_STOPPED_EVENT {
			continue
		}

		podID := types.UID(event.PodSandboxStatus.Metadata.Uid)
		e.podRelister.RequestRelist(logger, podID)

		logger.V(4).Info("Evented PLEG: requested pod relist after container termination",
			"podUID", podID,
			"containerID", event.ContainerId,
			"eventType", event.ContainerEventType)
	}
}

func (e *EventedPLEG) updateLatencyMetric(event *runtimeapi.ContainerEventResponse) {
	duration := time.Duration(time.Now().UnixNano()-event.CreatedAt) * time.Nanosecond
	metrics.EventedPLEGConnLatency.Observe(duration.Seconds())
}
