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
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

const eventedPLEGContainerEventsChannelCapacity = 1000

var (
	eventedPLEGUsage   = false
	eventedPLEGUsageMu = sync.RWMutex{}
)

// isEventedPLEGInUse indicates whether EventedPLEG's CRI event stream is active.
// Even after enabling the EventedPLEG feature gate the stream may be inactive due to
// runtime errors or exhausted retries, in which case only GenericPLEG runs.
func isEventedPLEGInUse() bool {
	eventedPLEGUsageMu.RLock()
	defer eventedPLEGUsageMu.RUnlock()
	return eventedPLEGUsage
}

// setEventedPLEGUsage should only be accessed from
// Start/Stop of Evented PLEG.
func setEventedPLEGUsage(enable bool) {
	eventedPLEGUsageMu.Lock()
	defer eventedPLEGUsageMu.Unlock()
	eventedPLEGUsage = enable
}

type EventedPLEG struct {
	// The runtime service.
	runtimeService internalapi.RuntimeService
	// GenericPLEG is used to request on-demand pod relists on crash detection.
	genericPleg podLifecycleEventGeneratorHandler
	// The maximum number of retries when getting container events from the runtime.
	eventedPlegMaxStreamRetries int
	// Stop the Evented PLEG by closing the channel.
	stopCh chan struct{}
	// Locks the start/stop operation of the Evented PLEG.
	runningMu sync.Mutex
}

// NewEventedPLEG instantiates a new EventedPLEG object and return it.
func NewEventedPLEG(runtimeService internalapi.RuntimeService, genericPleg PodLifecycleEventGenerator,
	eventedPlegMaxStreamRetries int) (PodLifecycleEventGenerator, error) {
	handler, ok := genericPleg.(podLifecycleEventGeneratorHandler)
	if !ok {
		return nil, fmt.Errorf("%v doesn't implement podLifecycleEventGeneratorHandler interface", genericPleg)
	}
	return &EventedPLEG{
		runtimeService:              runtimeService,
		genericPleg:                 handler,
		eventedPlegMaxStreamRetries: eventedPlegMaxStreamRetries,
	}, nil
}

// Watch returns a channel from which the subscriber can receive PodLifecycleEvent events.
func (e *EventedPLEG) Watch() chan *PodLifecycleEvent {
	return e.genericPleg.Watch()
}

// Relist relists all containers using GenericPLEG
func (e *EventedPLEG) Relist(ctx context.Context) {
	e.genericPleg.Relist(ctx)
}

func (e *EventedPLEG) RequestRelist(logger klog.Logger, podUID types.UID) {
	e.genericPleg.RequestRelist(logger, podUID)
}

// Start starts the Evented PLEG
func (e *EventedPLEG) Start(ctx context.Context) {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	if isEventedPLEGInUse() {
		return
	}
	setEventedPLEGUsage(true)
	e.stopCh = make(chan struct{})
	go wait.Until(func() { e.watchEventsChannel(ctx) }, 0, e.stopCh)
}

// Stop stops the Evented PLEG
func (e *EventedPLEG) Stop() {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	if !isEventedPLEGInUse() {
		return
	}
	setEventedPLEGUsage(false)
	close(e.stopCh)
}

// Healthy returns true unconditionally. EventedPLEG is an auxiliary accelerator;
// its failure does not degrade the node because GenericPLEG continues to handle all
// lifecycle events at its normal relisting period.
func (e *EventedPLEG) Healthy() (bool, error) {
	return true, nil
}

func (e *EventedPLEG) watchEventsChannel(ctx context.Context) {
	logger := klog.FromContext(ctx)
	containerEventsResponseCh := make(chan *runtimeapi.ContainerEventResponse, eventedPLEGContainerEventsChannelCapacity)
	defer close(containerEventsResponseCh)

	// Get the container events from the runtime.
	go func() {
		numAttempts := 0
		for {
			if numAttempts >= e.eventedPlegMaxStreamRetries {
				if isEventedPLEGInUse() {
					// Disable the accelerator. GenericPLEG continues running at its
					// normal period and handles all lifecycle events on its own.
					logger.Error(nil, "Evented PLEG: disabling unexpected-termination fast path after exhausting event stream retries")
					e.Stop()
					break
				}
			}

			err := e.runtimeService.GetContainerEvents(ctx, containerEventsResponseCh, func(runtimeapi.RuntimeService_GetContainerEventsClient) {
				metrics.EventedPLEGConn.Inc()
			})
			if err != nil {
				metrics.EventedPLEGConnErr.Inc()
				numAttempts++
				logger.V(4).Info("Evented PLEG: failed to get container events, retrying", "err", err)
			}
		}
	}()

	if isEventedPLEGInUse() {
		e.processCRIEvents(logger, containerEventsResponseCh)
	}
}

// processCRIEvents processes the incoming CRI container events stream.
// It acts as a fast-path hint: unexpected container terminations request an
// immediate GenericPLEG relist. GenericPLEG remains the only component that
// updates the pod cache and emits pod lifecycle events.
func (e *EventedPLEG) processCRIEvents(logger klog.Logger, containerEventsResponseCh chan *runtimeapi.ContainerEventResponse) {
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

		criStatus, unexpected := e.unexpectedTerminationStatusForEventContainer(logger, event)
		if !unexpected {
			// Not an unexpected termination; GenericPLEG handles this on its next relist.
			continue
		}

		podID := types.UID(event.PodSandboxStatus.Metadata.Uid)
		e.genericPleg.RequestRelist(logger, podID)

		logger.V(4).Info("Evented PLEG: requested pod relist after unexpected container termination",
			"podUID", podID,
			"containerID", event.ContainerId,
			"exitCode", criStatus.ExitCode,
			"reason", criStatus.Reason,
			"eventType", event.ContainerEventType)
	}
}

// unexpectedTerminationStatusForEventContainer returns the CRI ContainerStatus
// for the triggering container and true when that container unexpectedly
// terminated. ContainersStatuses may contain older and newer instances of the
// same named container, so event.ContainerId is the only safe match key.
func (e *EventedPLEG) unexpectedTerminationStatusForEventContainer(logger klog.Logger, event *runtimeapi.ContainerEventResponse) (*runtimeapi.ContainerStatus, bool) {
	if event.ContainerEventType != runtimeapi.ContainerEventType_CONTAINER_STOPPED_EVENT {
		return nil, false
	}

	cs := eventContainerStatus(event)
	if cs == nil {
		logger.V(5).Info("Evented PLEG: no matching container status in STOPPED event; skipping fast path",
			"containerID", event.ContainerId)
		return nil, false
	}

	if isUnexpectedTermination(cs) {
		return cs, true
	}
	return cs, false
}

func eventContainerStatus(event *runtimeapi.ContainerEventResponse) *runtimeapi.ContainerStatus {
	for _, cs := range event.ContainersStatuses {
		if cs == nil || cs.Id != event.ContainerId {
			continue
		}
		return cs
	}
	return nil
}

func isUnexpectedTermination(status *runtimeapi.ContainerStatus) bool {
	// OOMKilled is always unexpected regardless of exit code.
	if status.Reason == "OOMKilled" {
		return true
	}
	// Non-zero exit on an exited container means the process crashed or was killed.
	return status.State == runtimeapi.ContainerState_CONTAINER_EXITED && status.ExitCode != 0
}

func (e *EventedPLEG) updateLatencyMetric(event *runtimeapi.ContainerEventResponse) {
	duration := time.Duration(time.Now().UnixNano()-event.CreatedAt) * time.Nanosecond
	metrics.EventedPLEGConnLatency.Observe(duration.Seconds())
}

func (e *EventedPLEG) RequestReinspect(podUID types.UID) {
	e.genericPleg.RequestReinspect(podUID)
}
