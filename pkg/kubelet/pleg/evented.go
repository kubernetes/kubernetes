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
	"fmt"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/clock"
)

type EventedPLEG struct {
	// The container runtime.
	runtime kubecontainer.Runtime

	runtimeService internalapi.RuntimeService

	// The channel from which the subscriber listens events.
	eventChannel chan *PodLifecycleEvent

	// Cache for storing the runtime states required for syncing pods.
	cache kubecontainer.Cache
	// For testability.
	clock clock.Clock
}

// NewEventedPLEG instantiates a new EventedPLEG object and return it.
func NewEventedPLEG(runtime kubecontainer.Runtime, runtimeService internalapi.RuntimeService, eventChannel chan *PodLifecycleEvent,
	cache kubecontainer.Cache, clock clock.Clock) PodLifecycleEventGenerator {
	return &EventedPLEG{
		runtime:        runtime,
		runtimeService: runtimeService,
		eventChannel:   eventChannel,
		cache:          cache,
		clock:          clock,
	}
}

// Watch returns a channel from which the subscriber can receive PodLifecycleEvent events.
// TODO: support multiple subscribers.
func (e *EventedPLEG) Watch() chan *PodLifecycleEvent {
	return e.eventChannel
}

// Start spawns a goroutine to relist periodically.
func (e *EventedPLEG) Start() {
	go wait.Forever(e.watchEventsChannel, 0)
}

// Healthy check if PLEG work properly.
func (e *EventedPLEG) Healthy() (bool, error) {
	// GenericPLEG is declared unhealthy when relisting time is more than
	// the relistThreshold. In case, EventedPLEG is turned on the value of
	// relistingPeriod, relistingThreshold is adjusted to higher values and
	// so health of Generic PLEG should take care of the condition checking
	// higher relistThreshold between last two relist.

	// EventedPLEG is declared unhealthy only if eventChannel is out of capacity.
	if len(e.eventChannel) >= cap(e.eventChannel) {
		return false, fmt.Errorf("EventedPLEG: pleg event channel capacity is full with %v events", len(e.eventChannel))
	}

	timestamp := e.clock.Now()
	metrics.PLEGLastSeen.Set(float64(timestamp.Unix()))
	return true, nil
}

func (e *EventedPLEG) watchEventsChannel() {

	containerEventsResponseCh := make(chan *runtimeapi.ContainerEventResponse, cap(e.eventChannel))
	defer close(containerEventsResponseCh)

	// Get the container events from the runtime.
	go e.runtimeService.GetContainerEvents(containerEventsResponseCh)

	e.processCRIEvents(containerEventsResponseCh)
}

func (e *EventedPLEG) processCRIEvents(containerEventsResponseCh chan *runtimeapi.ContainerEventResponse) {
	for event := range containerEventsResponseCh {
		e.updatePodStatus(event)
		e.processCRIEvent(event)
	}
}

func (e *EventedPLEG) updatePodStatus(event *runtimeapi.ContainerEventResponse) {

	podID := types.UID(event.PodSandboxMetadata.Uid)
	podName := event.PodSandboxMetadata.Name
	podNamespace := event.PodSandboxMetadata.Namespace

	timestamp := e.clock.Now()
	status, err := e.runtime.GetPodStatus(podID, podName, podNamespace)

	if err != nil {
		// nolint:logcheck // Not using the result of klog.V inside the
		// if branch is okay, we just use it to determine whether the
		// additional "podStatus" key and its value should be added.
		if klog.V(6).Enabled() {
			klog.ErrorS(err, "Evented PLEG: Write status", "pod", klog.KRef(podNamespace, podName), "podStatus", status)
		} else {
			klog.ErrorS(err, "Evented PLEG: Write status", "pod", klog.KRef(podNamespace, podName))
		}
	} else {
		if klogV := klog.V(6); klogV.Enabled() {
			klogV.InfoS("Evented PLEG: Write status", "pod", klog.KRef(podNamespace, podName), "podStatus", status)
		} else {
			klog.V(4).InfoS("Evented PLEG: Write status", "pod", klog.KRef(podNamespace, podName))
		}
		// Preserve the pod IP across cache updates if the new IP is empty.
		// When a pod is torn down, kubelet may race with PLEG and retrieve
		// a pod status after network teardown, but the kubernetes API expects
		// the completed pod's IP to be available after the pod is dead.
		status.IPs = e.getPodIPs(podID, status)
	}

	e.updateRunningPodMetric(status)
	e.updateRunningContainerMetric(status)

	if event.ContainerEventType == runtimeapi.ContainerEventType_CONTAINER_DELETED_EVENT {
		for _, sandbox := range status.SandboxStatuses {
			if sandbox.Id == event.ContainerId {
				e.cache.Delete(podID)
			}
		}
	} else {
		e.cache.Set(podID, status, err, timestamp)
	}
	e.cache.UpdateTime(timestamp)
}

func (e *EventedPLEG) processCRIEvent(event *runtimeapi.ContainerEventResponse) {
	switch event.ContainerEventType {
	case runtimeapi.ContainerEventType_CONTAINER_STOPPED_EVENT:
		e.sendPodLifecycleEvent(&PodLifecycleEvent{ID: types.UID(event.PodSandboxMetadata.Uid), Type: ContainerDied, Data: event.ContainerId})
		klog.V(4).InfoS("Received Container Stopped Event", "event", event)
	case runtimeapi.ContainerEventType_CONTAINER_CREATED_EVENT:
		// We only need to update the pod status on container create.
		// But we don't have to generate any PodLifeCycleEvent. Container creation related
		// PodLifeCycleEvent is ignored by the existing Generic PLEG as well.
		// https://github.com/kubernetes/kubernetes/blob/24753aa8a4df8d10bfd6330e0f29186000c018be/pkg/kubelet/pleg/generic.go#L88 and
		// https://github.com/kubernetes/kubernetes/blob/24753aa8a4df8d10bfd6330e0f29186000c018be/pkg/kubelet/pleg/generic.go#L273
		klog.V(4).InfoS("Received Container Created Event", "event", event)
	case runtimeapi.ContainerEventType_CONTAINER_STARTED_EVENT:
		e.sendPodLifecycleEvent(&PodLifecycleEvent{ID: types.UID(event.PodSandboxMetadata.Uid), Type: ContainerStarted, Data: event.ContainerId})
		klog.V(4).InfoS("Received Container Started Event", "event", event)
	case runtimeapi.ContainerEventType_CONTAINER_DELETED_EVENT:
		// In case the pod is deleted it is safe to generate both ContainerDied and ContainerRemoved events, just like in the case of
		// Generic PLEG. https://github.com/kubernetes/kubernetes/blob/24753aa8a4df8d10bfd6330e0f29186000c018be/pkg/kubelet/pleg/generic.go#L169
		e.sendPodLifecycleEvent(&PodLifecycleEvent{ID: types.UID(event.PodSandboxMetadata.Uid), Type: ContainerDied, Data: event.ContainerId})
		e.sendPodLifecycleEvent(&PodLifecycleEvent{ID: types.UID(event.PodSandboxMetadata.Uid), Type: ContainerRemoved, Data: event.ContainerId})
		klog.V(4).InfoS("Received Container Deleted Event", "event", event)
	}
}

func (e *EventedPLEG) getPodIPs(pid types.UID, status *kubecontainer.PodStatus) []string {
	if len(status.IPs) != 0 {
		return status.IPs
	}

	oldStatus, err := e.cache.Get(pid)
	if err != nil || len(oldStatus.IPs) == 0 {
		return nil
	}

	for _, sandboxStatus := range status.SandboxStatuses {
		// If at least one sandbox is ready, then use this status update's pod IP
		if sandboxStatus.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			return status.IPs
		}
	}

	// For pods with no ready containers or sandboxes (like exited pods)
	// use the old status' pod IP
	return oldStatus.IPs
}

func (e *EventedPLEG) sendPodLifecycleEvent(event *PodLifecycleEvent) {
	select {
	case e.eventChannel <- event:
	default:
		// record how many events were discarded due to channel out of capacity
		metrics.PLEGDiscardEvents.Inc()
		klog.ErrorS(nil, "Evented PLEG: Event channel is full, discarded pod lifecycle event")
	}
}

func getPodSandboxState(podStatus *kubecontainer.PodStatus) kubecontainer.State {
	// increase running pod count when cache doesn't contain podID
	var sandboxId string
	for _, sandbox := range podStatus.SandboxStatuses {
		sandboxId = sandbox.Id
		// pod must contain only one sandbox
		break
	}

	for _, containerStatus := range podStatus.ContainerStatuses {
		if containerStatus.ID.ID == sandboxId {
			if containerStatus.State == kubecontainer.ContainerStateRunning {
				return containerStatus.State
			}
		}
	}
	return kubecontainer.ContainerStateExited
}

func (e *EventedPLEG) updateRunningPodMetric(podStatus *kubecontainer.PodStatus) {
	cachedPodStatus, err := e.cache.Get(podStatus.ID)
	if err != nil {
		klog.ErrorS(err, "Evented PLEG: Get cache", "podID", podStatus.ID)
	}
	// cache miss condition: The pod status object will have empty state if missed in cache
	if len(cachedPodStatus.SandboxStatuses) < 1 {
		sandboxState := getPodSandboxState(podStatus)
		if sandboxState == kubecontainer.ContainerStateRunning {
			metrics.RunningPodCount.Inc()
		}
	} else {
		oldSandboxState := getPodSandboxState(cachedPodStatus)
		currentSandboxState := getPodSandboxState(podStatus)

		if oldSandboxState == kubecontainer.ContainerStateRunning && currentSandboxState != kubecontainer.ContainerStateRunning {
			metrics.RunningPodCount.Dec()
		} else if oldSandboxState != kubecontainer.ContainerStateRunning && currentSandboxState == kubecontainer.ContainerStateRunning {
			metrics.RunningPodCount.Inc()
		}
	}
}

func getContainerStateCount(podStatus *kubecontainer.PodStatus) map[kubecontainer.State]int {
	containerStateCount := make(map[kubecontainer.State]int)
	for _, container := range podStatus.ContainerStatuses {
		containerStateCount[container.State]++
	}
	return containerStateCount
}

func (e *EventedPLEG) updateRunningContainerMetric(podStatus *kubecontainer.PodStatus) {
	cachedPodStatus, err := e.cache.Get(podStatus.ID)
	if err != nil {
		klog.ErrorS(err, "Evented PLEG: Get cache", "podID", podStatus.ID)
	}

	// cache miss condition: The pod status object will have empty state if missed in cache
	if len(cachedPodStatus.SandboxStatuses) < 1 {
		containerStateCount := getContainerStateCount(podStatus)
		for state, count := range containerStateCount {
			// add currently obtained count
			metrics.RunningContainerCount.WithLabelValues(string(state)).Add(float64(count))
		}
	} else {
		oldContainerStateCount := getContainerStateCount(cachedPodStatus)
		currentContainerStateCount := getContainerStateCount(podStatus)

		// old and new set of container states may vary;
		// get a unique set of container states combining both
		containerStates := make(map[kubecontainer.State]bool)
		for state := range oldContainerStateCount {
			containerStates[state] = true
		}
		for state := range currentContainerStateCount {
			containerStates[state] = true
		}

		// update the metric via difference of old and current counts
		for state := range containerStates {
			diff := currentContainerStateCount[state] - oldContainerStateCount[state]
			metrics.RunningContainerCount.WithLabelValues(string(state)).Add(float64(diff))
		}
	}
}
