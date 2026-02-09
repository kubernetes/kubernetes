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
	"time"

	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// PodLifeCycleEventType define the event type of pod life cycle events.
type PodLifeCycleEventType string

type RelistDuration struct {
	// The period for relisting.
	RelistPeriod time.Duration
	// The relisting threshold needs to be greater than the relisting period +
	// the relisting time, which can vary significantly. Set a conservative
	// threshold to avoid flipping between healthy and unhealthy.
	RelistThreshold time.Duration
}

const (
	// ContainerStarted - event type when the new state of container is running.
	ContainerStarted PodLifeCycleEventType = "ContainerStarted"
	// ContainerDied - event type when the new state of container is exited.
	ContainerDied PodLifeCycleEventType = "ContainerDied"
	// ContainerRemoved - event type when the old state of container is exited.
	ContainerRemoved PodLifeCycleEventType = "ContainerRemoved"
	// PodSync is used to trigger syncing of a pod when the observed change of
	// the state of the pod cannot be captured by any single event above.
	PodSync PodLifeCycleEventType = "PodSync"
	// ContainerChanged - event type when the new state of container is unknown.
	ContainerChanged PodLifeCycleEventType = "ContainerChanged"
	// ConditionMet - event type triggered when any number of watch conditions are met.
	ConditionMet PodLifeCycleEventType = "ConditionMet"
)

// PodLifecycleEvent is an event that reflects the change of the pod state.
type PodLifecycleEvent struct {
	// The pod ID.
	ID types.UID
	// The type of the event.
	Type PodLifeCycleEventType
	// The accompanied data which varies based on the event type.
	//   - ContainerStarted/ContainerStopped: the container name (string).
	//   - All other event types: unused.
	Data interface{}
}

// PodLifecycleEventGenerator contains functions for generating pod life cycle events.
type PodLifecycleEventGenerator interface {
	Start()
	Watch() chan *PodLifecycleEvent
	Healthy() (bool, error)
	// SetPodWatchCondition flags the pod for reinspection on every Relist iteration until the watch
	// condition is met. The condition is keyed so it can be updated before the condition
	// is met.
	SetPodWatchCondition(podUID types.UID, conditionKey string, condition WatchCondition)
}

// podLifecycleEventGeneratorHandler contains functions that are useful for different PLEGs
// and need not be exposed to rest of the kubelet
type podLifecycleEventGeneratorHandler interface {
	PodLifecycleEventGenerator
	Stop()
	Update(relistDuration *RelistDuration)
	Relist()
}

// WatchCondition takes the latest PodStatus, and returns whether the condition is met.
type WatchCondition = func(*kubecontainer.PodStatus) bool

// RunningContainerWatchCondition wraps a condition on the container status to make a pod
// WatchCondition. If the container is no longer running, the condition is implicitly cleared.
func RunningContainerWatchCondition(containerName string, condition func(*kubecontainer.Status) bool) WatchCondition {
	return func(podStatus *kubecontainer.PodStatus) bool {
		status := podStatus.FindContainerStatusByName(containerName)
		if status == nil || status.State != kubecontainer.ContainerStateRunning {
			// Container isn't running. Consider the condition "completed" so it is cleared.
			return true
		}
		return condition(status)
	}
}
