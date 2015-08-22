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
	"k8s.io/kubernetes/pkg/types"
)

type PodLifeCycleEventType string

const (
	ContainerStarted      PodLifeCycleEventType = "ContainerStarted"
	ContainerStopped      PodLifeCycleEventType = "ContainerStopped"
	NetworkSetupCompleted PodLifeCycleEventType = "NetworkSetupCompleted"
	NetworkFailed         PodLifeCycleEventType = "NetworkFailed"
	// PodSync is used to trigger syncing of a pod when the observed change of
	// the state of the pod cannot be captured by any single event above. E.g.,
	// during the bootstrapping phase, sending a PodSync event to each pod
	// ensures that all pods are sync'd once.
	PodSync PodLifeCycleEventType = "PodSync"
)

// PodLifecycleEvent is an event reflects the change of the pod state.
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

type PodLifecycleEventGenerator interface {
	Start()
	Relist()
	Watch() chan *PodLifecycleEvent
}
