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
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
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
	Start(ctx context.Context)
	Watch() chan *PodLifecycleEvent
	Healthy() (bool, error)
	// RequestReinspect flags the pod for reinspection on the next Relist iteration.
	RequestReinspect(podUID types.UID)
	// RequestRelist queues up the pod for an on-demand relist.
	RequestRelist(logger klog.Logger, podUID types.UID)
	// SuspendRelist holds back relist observations for the pod while it is being synced.
	// Any relist (on-demand or global) that would observe a change for the pod is deferred
	// until ResumeRelist is called.
	SuspendRelist(podUID types.UID)
	// ResumeRelist ends the suspension started by SuspendRelist. If any relist was
	// deferred during the suspension and the sync did not change the pod (changed == false),
	// a single on-demand relist is requested. If the sync did change the pod, the caller is
	// expected to request its own relist, which will observe the deferred changes as well.
	ResumeRelist(logger klog.Logger, podUID types.UID, changed bool)
}
