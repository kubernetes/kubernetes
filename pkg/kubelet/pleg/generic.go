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
	"fmt"
	"time"

	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// GenericPLEG is an extremely simple generic PLEG that relies solely on
// periodic listing to discover container changes. It should be be used
// as temporary replacement for container runtimes do not support a proper
// event generator yet.
//
// Note that GenericPLEG assumes that a container would not be created,
// terminated, and garbage collected within one relist period. If such an
// incident happens, GenenricPLEG would miss all events regarding this
// container. In the case of relisting failure, the window may become longer.
// Note that this assumption is not unique -- many kubelet internal components
// rely on terminated containers as tombstones for bookkeeping purposes. The
// garbage collector is implemented to work with such situtations. However, to
// guarantee that kubelet can handle missing container events, it is
// recommended to set the relist period short and have an auxiliary, longer
// periodic sync in kubelet as the safety net.
type GenericPLEG struct {
	// The period for relisting.
	relistPeriod time.Duration
	// The container runtime.
	runtime kubecontainer.Runtime
	// The channel from which the subscriber listens events.
	eventChannel chan *PodLifecycleEvent
	// The internal cache for container information.
	containers map[string]containerInfo
}

type containerInfo struct {
	podID types.UID
	state kubecontainer.ContainerState
}

func NewGenericPLEG(runtime kubecontainer.Runtime, channelCapacity int,
	relistPeriod time.Duration) PodLifecycleEventGenerator {
	return &GenericPLEG{
		relistPeriod: relistPeriod,
		runtime:      runtime,
		eventChannel: make(chan *PodLifecycleEvent, channelCapacity),
		containers:   make(map[string]containerInfo),
	}
}

// Returns a channel from which the subscriber can recieve PodLifecycleEvent
// events.
// TODO: support multiple subscribers.
func (g *GenericPLEG) Watch() chan *PodLifecycleEvent {
	return g.eventChannel
}

// Start spawns a goroutine to relist periodically.
func (g *GenericPLEG) Start() {
	go util.Until(g.relist, g.relistPeriod, util.NeverStop)
}

func generateEvent(podID types.UID, cid string, oldState, newState kubecontainer.ContainerState) *PodLifecycleEvent {
	if newState == oldState {
		return nil
	}
	switch newState {
	case kubecontainer.ContainerStateRunning:
		return &PodLifecycleEvent{ID: podID, Type: ContainerStarted, Data: cid}
	case kubecontainer.ContainerStateExited:
		return &PodLifecycleEvent{ID: podID, Type: ContainerDied, Data: cid}
	case kubecontainer.ContainerStateUnknown:
		// Don't generate any event if the status is unknown.
		return nil
	default:
		panic(fmt.Sprintf("unrecognized container state: %v", newState))
	}
	return nil
}

// relist queries the container runtime for list of pods/containers, compare
// with the internal pods/containers, and generats events accordingly.
func (g *GenericPLEG) relist() {
	glog.V(5).Infof("GenericPLEG: Relisting")
	// Get all the pods.
	pods, err := g.runtime.GetPods(true)
	if err != nil {
		glog.Errorf("GenericPLEG: Unable to retrieve pods: %v", err)
		return
	}

	events := []*PodLifecycleEvent{}
	containers := make(map[string]containerInfo, len(g.containers))
	// Create a new containers map, compares container statuses, and generates
	// correspoinding events.
	for _, p := range pods {
		for _, c := range p.Containers {
			cid := c.ID.ID
			// Get the of existing container info. Defaults to state unknown.
			oldState := kubecontainer.ContainerStateUnknown
			if info, ok := g.containers[cid]; ok {
				oldState = info.state
			}
			// Generate an event if required.
			glog.V(7).Infof("GenericPLEG: %v/%v: %v -> %v", p.ID, cid, oldState, c.State)
			if e := generateEvent(p.ID, cid, oldState, c.State); e != nil {
				events = append(events, e)
			}
			// Write to the new cache.
			containers[cid] = containerInfo{podID: p.ID, state: c.State}
		}
	}

	// Swap the container info cache. This is purely to avoid the need of
	// garbage collection.
	g.containers = containers

	// Send out the events.
	for i := range events {
		g.eventChannel <- events[i]
	}
}
