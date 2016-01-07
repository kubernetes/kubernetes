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
	"k8s.io/kubernetes/pkg/kubelet/metrics"
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
	// The internal cache for pod/container information.
	podRecords podRecords
	// Time of the last relisting.
	lastRelistTime time.Time
}

// plegContainerState has an one-to-one mapping to the
// kubecontainer.ContainerState except for the Non-existent state. This state
// is introduced here to complete the state transition scenarios.
type plegContainerState string

const (
	plegContainerRunning     plegContainerState = "running"
	plegContainerExited      plegContainerState = "exited"
	plegContainerUnknown     plegContainerState = "unknown"
	plegContainerNonExistent plegContainerState = "non-existent"
)

func convertState(state kubecontainer.ContainerState) plegContainerState {
	switch state {
	case kubecontainer.ContainerStateRunning:
		return plegContainerRunning
	case kubecontainer.ContainerStateExited:
		return plegContainerExited
	case kubecontainer.ContainerStateUnknown:
		return plegContainerUnknown
	default:
		panic(fmt.Sprintf("unrecognized container state: %v", state))
	}
}

type podRecord struct {
	old     *kubecontainer.Pod
	current *kubecontainer.Pod
}

type podRecords map[types.UID]*podRecord

func NewGenericPLEG(runtime kubecontainer.Runtime, channelCapacity int,
	relistPeriod time.Duration) PodLifecycleEventGenerator {
	return &GenericPLEG{
		relistPeriod: relistPeriod,
		runtime:      runtime,
		eventChannel: make(chan *PodLifecycleEvent, channelCapacity),
		podRecords:   make(podRecords),
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

func generateEvent(podID types.UID, cid string, oldState, newState plegContainerState) *PodLifecycleEvent {
	glog.V(7).Infof("GenericPLEG: %v/%v: %v -> %v", podID, cid, oldState, newState)
	if newState == oldState {
		return nil
	}
	switch newState {
	case plegContainerRunning:
		return &PodLifecycleEvent{ID: podID, Type: ContainerStarted, Data: cid}
	case plegContainerExited:
		return &PodLifecycleEvent{ID: podID, Type: ContainerDied, Data: cid}
	case plegContainerUnknown:
		// Don't generate any event if the status is unknown.
		return nil
	case plegContainerNonExistent:
		// We report "ContainerDied" when container was stopped OR removed. We
		// may want to distinguish the two cases in the future.
		switch oldState {
		case plegContainerExited:
			// We already reported that the container died before. There is no
			// need to do it again.
			return nil
		default:
			return &PodLifecycleEvent{ID: podID, Type: ContainerDied, Data: cid}
		}
	default:
		panic(fmt.Sprintf("unrecognized container state: %v", newState))
	}
	return nil
}

// relist queries the container runtime for list of pods/containers, compare
// with the internal pods/containers, and generats events accordingly.
func (g *GenericPLEG) relist() {
	glog.V(5).Infof("GenericPLEG: Relisting")
	timestamp := time.Now()

	if !g.lastRelistTime.IsZero() {
		metrics.PLEGRelistInterval.Observe(metrics.SinceInMicroseconds(g.lastRelistTime))
	}
	defer func() {
		// Update the relist time.
		g.lastRelistTime = timestamp
		metrics.PLEGRelistLatency.Observe(metrics.SinceInMicroseconds(timestamp))
	}()

	// Get all the pods.
	podList, err := g.runtime.GetPods(true)
	if err != nil {
		glog.Errorf("GenericPLEG: Unable to retrieve pods: %v", err)
		return
	}
	pods := kubecontainer.Pods(podList)

	eventsByPodID := map[types.UID][]*PodLifecycleEvent{}
	// Process all currently visible pods.
	for _, pod := range pods {
		g.podRecords.setCurrent(pod)
		// Locate the old pod.
		oldPod := g.podRecords.getOld(pod.ID)

		// Process all currently visible containers in the pod.
		for _, container := range pod.Containers {
			cid := container.ID
			oldState := getContainerState(oldPod, cid)
			newState := convertState(container.State)
			e := generateEvent(pod.ID, cid.ID, oldState, newState)
			updateEvents(eventsByPodID, e)
		}

		if oldPod == nil {
			continue
		}
		// Process all containers in the old pod, but no longer in the new pod.
		for _, oldContainer := range oldPod.Containers {
			cid := oldContainer.ID
			oldState := convertState(oldContainer.State)
			newState := getContainerState(pod, cid)
			if newState != plegContainerNonExistent {
				// We already processed the container.
				continue
			}
			// Container no longer visible, generate an event.
			e := generateEvent(pod.ID, cid.ID, oldState, plegContainerNonExistent)
			updateEvents(eventsByPodID, e)
		}
	}

	// Process all pods that are no longer visible.
	for pid := range g.podRecords {
		if pod := g.podRecords.getCurrent(pid); pod != nil {
			continue
		}
		oldPod := g.podRecords.getOld(pid)
		for _, oldContainer := range oldPod.Containers {
			cid := oldContainer.ID
			oldState := convertState(oldContainer.State)
			e := generateEvent(oldPod.ID, cid.ID, oldState, plegContainerNonExistent)
			updateEvents(eventsByPodID, e)
		}
	}

	// Update the internal storage.
	g.podRecords.updateAll()

	// Send out the events.
	for _, events := range eventsByPodID {
		for i := range events {
			g.eventChannel <- events[i]
		}
	}
}

func updateEvents(eventsByPodID map[types.UID][]*PodLifecycleEvent, e *PodLifecycleEvent) {
	if e == nil {
		return
	}
	eventsByPodID[e.ID] = append(eventsByPodID[e.ID], e)
}

func getContainerState(pod *kubecontainer.Pod, cid kubecontainer.ContainerID) plegContainerState {
	// Default to the non-existent state.
	state := plegContainerNonExistent
	if pod == nil {
		return state
	}
	container := pod.FindContainerByID(cid)
	if container == nil {
		return state
	}
	return convertState(container.State)
}

func (pr podRecords) getOld(id types.UID) *kubecontainer.Pod {
	r, ok := pr[id]
	if !ok {
		return nil
	}
	return r.old
}

func (pr podRecords) getCurrent(id types.UID) *kubecontainer.Pod {
	r, ok := pr[id]
	if !ok {
		return nil
	}
	return r.current
}

func (pr podRecords) setCurrent(pod *kubecontainer.Pod) {
	if r, ok := pr[pod.ID]; ok {
		r.current = pod
		return
	}
	pr[pod.ID] = &podRecord{current: pod}
}

func (pr podRecords) updateAll() {
	for k, r := range pr {
		if r.current == nil {
			// Pod no longer exists; delete the entry.
			delete(pr, k)
			continue
		}
		r.old = r.current
		r.current = nil
	}
}
