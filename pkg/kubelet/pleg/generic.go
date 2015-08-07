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
	"time"

	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// GenericPLEG is an extremely simple generic PLEG that relies solely on
// periodic listing to discover container changes. It should be be used
// as temporary replacement or container runtimes do not support a proper
// event generator yet.
//
// Note that Generic PLEG assumes that a container would not be created and
// garbage collected within one relist period. If this is not true, it might
// miss the container completely. Relisting failure would also lead to a longer
// window where a container could be missing. It is recommended to set the
// relist period short and have an auxiliary, longer periodic sync in kubelet
// the safety net.
type GenericPLEG struct {
	// The period for relisting.
	relistPeriod time.Duration
	runtime      kubecontainer.Runtime
	eventChannel chan *PodLifecycleEvent
	// IDs of all visible containers.
	containers sets.String
	// IDs of all running containers
	runningContainers sets.String
	// Map container ID to pod ID.
	idMap map[string]types.UID
}

var _ PodLifecycleEventGenerator = &GenericPLEG{}

func NewGenericPLEG(runtime kubecontainer.Runtime, channelCapacity int,
	relistPeriod time.Duration) *GenericPLEG {
	return &GenericPLEG{
		relistPeriod:      relistPeriod,
		runtime:           runtime,
		eventChannel:      make(chan *PodLifecycleEvent, channelCapacity),
		idMap:             make(map[string]types.UID),
		containers:        sets.NewString(),
		runningContainers: sets.NewString(),
	}
}

// Returns a channel from which the subscriber can recieve PodLifecycleEvent
// events.
func (g *GenericPLEG) Watch() chan *PodLifecycleEvent {
	return g.eventChannel
}

func (g *GenericPLEG) doWork() {
	select {
	case <-time.After(g.relistPeriod):
		glog.V(3).Infof("GenericPLEG: Relisting")
		g.relist()
	}
}

// Start spawns a goroutine to relist periodically.
func (g *GenericPLEG) Start() {
	g.relist()
	go util.Until(g.doWork, 0, util.NeverStop)
}

func buildPodIDSet(containerIDs []string, idMap map[string]types.UID) sets.String {
	pids := sets.NewString()
	for _, cid := range containerIDs {
		if pid, ok := idMap[cid]; ok {
			pids.Insert(string(pid))
		}
	}
	return pids
}

// relist relists and sends out PodSync events for pods that have changed.
// Changes may include container start/deletion, etc.
func (g *GenericPLEG) relist() {
	// We ask the runtime twice in order to distinguish between running and
	// non-running containers. We may want to augment the runtime interface to
	// make this easier.
	runningPods, err := g.runtime.GetPods(false)
	if err != nil {
		glog.Errorf("GenericPLEG: Unable to retrieve pods: %v", err)
		return
	}
	pods, err := g.runtime.GetPods(true)
	if err != nil {
		glog.Errorf("GenericPLEG: Unable to retrieve pods: %v", err)
		return
	}

	runningContainers := buildContainerSet(runningPods)
	containers := buildContainerSet(pods)
	idMap := buildIDMap(append(pods, runningPods...))
	toSync := sets.NewString()

	// Newly observed running containers.
	c := runningContainers.Difference(g.runningContainers).List()
	toSync = toSync.Union(buildPodIDSet(c, idMap))
	if len(c) != 0 {
		glog.V(4).Infof("GenericPLEG: Discovered new running containers: %v", c)
	}
	// Containers that were running before, but are no longer running, i.e.,
	// containers that recently died.
	c = g.runningContainers.Difference(runningContainers).List()
	toSync = toSync.Union(buildPodIDSet(c, g.idMap))
	if len(c) != 0 {
		glog.V(4).Infof("GenericPLEG: Discovered containers that became non-running: %v", c)
	}
	// Newly observed non-running containers. This may include recently died
	// containers, and newly created (but not yet running) containers, etc.
	// What we really want is to capture containers that were created and died
	// (but not yet removed) after the last relist.
	c = containers.Difference(g.containers).List()
	toSync = toSync.Union(buildPodIDSet(c, idMap))
	if len(c) != 0 {
		glog.V(4).Infof("GenericPLEG: Discovered new non-running containers: %v", c)
	}

	// Update the internal storage.
	g.runningContainers = runningContainers
	g.containers = containers
	g.idMap = idMap
	// Send out sync events.
	for _, pid := range toSync.List() {
		g.eventChannel <- &PodLifecycleEvent{ID: types.UID(pid), Type: PodSync}
	}
}
