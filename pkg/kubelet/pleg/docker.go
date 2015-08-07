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
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// For testability.
type DockerContainerGetter interface {
	GetRunningContainers() ([]*kubecontainer.Container, error)
	GetTerminatedContainers() ([]*kubecontainer.Container, error)
	GetAllContainers() ([]*kubecontainer.Container, error)
	ExamineContainer(dockerID string) (*dockertools.ContainerExaminationResult, error)
}

// DockerPLEG implements PodLifecycleEventGenerator for the Docker container
// runtime. When receiving an upstream container event, it inspects the
// container and generate a corresponding pod event. The inspection is
// necessary for retrieving information such as pod ID, which may not be
// provided by the upstream.
//
// DockerPLEG may miss a container event if *both* of the following
// requirements are met:
//   1. The container event stream misses an event.
//   2. The specific container was removed (i.e., garbage collected) before
//      next relisting.
// Genernally, (2) should not happen with kubelet's GC policy, unless user
// employs an external garbage collector. In the case where DockerPLEG is unable
// to inspect a container, the event'd would be skipped until the next relist.
//
// After being restarted, DockerPLEG may report outdated container events
// from the first relisting, due to the loss of internal states. Subscribers
// are expected to handle these irrelevant events.
type DockerPLEG struct {
	// The period for relisting.
	relistPeriod time.Duration
	// The timestamp of the last relist.
	relistTimestamp time.Time
	// The upstream container event watcher.
	upstreamWatcher ContainerEventWatcher
	upstreamCh      <-chan *ContainerEvent
	// The channel for the subscriber to receive.
	// TODO: Support multiple subscribers.
	eventChannel chan *PodLifecycleEvent
	// The underlying container runtime.
	// TODO(yujuhong): Replace this with kubecontainer.Runtime, or define a new
	// interface.
	runtime DockerContainerGetter
	// The set of containers (IDs) that we know are alive.
	aliveContainers sets.String
	// The set of containers (IDs) that we know are dead.
	deadContainers sets.String
}

var _ PodLifecycleEventGenerator = &DockerPLEG{}

func NewDockerPLEG(upstreamWatcher ContainerEventWatcher, runtime DockerContainerGetter, channelCapacity int,
	relistPeriod time.Duration) *DockerPLEG {
	return &DockerPLEG{
		relistPeriod:    relistPeriod,
		upstreamWatcher: upstreamWatcher,
		runtime:         runtime,
		eventChannel:    make(chan *PodLifecycleEvent, channelCapacity),
		aliveContainers: sets.NewString(),
		deadContainers:  sets.NewString(),
	}
}

const (
	// Time between retries if the container event stream channel has been
	// closed.
	retryInterval = time.Second * 3
)

// Returns a channel from which the subscriber can recieve PodLifecycleEvent
// events.
func (d *DockerPLEG) Watch() chan *PodLifecycleEvent {
	return d.eventChannel
}

// Instruct DockerPLEG to start watch upstrem for changes (and generate pod
// lifcycle events to the downstream channel).
func (d *DockerPLEG) Start() {
	go util.Until(d.doWork, retryInterval, util.NeverStop)
	glog.V(3).Infof("DockerPLEG: Started.")
}

// startWatchingUpstream starts watching upstream for container changes, and
// sets the upstreamCh channel.
func (d *DockerPLEG) startWatchingUpstream() error {
	// TODO(yujuhong): Make sure there is enough channel capacity to buffer
	// the events, or we should buffer it internally.
	ch, err := d.upstreamWatcher.Watch()
	if err != nil {
		return err
	}
	d.upstreamCh = ch
	return nil
}

func (d *DockerPLEG) doWork() {
	// We need to start watching the upstream before relisting to ensure
	// that we don't miss any event. Duplicated events will be filtered out and
	// ignored.
	if err := d.startWatchingUpstream(); err != nil {
		glog.Errorf("Unable to watch upstream %v: %v", d.upstreamWatcher, err)
	}

	d.relist()
	for {
		select {
		case e, ok := <-d.upstreamCh:
			if !ok {
				glog.Errorf("DockerPLEG: Upstream channel closed")
				return
			}
			glog.V(3).Infof("DockerPLEG: Received an event from upstream: %+v", e)
			if e.Timestamp.Before(d.relistTimestamp) {
				// Any event that is older than the last relist timestamp
				// is considered outdated, and should be discarded
				glog.V(3).Infof("DockerPLEG: Discarding outdated event %+v", e)
				break
			}
			d.processEvent(e)
		case <-time.After(d.relistPeriod):
			glog.V(3).Infof("DockerPLEG: Relisting")
			d.relist()
		}
	}
}

func (d *DockerPLEG) processEvent(e *ContainerEvent) {
	switch e.Type {
	case ContainerEventStarted:
		d.handleContainerStarted(e)
	case ContainerEventStopped:
		d.handleContainerStopped(e)
	default:
		glog.Errorf("DockerPLEG: Unknown event: %+v", e)
	}
}

// relist scans the docker containers to discover missing container events.
func (d *DockerPLEG) relist() {
	oldAlive := d.aliveContainers
	oldDead := d.deadContainers
	// Set the relist timestamp.
	d.relistTimestamp = time.Now()
	alive, dead, err := d.getAliveAndDeadConainerSets()
	if err != nil {
		glog.Errorf("DockerPLEG: Unable to get pods from the container runtime: %v", err)
		return
	}
	// A set of dead containers whose existence we weren't aware of prior to
	// relist. This means that we may have missed both the creation and deletion
	// events of a container. We'd send out both creation and deletion events for
	// them.
	missed := dead.Difference(oldDead).Difference(oldAlive)

	// Generate corresponding container events, which will be treated the same
	// way as the events from upstream. Note that the internal alive/dead
	// container sets will be modified accordingly when processing the events.
	started := alive.Difference(oldAlive)
	stopped := oldAlive.Difference(alive)
	if started.Len() != 0 || stopped.Len() != 0 || missed.Len() != 0 {
		glog.V(2).Infof("DockerPLEG: Discovered missing events; started: %v, stopped: %v, missed: %v",
			started.List(), stopped.List(), missed.List())
	}
	for _, c := range started.Union(missed).List() {
		d.processEvent(&ContainerEvent{
			ID:        c,
			Timestamp: d.relistTimestamp,
			Type:      ContainerEventStarted,
		})
	}
	for _, c := range stopped.Union(missed).List() {
		d.processEvent(&ContainerEvent{
			ID:        c,
			Timestamp: d.relistTimestamp,
			Type:      ContainerEventStopped,
		})
	}
}

func (d *DockerPLEG) getAliveAndDeadConainerSets() (sets.String, sets.String, error) {
	running, err := d.runtime.GetRunningContainers()
	if err != nil {
		return nil, nil, err
	}
	dead, err := d.runtime.GetTerminatedContainers()
	if err != nil {
		return nil, nil, err
	}
	return containerListToSet(running), containerListToSet(dead), nil
}

func (d *DockerPLEG) handleContainerStarted(e *ContainerEvent) {
	if d.aliveContainers.Has(e.ID) && !d.deadContainers.Has(e.ID) {
		// TODO(yujuhong): Why would we see duplicated events?
		glog.Warningf("DockerPLEG: Received duplicated event: %#v", e)
		return
	}
	// We need to derive some information from the container ID: pod ID,
	// whether the container is a network container.
	result, err := d.runtime.ExamineContainer(e.ID)
	if err != nil {
		// We haven't updated the internal container sets yet. If this
		// container is observed in the next relisting, we'd try inspecting
		// again.
		glog.Errorf("DockerPLEG: Unable to examine container %q: %v", e.ID, err)
		return
	}
	// Update internal storage.
	d.aliveContainers.Insert(e.ID)
	d.deadContainers.Insert(e.ID)

	pod := result.Pod
	container := result.Pod.Containers[0]
	if result.IsInfraContainer {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: NetworkSetupCompleted}
	} else {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: ContainerStarted, Data: container.Name}
	}
}

func (d *DockerPLEG) handleContainerStopped(e *ContainerEvent) {
	if !d.aliveContainers.Has(e.ID) && d.deadContainers.Has(e.ID) {
		// TODO(yjhong): Why would we see duplicated events?
		glog.V(4).Infof("DockerPLEG: Received duplicated event: %#v", e)
		return
	}
	// We need to derive some information from the container ID: pod ID,
	// whether the container is a network container.
	result, err := d.runtime.ExamineContainer(e.ID)
	if err != nil {
		// We haven't updated the internal container sets yet. If this
		// container is observed in the next relisting, we'd try inspecting
		// again.
		glog.Errorf("DockerPLEG: Unable to examine container %q: %v", e.ID, err)
		return
	}
	// Update internal storage.
	d.aliveContainers.Delete(e.ID)
	d.deadContainers.Insert(e.ID)

	pod := result.Pod
	container := result.Pod.Containers[0]
	if result.IsInfraContainer {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: NetworkFailed}
	} else {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: ContainerStopped, Data: container.Name}
	}
}

// containerListToSet converts a list of containers to a set of container IDs
// in strings.
func containerListToSet(containers []*kubecontainer.Container) sets.String {
	cset := sets.NewString()
	for _, c := range containers {
		cset.Insert(string(c.ID))
	}
	return cset
}
