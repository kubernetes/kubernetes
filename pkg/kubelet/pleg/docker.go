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
)

// DockerPLEG implements PodLifecycleEventGenerator for the Docker container
// runtime. When receiving an upstream container event, it inspects the
// container and generate corresponding pod event.
//
// It should be noted that the container state may have changed after receving
// a container event, but before DockerPLEG inspects the container. However,
// the separate inspection is needed because the upstream may not provide all
// necessary information (e.g., pod ID). The subscribers are expected to deal
// with such inconsistency.
type DockerPLEG struct {
	// The period for relisting.
	relistPeriod time.Duration
	// The timestamp of the last relist.
	relistTimestamp time.Time
	// The upstream container event watcher.
	upstream ContainerEventWatcher
	// The channel for the subscriber to receive.
	// TODO: Support multiple subscribers.
	eventChannel chan *PodLifecycleEvent
	// The underlying container runtime.
	// TODO(yujuhong): Replace this with kubecontainer.Runtime, or define a new
	// interface.
	runtime *dockertools.DockerManager
	// The set of containers (IDs) that we know are alive.
	containerSet util.StringSet
}

var _ PodLifecycleEventGenerator = &DockerPLEG{}

func NewDockerPLEG(upstream ContainerEventWatcher, runtime *dockertools.DockerManager, channelCapacity int,
	relistPeriod time.Duration) *DockerPLEG {
	return &DockerPLEG{
		relistPeriod: relistPeriod,
		upstream:     upstream,
		runtime:      runtime,
		eventChannel: make(chan *PodLifecycleEvent, channelCapacity),
		containerSet: util.NewStringSet(),
	}
}

// Returns a channel from which the subscriber can recieve PodLifecycleEvent
// events.
func (d *DockerPLEG) Watch() chan *PodLifecycleEvent {
	return d.eventChannel
}

// Instruct DockerPLEG to start watch upstrem for changes (and generate pod
// lifcycle events to the downstream channel).
func (d *DockerPLEG) Start() {
	// We need to start watching the upstream before relisting to ensure
	// that we don't miss events. However, we should not start processing
	// those events until relist is completed so that we can deduplicate
	// events.

	// Start watching upstream for container changes.
	// TODO(yujuhong): Make sure there is enough channel capacity to buffer
	// the events, or we should buffer it internally.
	ch, err := d.upstream.Watch()
	if err != nil {
		// TODO(yujuhong): Should we retry?
		glog.Errorf("Unable to watch upstream %v: %v", d.upstream, err)

		// Create a dummy channel with no sender. We will rely on relisting to
		// generate the events.
		ch = make(chan *ContainerEvent, 1)
	}

	d.bootstrap()
	glog.V(3).Infof("DockerPLEG: Bootstrapping completed.")
	// Start a goroutine to process events.
	go func() {
		for {
			select {
			case e, ok := <-ch:
				if !ok {
					glog.Infof("DockerPLEG: Upstream channel closed")
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
				d.Relist()
			}
		}
	}()
}

func (d *DockerPLEG) processEvent(e *ContainerEvent) {
	switch e.Type {
	case ContainerEventStarted:
		d.handleContainerStarted(e)
	case ContainerEventStopped:
		d.handleContainerStopped(e)
	default:
		glog.Errorf("Unknown event: %+v", e)
	}
}

func (d *DockerPLEG) Relist() {
	oldSet := d.containerSet
	// Set the relist timestamp.
	d.relistTimestamp = time.Now()
	pods, err := d.runtime.GetPods(false)
	if err != nil {
		// wait until the next period
		return
	}
	newSet := buildContainerSet(pods)
	started := newSet.Difference(oldSet)
	stopped := oldSet.Difference(newSet)
	// Generate corresponding container events, which will be treated the same
	// way as the events from upstream.
	for _, c := range started.List() {
		d.processEvent(&ContainerEvent{
			ID:        c,
			Timestamp: d.relistTimestamp,
			Type:      ContainerEventStarted,
		})
	}
	for _, c := range stopped.List() {
		d.processEvent(&ContainerEvent{
			ID:        c,
			Timestamp: d.relistTimestamp,
			Type:      ContainerEventStopped,
		})
	}
}

// bootstrap relists and sends out PodSync events for each pod.
func (d *DockerPLEG) bootstrap() {
	// We call GetPods() directly (instead of Relist()) because all we want to
	// skip regular container event processing, and just sends out a PodSync
	// event for each pod.
	d.relistTimestamp = time.Now()
	pods, err := d.runtime.GetPods(false)
	if err != nil {
		// TODO(yujuhong): should we retry?
		glog.Errorf("Unable to get pods from the container runtime: %v", err)
	}
	d.containerSet = buildContainerSet(pods)
	for _, pod := range pods {
		// TODO(yujuhong): Insert some time to interleave the pod workers?
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: PodSync}
	}
}

func (d *DockerPLEG) handleContainerStarted(e *ContainerEvent) {
	if d.containerSet.Has(e.ID) {
		// TODO(yujuhong): Why would we see duplicated events?
		glog.Warningf("Received duplicated event: %#v", e)
		return
	}
	d.containerSet.Insert(e.ID)

	// We need to derive some information from the container ID: pod ID,
	// whether the container is a network container.
	result, err := d.runtime.ExamineContainer(e.ID)
	if err != nil {
		glog.Errorf("DockerPLEG: Unable to examine container %q: %v", e.ID, err)
		return
	}
	pod := result.Pod
	container := result.Pod.Containers[0]
	if result.IsInfraContainer {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: NetworkSetupCompleted}
	} else {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: ContainerStarted, Data: container.Name}
	}
}

func (d *DockerPLEG) handleContainerStopped(e *ContainerEvent) {
	if !d.containerSet.Has(e.ID) {
		// TODO(yjhong): Why would we see duplicated events?
		glog.Warningf("Received duplicated event: %#v", e)
		return
	}
	d.containerSet.Delete(e.ID)

	// We need to derive some information from the container ID: pod ID,
	// whether the container is a network container.
	result, err := d.runtime.ExamineContainer(e.ID)
	if err != nil {
		glog.Errorf("Unable to examine container %q: %v", e.ID, err)
		return
	}
	pod := result.Pod
	container := result.Pod.Containers[0]
	if result.IsInfraContainer {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: NetworkFailed}
	} else {
		d.eventChannel <- &PodLifecycleEvent{ID: pod.ID, Type: ContainerStopped, Data: container.Name}
	}
}

func buildContainerSet(pods []*kubecontainer.Pod) util.StringSet {
	cset := util.NewStringSet()
	for _, p := range pods {
		for _, c := range p.Containers {
			cset.Insert(string(c.ID))
		}
	}
	return cset
}
