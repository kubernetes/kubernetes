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

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
)

// dockerEventWatcher watches event stream from docker, and translates the
// events into ContainerEvent events.
type dockerEventWatcher struct {
	client dockertools.DockerInterface
	source chan *docker.APIEvents
	sink   chan *ContainerEvent
}

var _ ContainerEventWatcher = &dockerEventWatcher{}

const (
	dockerContainerStartStatus = "start"
	dockerContainerDieStatus   = "die"
	// Pick an arbitrary number that is considered large enough.
	dockerSourceChannelCapacity = 1000
	dockerSinkChannelCapacity   = 1000
)

func NewDockerEventWatcher(client dockertools.DockerInterface) *dockerEventWatcher {
	return &dockerEventWatcher{client: client}
}

func (d *dockerEventWatcher) doWork() {
	d.source = make(chan *docker.APIEvents, dockerSourceChannelCapacity)
	defer close(d.source)
	if err := d.client.AddEventListener(d.source); err != nil {
		glog.Errorf("dockerEventWatcher: Unable to watch docker events: %v", err)
		return
	}
	for event := range d.source {
		glog.Infof("dockerEventWatcher: Receiving event %#v", event)
		switch event.Status {
		case dockerContainerStartStatus:
			d.sink <- &ContainerEvent{
				ID:        event.ID,
				Timestamp: time.Unix(event.Time, 0),
				Type:      ContainerEventStarted,
			}
		case dockerContainerDieStatus:
			d.sink <- &ContainerEvent{
				ID:        event.ID,
				Timestamp: time.Unix(event.Time, 0),
				Type:      ContainerEventStopped,
			}
		default:
			break
		}
	}
}

func (d *dockerEventWatcher) Watch() (<-chan *ContainerEvent, error) {
	d.sink = make(chan *ContainerEvent, dockerSinkChannelCapacity)
	// Launch a goroutine to adapt the events from the upstream docker
	// channel to ContainerEvent events.
	go util.Until(d.doWork, 0, util.NeverStop)
	return d.sink, nil
}

func (d *dockerEventWatcher) Stop() {
	d.client.RemoveEventListener(d.source)
	close(d.source)
	close(d.sink)
}
