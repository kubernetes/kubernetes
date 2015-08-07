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
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/events"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

type ContainerEvent struct {
	ID        string
	Timestamp time.Time
	Type      ContainerEventType
}

type ContainerEventType string

const (
	ContainerEventStarted = "STARTED"
	ContainerEventStopped = "STOPPED"
)

type ContainerEventWatcher interface {
	Watch() (<-chan *ContainerEvent, error)
	Stop()
}

// cadvisorEventWatcher watches event stream from cadvisor, and translate the
// events into ContainerEvent events.
type cadvisorEventWatcher struct {
	client cadvisor.Interface
	source *events.EventChannel
	sink   chan *ContainerEvent
}

var _ ContainerEventWatcher = &cadvisorEventWatcher{}

const (
	// Pick an arbitrary number that is considered large enough.
	cadvisorSinkChannelCapacity = 1000
)

func NewCadvisorEventWatcher(client cadvisor.Interface) *cadvisorEventWatcher {
	return &cadvisorEventWatcher{client: client}
}

// getContainerID extract the container ID fro the container path.
// E.g., "/abcd123" to "abcd123".
func getContainerID(path string) string {
	chunks := strings.Split(path, "/")
	return chunks[len(chunks)-1]
}

func (c *cadvisorEventWatcher) Watch() (<-chan *ContainerEvent, error) {
	// Only watch the creation/deletion events.
	source, err := c.client.WatchEvents(&events.Request{
		EventType: map[cadvisorApi.EventType]bool{
			cadvisorApi.EventContainerCreation: true,
			cadvisorApi.EventContainerDeletion: true,
		}})
	if err != nil {
		glog.Errorf("cadvisorEventWatcher: Unable to watch cadvisor events: %v", err)
		return nil, err
	}
	c.source = source
	c.sink = make(chan *ContainerEvent, cadvisorSinkChannelCapacity)
	// Launch a goroutine to adapt the events from the upstream cadvisor
	// channel to ContainerEvent events.
	go func() {
		defer c.Stop()
		for event := range c.source.GetChannel() {
			glog.Infof("cadvisorEventWatcher: Receiving event %#v", event)
			switch event.EventType {
			case cadvisorApi.EventContainerCreation:
				c.sink <- &ContainerEvent{
					ID:        getContainerID(event.ContainerName),
					Timestamp: event.Timestamp,
					Type:      ContainerEventStarted,
				}
			case cadvisorApi.EventContainerDeletion:
				c.sink <- &ContainerEvent{
					ID:        getContainerID(event.ContainerName),
					Timestamp: event.Timestamp,
					Type:      ContainerEventStopped,
				}
			default:
				break
			}
		}
	}()
	return c.sink, nil
}

func (c *cadvisorEventWatcher) Stop() {
	// Tell cadvisor to close the channel.
	c.client.StopWatch(c.source.GetWatchId())
	// Close our own result channel.
	close(c.sink)
}
