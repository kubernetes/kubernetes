// +build !solaris

package stats

import (
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/container"
	"github.com/docker/docker/pkg/pubsub"
	"github.com/sirupsen/logrus"
)

// Collect registers the container with the collector and adds it to
// the event loop for collection on the specified interval returning
// a channel for the subscriber to receive on.
func (s *Collector) Collect(c *container.Container) chan interface{} {
	s.m.Lock()
	defer s.m.Unlock()
	publisher, exists := s.publishers[c]
	if !exists {
		publisher = pubsub.NewPublisher(100*time.Millisecond, 1024)
		s.publishers[c] = publisher
	}
	return publisher.Subscribe()
}

// StopCollection closes the channels for all subscribers and removes
// the container from metrics collection.
func (s *Collector) StopCollection(c *container.Container) {
	s.m.Lock()
	if publisher, exists := s.publishers[c]; exists {
		publisher.Close()
		delete(s.publishers, c)
	}
	s.m.Unlock()
}

// Unsubscribe removes a specific subscriber from receiving updates for a container's stats.
func (s *Collector) Unsubscribe(c *container.Container, ch chan interface{}) {
	s.m.Lock()
	publisher := s.publishers[c]
	if publisher != nil {
		publisher.Evict(ch)
		if publisher.Len() == 0 {
			delete(s.publishers, c)
		}
	}
	s.m.Unlock()
}

// Run starts the collectors and will indefinitely collect stats from the supervisor
func (s *Collector) Run() {
	type publishersPair struct {
		container *container.Container
		publisher *pubsub.Publisher
	}
	// we cannot determine the capacity here.
	// it will grow enough in first iteration
	var pairs []publishersPair

	for range time.Tick(s.interval) {
		// it does not make sense in the first iteration,
		// but saves allocations in further iterations
		pairs = pairs[:0]

		s.m.Lock()
		for container, publisher := range s.publishers {
			// copy pointers here to release the lock ASAP
			pairs = append(pairs, publishersPair{container, publisher})
		}
		s.m.Unlock()
		if len(pairs) == 0 {
			continue
		}

		systemUsage, err := s.getSystemCPUUsage()
		if err != nil {
			logrus.Errorf("collecting system cpu usage: %v", err)
			continue
		}

		onlineCPUs, err := s.getNumberOnlineCPUs()
		if err != nil {
			logrus.Errorf("collecting system online cpu count: %v", err)
			continue
		}

		for _, pair := range pairs {
			stats, err := s.supervisor.GetContainerStats(pair.container)

			switch err.(type) {
			case nil:
				// FIXME: move to containerd on Linux (not Windows)
				stats.CPUStats.SystemUsage = systemUsage
				stats.CPUStats.OnlineCPUs = onlineCPUs

				pair.publisher.Publish(*stats)

			case notRunningErr, notFoundErr:
				// publish empty stats containing only name and ID if not running or not found
				pair.publisher.Publish(types.StatsJSON{
					Name: pair.container.Name,
					ID:   pair.container.ID,
				})

			default:
				logrus.Errorf("collecting stats for %s: %v", pair.container.ID, err)
			}
		}
	}
}

type notRunningErr interface {
	error
	ContainerIsRunning() bool
}

type notFoundErr interface {
	error
	ContainerNotFound() bool
}
