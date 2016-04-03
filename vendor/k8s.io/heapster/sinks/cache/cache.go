// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cache

import (
	"time"

	source_api "k8s.io/heapster/sources/api"
	kube_api "k8s.io/kubernetes/pkg/api"
)

type Metadata struct {
	Name         string
	Namespace    string
	NamespaceUID string
	UID          string
	Hostname     string
	Labels       map[string]string
	ExternalID   string
	LastUpdate   time.Time
}

type Event struct {
	Metadata
	// Detailed description of the event.
	Message string
	// The source component that generated the event.
	Source string
	// Store a public API version instead of internal version.
	Raw kube_api.Event
}

type ContainerMetricElement struct {
	Spec  *source_api.ContainerSpec
	Stats *source_api.ContainerStats
}

type ContainerElement struct {
	Metadata
	// Container base image.
	Image string
	// Data points are in reverse chronological order (most recent to oldest).
	Metrics []*ContainerMetricElement
}

type PodElement struct {
	Metadata
	// map of container name to container element.
	Containers []*ContainerElement
	// TODO: Cache history of Spec and Status.
}

// NodeContainerName is the container name assigned to node level metrics.
const NodeContainerName = "machine"

type EventsCache interface {
	StoreEvents([]*Event) error
}

type CacheListener struct {
	NodeEvicted          func(hostName string)
	NamespaceEvicted     func(namespace string)
	PodEvicted           func(namespace string, podName string)
	PodContainerEvicted  func(namespace string, podName string, containerName string)
	FreeContainerEvicted func(hostName string, containerName string)
}

type Cache interface {
	EventsCache
	StorePods([]source_api.Pod) error
	StoreContainers([]source_api.Container) error
	// TODO: Handle events.
	// GetPods returns a list of pod elements holding the metrics between 'start' and 'end' in the cache.
	// If 'start' is zero, it returns all the elements up until 'end'.
	// If 'end' is zero, it returns all the elements from 'start'.
	// If both 'start' and 'end' are zero, it returns all the elements in the cache.
	GetPods(start, end time.Time) []*PodElement

	// GetNodes returns a list of container elements holding the node level metrics between 'start' and 'end' in the cache.
	// If 'start' is zero, it returns all the elements up until 'end'.
	// If 'end' is zero, it returns all the elements from 'start'.
	// If both 'start' and 'end' are zero, it returns all the elements in the cache.
	GetNodes(start, end time.Time) []*ContainerElement

	// GetFreeContainers returns a list of container elements holding the metrics between 'start' and 'end' in the cache.
	// If 'start' is zero, it returns all the elements up until 'end'.
	// If 'end' is zero, it returns all the elements from 'start'.
	// If both 'start' and 'end' are zero, it returns all the elements in the cache.
	GetFreeContainers(start, end time.Time) []*ContainerElement
	// GetEvents returns a list of events in the cache between 'start' and 'end.
	// If 'start' is zero, it returns all the events up until 'end'.
	// If 'end' is zero, it returns all the events from 'start'.
	// If both 'start' and 'end' are zero, it returns all the events in the cache.
	GetEvents(start, end time.Time) []*Event

	// Adds eviction listener to the list of listeners. The listeners
	// will be triggered when the correspoinding eviction events occur.
	AddCacheListener(listener CacheListener)
}
