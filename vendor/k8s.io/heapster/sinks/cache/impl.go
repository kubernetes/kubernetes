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
	"fmt"
	"sync"
	"time"

	source_api "k8s.io/heapster/sources/api"
	"k8s.io/heapster/store"
	hUtil "k8s.io/heapster/util"
	"k8s.io/kubernetes/pkg/util"
)

const rootContainer = "/"

type containerElement struct {
	lastUpdated time.Time
	Metadata
	Image   string
	metrics store.TimeStore
}

type podElement struct {
	lastUpdated time.Time
	Metadata
	// map of container name to container element.
	containers map[string]*containerElement
	// TODO: Cache history of Spec and Status.
}

type nodeElement struct {
	node *containerElement
	// FreeContainers refers to all the containers in a node
	// that do not belong to a pod.
	freeContainers map[string]*containerElement
	// TODO: Cache history of Spec and Status.
}

type realCache struct {
	bufferDuration time.Duration
	// Map of pod UIDs to pod cache entry.
	pods map[string]*podElement
	// Map of node hostnames to node cache entry.
	nodes map[string]*nodeElement
	// Event UIDs. This is required to avoid storing duplicate events.
	eventUIDs map[string]struct{}
	// Events store.
	events store.TimeStore

	cacheListeners []CacheListener

	sync.RWMutex
}

func (rc *realCache) newContainerElement() *containerElement {
	return &containerElement{
		metrics:     store.NewGCStore(store.NewTimeStore(), rc.bufferDuration),
		lastUpdated: time.Now(),
	}
}

func (rc *realCache) isTooOld(lastUpdated time.Time) bool {
	if time.Now().Sub(lastUpdated) >= rc.bufferDuration {
		return true
	}
	return false
}

func (rc *realCache) AddCacheListener(cacheListener CacheListener) {
	rc.Lock()
	defer rc.Unlock()
	rc.cacheListeners = append(rc.cacheListeners, cacheListener)
}

func (rc *realCache) runGC() {
	rc.Lock()
	defer rc.Unlock()
	for podName, podElem := range rc.pods {
		for contName, contElem := range podElem.containers {
			if rc.isTooOld(contElem.lastUpdated) {
				delete(podElem.containers, contName)
				for _, listener := range rc.cacheListeners {
					if listener.PodContainerEvicted != nil {
						listener.PodContainerEvicted(podElem.Namespace, podElem.Name, contName)
					}
				}
			}
		}
		if rc.isTooOld(podElem.lastUpdated) {
			delete(rc.pods, podName)
			for _, listener := range rc.cacheListeners {
				if listener.PodEvicted != nil {
					listener.PodEvicted(podElem.Namespace, podElem.Name)
				}
			}
		}
	}

	for nodeName, nodeElem := range rc.nodes {
		for contName, contElem := range nodeElem.freeContainers {
			if rc.isTooOld(contElem.lastUpdated) {
				delete(nodeElem.freeContainers, contName)
				for _, listener := range rc.cacheListeners {
					if listener.FreeContainerEvicted != nil {
						listener.FreeContainerEvicted(nodeName, contName)
					}
				}
			}
		}

		if rc.isTooOld(nodeElem.node.lastUpdated) {
			delete(rc.nodes, nodeName)
			for _, listener := range rc.cacheListeners {
				if listener.NodeEvicted != nil {
					listener.NodeEvicted(nodeName)
				}
			}
		}
	}
}

func (rc *realCache) newPodElement() *podElement {
	return &podElement{
		containers:  make(map[string]*containerElement),
		lastUpdated: time.Now(),
	}
}

func (rc *realCache) newNodeElement() *nodeElement {
	return &nodeElement{
		node:           rc.newContainerElement(),
		freeContainers: make(map[string]*containerElement),
	}
}

func storeSpecAndStats(ce *containerElement, c *source_api.Container) time.Time {
	if ce == nil || c == nil {
		return time.Time{}
	}
	latestTimestamp := time.Time{}
	for idx := range c.Stats {
		if c.Stats[idx] == nil {
			continue
		}
		cme := &ContainerMetricElement{
			Spec:  &c.Spec,
			Stats: c.Stats[idx],
		}
		ce.metrics.Put(store.TimePoint{
			Timestamp: c.Stats[idx].Timestamp,
			Value:     cme,
		})
		latestTimestamp = hUtil.GetLatest(latestTimestamp, c.Stats[idx].Timestamp)
	}
	return latestTimestamp
}

func (rc *realCache) StorePods(pods []source_api.Pod) error {
	rc.Lock()
	defer rc.Unlock()
	now := time.Now()
	for _, pod := range pods {
		pe, ok := rc.pods[pod.ID]
		if !ok {
			pe = rc.newPodElement()
			pe.Metadata = Metadata{
				Name:         pod.Name,
				Namespace:    pod.Namespace,
				NamespaceUID: pod.NamespaceUID,
				UID:          pod.ID,
				Hostname:     pod.Hostname,
				Labels:       pod.Labels,
				ExternalID:   pod.ExternalID,
			}
			rc.pods[pod.ID] = pe
		}
		for idx := range pod.Containers {
			cont := &pod.Containers[idx]
			ce, ok := pe.containers[cont.Name]
			if !ok {
				ce = rc.newContainerElement()
				pe.containers[cont.Name] = ce
			}
			ce.Metadata = Metadata{
				Name:     cont.Name,
				Hostname: cont.Hostname,
				Labels:   cont.Spec.Labels,
			}
			ce.Image = cont.Image
			ce.Metadata.LastUpdate = storeSpecAndStats(ce, cont)
			ce.lastUpdated = now
			pe.LastUpdate = hUtil.GetLatest(pe.LastUpdate, ce.Metadata.LastUpdate)
		}
		pe.lastUpdated = now
	}
	return nil
}

func (rc *realCache) StoreContainers(containers []source_api.Container) error {
	rc.Lock()
	defer rc.Unlock()
	now := time.Now()
	for idx := range containers {
		cont := &containers[idx]
		ne, ok := rc.nodes[cont.Hostname]
		if !ok {
			ne = rc.newNodeElement()
			rc.nodes[cont.Hostname] = ne
		}
		var ce *containerElement
		if cont.Name == rootContainer {
			// This is at the node level.
			ne.node.Hostname = cont.Hostname
			ne.node.Name = NodeContainerName
			ne.node.ExternalID = cont.ExternalID
			ce = ne.node
		} else {
			var ok bool
			ce, ok = ne.freeContainers[cont.Name]
			if !ok {
				ce = rc.newContainerElement()
				ce.Metadata = Metadata{
					Name:       cont.Name,
					Hostname:   cont.Hostname,
					ExternalID: cont.ExternalID,
				}
				ne.freeContainers[cont.Name] = ce
			}
		}
		ce.Metadata.Labels = cont.Spec.Labels
		ce.Image = cont.Image
		ce.Metadata.LastUpdate = storeSpecAndStats(ce, cont)
		ce.lastUpdated = now
	}
	return nil
}

func (rc *realCache) GetPods(start, end time.Time) []*PodElement {
	rc.RLock()
	defer rc.RUnlock()
	var result []*PodElement
	for _, pe := range rc.pods {
		podElement := &PodElement{
			Metadata: pe.Metadata,
		}
		for _, ce := range pe.containers {
			containerElement := &ContainerElement{
				Metadata: ce.Metadata,
				Image:    ce.Image,
			}
			metrics := ce.metrics.Get(start, end)
			for idx := range metrics {
				cme := metrics[idx].Value.(*ContainerMetricElement)
				containerElement.Metrics = append(containerElement.Metrics, cme)
			}
			podElement.Containers = append(podElement.Containers, containerElement)
		}
		result = append(result, podElement)
	}

	return result
}

func (rc *realCache) GetNodes(start, end time.Time) []*ContainerElement {
	rc.RLock()
	defer rc.RUnlock()
	var result []*ContainerElement
	for _, ne := range rc.nodes {
		ce := &ContainerElement{
			Metadata: ne.node.Metadata,
		}
		metrics := ne.node.metrics.Get(start, end)
		for idx := range metrics {
			cme := metrics[idx].Value.(*ContainerMetricElement)
			ce.Metrics = append(ce.Metrics, cme)
		}
		result = append(result, ce)
	}
	return result
}

func (rc *realCache) GetFreeContainers(start, end time.Time) []*ContainerElement {
	rc.RLock()
	defer rc.RUnlock()
	var result []*ContainerElement
	for _, ne := range rc.nodes {
		for _, ce := range ne.freeContainers {
			containerElement := &ContainerElement{
				Metadata: ce.Metadata,
				Image:    ce.Image,
			}
			metrics := ce.metrics.Get(start, end)
			for idx := range metrics {
				cme := metrics[idx].Value.(*ContainerMetricElement)
				containerElement.Metrics = append(containerElement.Metrics, cme)
			}
			result = append(result, containerElement)
		}
	}
	return result
}

func (rc *realCache) StoreEvents(e []*Event) error {
	rc.Lock()
	defer rc.Unlock()
	for idx := range e {
		uid := e[idx].UID
		if uid == "" {
			return fmt.Errorf("failed to store event %v - UID cannot be empty", *e[idx])
		}
		if _, exists := rc.eventUIDs[uid]; exists {
			continue
		}
		rc.eventUIDs[e[idx].UID] = struct{}{}
		if err := rc.events.Put(store.TimePoint{
			Timestamp: e[idx].LastUpdate,
			Value:     e[idx],
		}); err != nil {
			return err
		}
	}
	return nil
}

func (rc *realCache) GetEvents(start, end time.Time) []*Event {
	rc.RLock()
	defer rc.RUnlock()
	var result []*Event
	timePoints := rc.events.Get(start, end)
	for idx := range timePoints {
		result = append(result, timePoints[idx].Value.(*Event))
	}
	return result
}

func NewCache(bufferDuration, gcDuration time.Duration) Cache {
	rc := &realCache{
		pods:           make(map[string]*podElement),
		nodes:          make(map[string]*nodeElement),
		events:         store.NewGCStore(store.NewTimeStore(), bufferDuration),
		eventUIDs:      make(map[string]struct{}),
		bufferDuration: bufferDuration,
	}
	go util.Until(rc.runGC, gcDuration, util.NeverStop)
	return rc
}
