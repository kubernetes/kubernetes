/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"log"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// PodCache contains both a cache of container information, as well as the mechanism for keeping
// that cache up to date.
type PodCache struct {
	containerInfo client.ContainerInfo
	pods          registry.PodRegistry
	podInfo       map[string]interface{}
	period        time.Duration
	podLock       sync.Mutex
}

func NewPodCache(info client.ContainerInfo, pods registry.PodRegistry, period time.Duration) *PodCache {
	return &PodCache{
		containerInfo: info,
		pods:          pods,
		podInfo:       map[string]interface{}{},
		period:        period,
	}
}

// Implements the ContainerInfo interface
// The returned value should be treated as read-only
func (p *PodCache) GetContainerInfo(host, id string) (interface{}, error) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	value, ok := p.podInfo[id]
	if !ok {
		return nil, nil
	} else {
		return value, nil
	}
}

func (p *PodCache) updateContainerInfo(host, id string) error {
	info, err := p.containerInfo.GetContainerInfo(host, id)
	if err != nil {
		return err
	}
	p.podLock.Lock()
	defer p.podLock.Unlock()
	p.podInfo[id] = info
	return nil
}

// Update information about all containers.  Either called by Loop() below, or one-off.
func (p *PodCache) UpdateAllContainers() {
	pods, err := p.pods.ListPods(labels.Everything())
	if err != nil {
		log.Printf("Error synchronizing container: %#v", err)
		return
	}
	for _, pod := range pods {
		err := p.updateContainerInfo(pod.CurrentState.Host, pod.ID)
		if err != nil {
			log.Printf("Error synchronizing container: %#v", err)
		}
	}
}

// Loop runs forever, it is expected to be placed in a go routine.
func (p *PodCache) Loop() {
	util.Forever(func() { p.UpdateAllContainers() }, p.period)
}
