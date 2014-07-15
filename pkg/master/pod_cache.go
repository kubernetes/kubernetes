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
	"errors"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// PodCache contains both a cache of container information, as well as the mechanism for keeping
// that cache up to date.
type PodCache struct {
	containerInfo client.PodInfoGetter
	pods          registry.PodRegistry
	// This is a map of pod id to a map of container name to the
	podInfo map[string]api.PodInfo
	period  time.Duration
	podLock sync.Mutex
}

// NewPodCache returns a new PodCache which watches container information registered in the given PodRegistry.
func NewPodCache(info client.PodInfoGetter, pods registry.PodRegistry, period time.Duration) *PodCache {
	return &PodCache{
		containerInfo: info,
		pods:          pods,
		podInfo:       map[string]api.PodInfo{},
		period:        period,
	}
}

// GetPodInfo Implements the PodInfoGetter.GetPodInfo.
// The returned value should be treated as read-only.
func (p *PodCache) GetPodInfo(host, podID string) (api.PodInfo, error) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	value, ok := p.podInfo[podID]
	if !ok {
		return nil, errors.New("no cached pod info")
	}
	return value, nil
}

func (p *PodCache) updatePodInfo(host, id string) error {
	info, err := p.containerInfo.GetPodInfo(host, id)
	if err != nil {
		return err
	}
	p.podLock.Lock()
	defer p.podLock.Unlock()
	p.podInfo[id] = info
	return nil
}

// UpdateAllContainers updates information about all containers.  Either called by Loop() below, or one-off.
func (p *PodCache) UpdateAllContainers() {
	pods, err := p.pods.ListPods(labels.Everything())
	if err != nil {
		glog.Errorf("Error synchronizing container list: %#v", err)
		return
	}
	for _, pod := range pods {
		err := p.updatePodInfo(pod.CurrentState.Host, pod.ID)
		if err != nil {
			glog.Errorf("Error synchronizing container: %#v", err)
		}
	}
}

// Loop begins watching updates of container information.
// It runs forever, and is expected to be placed in a go routine.
func (p *PodCache) Loop() {
	util.Forever(func() { p.UpdateAllContainers() }, p.period)
}
