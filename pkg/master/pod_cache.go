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
	"reflect"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

type cacheEntry struct {
	info api.PodInfo
	mux  *watch.Mux
}

// PodCache contains both a cache of container information, as well as the mechanism for keeping
// that cache up to date.
type PodCache struct {
	containerInfo client.PodInfoGetter
	pods          pod.Registry
	// This is a map of pod id to a map of container name to the
	podInfo map[string]*cacheEntry
	podLock sync.Mutex
}

// NewPodCache returns a new PodCache which watches container information registered in the given PodRegistry.
func NewPodCache(info client.PodInfoGetter, pods pod.Registry) *PodCache {
	return &PodCache{
		containerInfo: info,
		pods:          pods,
		podInfo:       map[string]*cacheEntry{},
	}
}

// GetPodInfo implements the PodInfoGetter.GetPodInfo.
// The returned value should be treated as read-only.
// TODO: Remove the host from this call, it's totally unnecessary.
func (p *PodCache) GetPodInfo(host, podID string) (api.PodInfo, error) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	value, ok := p.podInfo[podID]
	if !ok {
		return nil, client.ErrPodInfoNotAvailable
	}
	return value.info, nil
}

func (p *PodCache) WatchPodInfo(host, podID string) (watch.Interface, error) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	entry, ok := p.podInfo[podID]
	if !ok {
		// TODO: Should we just watch anyway?
		return nil, client.ErrPodInfoNotAvailable
	}
	entry.mux = watch.NewMux(1)
	return entry.mux.Watch(), nil
}

func (p *PodCache) updatePodInfo(host, id string) error {
	info, err := p.containerInfo.GetPodInfo(host, id)
	if err != nil {
		return err
	}
	p.podLock.Lock()
	defer p.podLock.Unlock()
	entry, ok := p.podInfo[id]
	if !ok {
		entry = &cacheEntry{}
		p.podInfo[id] = entry
	}
	if entry.mux != nil && !reflect.DeepEqual(info, entry.info) {
		entry.mux.Action(watch.Modified, info)
	}
	entry.info = info

	return nil
}

// UpdateAllContainers updates information about all containers.  Either called by Loop() below, or one-off.
func (p *PodCache) UpdateAllContainers() {
	pods, err := p.pods.ListPods(labels.Everything())
	if err != nil {
		glog.Errorf("Error synchronizing container list: %v", err)
		return
	}
	for _, pod := range pods.Items {
		err := p.updatePodInfo(pod.CurrentState.Host, pod.ID)
		if err != nil && err != client.ErrPodInfoNotAvailable {
			glog.Errorf("Error synchronizing container: %v", err)
		}
	}
}
