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

package kubelet

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

type syncPodFunType func(*api.BoundPod, dockertools.DockerContainers) error

// TODO(wojtek-t) Add unit tests for this type.
type podWorkers struct {
	// Protects podUpdates field.
	podLock sync.Mutex

	// Tracks all running per-pod goroutines - per-pod goroutine will be
	// processing updates received through its corresponding channel.
	podUpdates map[types.UID]chan api.BoundPod
	// Track the current state of per-pod goroutines.
	// Currently all update request for a given pod coming when another
	// update of this pod is being processed are ignored.
	isWorking map[types.UID]bool
	// DockerCache is used for listing running containers.
	dockerCache dockertools.DockerCache

	// This function is run to sync the desired stated of pod.
	// NOTE: This function has to be thread-safe - it can be called for
	// different pods at the same time.
	syncPodFun syncPodFunType
}

func newPodWorkers(dockerCache dockertools.DockerCache, syncPodFun syncPodFunType) *podWorkers {
	return &podWorkers{
		podUpdates:  map[types.UID]chan api.BoundPod{},
		isWorking:   map[types.UID]bool{},
		dockerCache: dockerCache,
		syncPodFun:  syncPodFun,
	}
}

func (p *podWorkers) managePodLoop(podUpdates <-chan api.BoundPod) {
	for newPod := range podUpdates {
		// Since we use docker cache, getting current state shouldn't cause
		// performance overhead on Docker. Moreover, as long as we run syncPod
		// no matter if it changes anything, having an old version of "containers"
		// can cause starting eunended containers.
		func() {
			defer p.setIsWorking(newPod.UID, false)
			containers, err := p.dockerCache.RunningContainers()
			if err != nil {
				glog.Errorf("Error listing containers while syncing pod: %v", err)
				return
			}
			err = p.syncPodFun(&newPod, containers)
			if err != nil {
				glog.Errorf("Error syncing pod %s, skipping: %v", newPod.UID, err)
				record.Eventf(&newPod, "failedSync", "Error syncing pod, skipping: %v", err)
				return
			}
		}()
	}
}

func (p *podWorkers) UpdatePod(pod api.BoundPod) {
	uid := pod.UID
	var podUpdates chan api.BoundPod
	var exists bool

	p.podLock.Lock()
	defer p.podLock.Unlock()
	if podUpdates, exists = p.podUpdates[uid]; !exists {
		// Currently all update request for a given pod coming when another
		// update of this pod is being processed are ignored.
		podUpdates = make(chan api.BoundPod, 1)
		p.podUpdates[uid] = podUpdates
		go p.managePodLoop(podUpdates)
	}
	// TODO(wojtek-t): Consider changing to the following model:
	// - add a cache of "desired" pod state
	// - whenever an update of a pod comes, we update the "desired" cache
	// - if per-pod goroutine is currently iddle, we send the it immediately
	//   to the per-pod goroutine and clear the cache;
	// - when per-pod goroutine finishes processing an update it checks the
	//   desired cache for next update to proces
	// - the crucial thing in this approach is that we don't accumulate multiple
	//   updates for a given pod (at any point in time there will be at most
	//   one update queued for a given pod, plus potentially one currently being
	//   processed) and additionally don't rely on the fact that an update will
	//   be resend (because we don't drop it)
	if !p.isWorking[pod.UID] {
		p.isWorking[pod.UID] = true
		podUpdates <- pod
	}
}

func (p *podWorkers) ForgetNonExistingPodWorkers(desiredPods map[types.UID]empty) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	for key, channel := range p.podUpdates {
		if _, exists := desiredPods[key]; !exists {
			close(channel)
			delete(p.podUpdates, key)
		}
	}
}

func (p *podWorkers) setIsWorking(uid types.UID, isWorking bool) {
	p.podLock.Lock()
	p.isWorking[uid] = isWorking
	p.podLock.Unlock()
}
