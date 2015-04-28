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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const (
	minBackoffInterval = time.Second * 4
	maxBackoffInterval = time.Minute * 2
	// If no incidents have triggered an increase of backoff interval in the
	// past 5 minutes, we reset the interval and start over.
	backoffWindow = time.Minute * 5
)

type syncPodFnType func(*api.Pod, *api.Pod, kubecontainer.Pod) (*syncPodSummary, error)

type podWorkers struct {
	// Protects all per worker fields.
	podLock sync.Mutex

	// Tracks all running per-pod goroutines - per-pod goroutine will be
	// processing updates received through its corresponding channel.
	podUpdates map[types.UID]chan workUpdate
	// Track the current state of per-pod goroutines.
	// Currently all update request for a given pod coming when another
	// update of this pod is being processed are ignored.
	isWorking map[types.UID]bool
	// Tracks the last undelivered work item for this pod - a work item is
	// undelivered if it comes in while the worker is working.
	lastUndeliveredWorkUpdate map[types.UID]workUpdate
	// runtimeCache is used for listing running containers.
	runtimeCache kubecontainer.RuntimeCache

	// This function is run to sync the desired stated of pod.
	// NOTE: This function has to be thread-safe - it can be called for
	// different pods at the same time.
	syncPodFn syncPodFnType

	// The EventRecorder to use
	recorder record.EventRecorder
}

type workUpdate struct {
	// The pod state to reflect.
	pod *api.Pod

	// The mirror pod of pod; nil if it does not exist.
	mirrorPod *api.Pod

	// Function to call when the update is complete.
	updateCompleteFn func()
}

func newPodWorkers(runtimeCache kubecontainer.RuntimeCache, syncPodFn syncPodFnType,
	recorder record.EventRecorder) *podWorkers {
	return &podWorkers{
		podUpdates:                map[types.UID]chan workUpdate{},
		isWorking:                 map[types.UID]bool{},
		lastUndeliveredWorkUpdate: map[types.UID]workUpdate{},
		runtimeCache:              runtimeCache,
		syncPodFn:                 syncPodFn,
		recorder:                  recorder,
	}
}

// shouldBackOffMore returns true if the back off interval should be increased,
// based on the the summary of the last sync.
func shouldBackOffMore(summary *syncPodSummary) bool {
	if summary.Unhealthy > 0 || summary.NotFound > 0 {
		// Some containers were restarted in the last sync, we should increase
		// the backoff interval. Note that we "NotFound" could also mean that
		// the container was started the first time; the caller could rule out
		// the possiblity by checking whether they are seeing the pod the first
		// time. However, if we ever allow the addition/deletion of containers
		// in in a pod, we need to revise this logic.
		return true
	}
	return false
}

// computeNewBackoffTime computes and returns the new backoff interval,
// along with the timestamp where the interval was last increased.
func computeBackoffTime(summary *syncPodSummary, lastInterval time.Duration, firstSync bool,
	lastIncreaseTime time.Time) (time.Duration, time.Time) {
	if firstSync {
		// If we are syncing this pod for the first time, don't back off.
		return 0 * time.Second, lastIncreaseTime
	}

	if lastIncreaseTime.Add(backoffWindow).Before(time.Now()) {
		// If we increased the backoff interval more than backoffWindow ago, we
		// should reset the interval. This ensures that ancient restarts do not
		// affect the current backoff interval.
		lastInterval = 0 * time.Second
	}
	if shouldBackOffMore(summary) {
		if lastInterval == 0 {
			return minBackoffInterval, time.Now()
		}
		// Double the backoff interval unless it is capped by maxBackoffInterval.
		if lastInterval*2 > maxBackoffInterval {
			return maxBackoffInterval, time.Now()
		} else {
			return lastInterval * 2, time.Now()
		}
	}

	return lastInterval, lastIncreaseTime
}

func (p *podWorkers) managePodLoop(podUpdates <-chan workUpdate) {
	var minRuntimeCacheTime time.Time
	var backoffInterval time.Duration
	// This timestamp records the last time the backoff interval was increasd.
	var lastIncreaseTime time.Time
	firstSync := true
	for newWork := range podUpdates {
		func() {
			defer p.checkForUpdates(newWork.pod.UID, newWork.updateCompleteFn)

			// We would like to have the state of Docker from at least the moment
			// when we finished the previous processing of that pod.
			if err := p.runtimeCache.ForceUpdateIfOlder(minRuntimeCacheTime); err != nil {
				glog.Errorf("Error updating docker cache: %v", err)
				return
			}
			pods, err := p.runtimeCache.GetPods()
			if err != nil {
				glog.Errorf("Error getting pods while syncing pod: %v", err)
				return
			}

			summary, err := p.syncPodFn(newWork.pod, newWork.mirrorPod,
				kubecontainer.Pods(pods).FindPodByID(newWork.pod.UID))
			if err != nil {
				glog.Errorf("Error syncing pod %s, skipping: %v", newWork.pod.UID, err)
				p.recorder.Eventf(newWork.pod, "failedSync", "Error syncing pod, skipping: %v", err)
				return
			}
			minRuntimeCacheTime = time.Now()

			backoffInterval, lastIncreaseTime = computeBackoffTime(summary, backoffInterval, firstSync, lastIncreaseTime)
			if backoffInterval > 0 {
				p.recorder.Eventf(newWork.pod, "backOff", "Containers restarted; backing off %v", backoffInterval)
				time.Sleep(backoffInterval)
			}
			firstSync = false

			newWork.updateCompleteFn()
		}()
	}
}

// Apply the new setting to the specified pod. updateComplete is called when the update is completed.
func (p *podWorkers) UpdatePod(pod *api.Pod, mirrorPod *api.Pod, updateComplete func()) {
	uid := pod.UID
	var podUpdates chan workUpdate
	var exists bool

	p.podLock.Lock()
	defer p.podLock.Unlock()
	if podUpdates, exists = p.podUpdates[uid]; !exists {
		// We need to have a buffer here, because checkForUpdates() method that
		// puts an update into channel is called from the same goroutine where
		// the channel is consumed. However, it is guaranteed that in such case
		// the channel is empty, so buffer of size 1 is enough.
		podUpdates = make(chan workUpdate, 1)
		p.podUpdates[uid] = podUpdates
		go func() {
			defer util.HandleCrash()
			p.managePodLoop(podUpdates)
		}()
	}
	if !p.isWorking[pod.UID] {
		p.isWorking[pod.UID] = true
		podUpdates <- workUpdate{
			pod:              pod,
			mirrorPod:        mirrorPod,
			updateCompleteFn: updateComplete,
		}
	} else {
		p.lastUndeliveredWorkUpdate[pod.UID] = workUpdate{
			pod:              pod,
			mirrorPod:        mirrorPod,
			updateCompleteFn: updateComplete,
		}
	}
}

func (p *podWorkers) ForgetNonExistingPodWorkers(desiredPods map[types.UID]empty) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	for key, channel := range p.podUpdates {
		if _, exists := desiredPods[key]; !exists {
			close(channel)
			delete(p.podUpdates, key)
			// If there is an undelivered work update for this pod we need to remove it
			// since per-pod goroutine won't be able to put it to the already closed
			// channel when it finish processing the current work update.
			if _, cached := p.lastUndeliveredWorkUpdate[key]; cached {
				delete(p.lastUndeliveredWorkUpdate, key)
			}
		}
	}
}

func (p *podWorkers) checkForUpdates(uid types.UID, updateComplete func()) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if workUpdate, exists := p.lastUndeliveredWorkUpdate[uid]; exists {
		p.podUpdates[uid] <- workUpdate
		delete(p.lastUndeliveredWorkUpdate, uid)
	} else {
		p.isWorking[uid] = false
	}
}
