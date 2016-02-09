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

package schedulercache

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
)

// New returns a Cache implementation.
// It automatically starts a go routine that manages expiration of assumed pods.
// "ttl" is how long the assumed pod will get expired.
// "period" is how often the background goroutine should clean up expired pods.
// "stop" is the channel that signals stopping and we would close background goroutines.
func New(ttl, period time.Duration, stop chan struct{}) Cache {
	cache := newSchedulerCache(ttl, period, stop)
	cache.run()
	return cache
}

type schedulerCache struct {
	stop   chan struct{}
	ttl    time.Duration
	period time.Duration

	// This mutex guards all fields within this cache struct
	mu          sync.Mutex
	assumedPods map[string]assumedPod
	podStates   map[string]podState
	nodes       map[string]*NodeInfo
}

type podState int

const (
	podAsummed podState = iota + 1
	podAdded
)

type assumedPod struct {
	pod      *api.Pod
	deadline time.Time
}

func newSchedulerCache(ttl, period time.Duration, stop chan struct{}) *schedulerCache {
	return &schedulerCache{
		ttl:    ttl,
		period: period,
		stop:   stop,

		nodes:       make(map[string]*NodeInfo),
		podStates:   make(map[string]podState),
		assumedPods: make(map[string]assumedPod),
	}
}

func (cache *schedulerCache) GetNodeNameToInfoMap() map[string]*NodeInfo {
	nodeNameToInfo := make(map[string]*NodeInfo)
	cache.mu.Lock()
	defer cache.mu.Unlock()
	for name, info := range cache.nodes {
		nodeNameToInfo[name] = info.Clone()
	}
	return nodeNameToInfo
}

func (cache *schedulerCache) List(selector labels.Selector) ([]*api.Pod, error) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	var pods []*api.Pod
	for _, info := range cache.nodes {
		pods = append(pods, info.pods...)
	}
	return pods, nil
}

func (cache *schedulerCache) AssumePodIfBindSucceed(pod *api.Pod, bind func() bool) error {
	return cache.assumePodIfBindSucceed(pod, bind, time.Now())
}

// assumePodScheduled exists for making test deterministic by taking time as input argument.
func (cache *schedulerCache) assumePodIfBindSucceed(pod *api.Pod, bind func() bool, now time.Time) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	if !bind() {
		return nil
	}

	key := mustGetPodKey(pod)
	if _, ok := cache.podStates[key]; ok {
		return fmt.Errorf("pod state wasn't initial but get assumed. Pod key: %v", key)
	}

	cache.addPod(pod)
	cache.podStates[key] = podAsummed
	aPod := assumedPod{
		pod:      pod,
		deadline: now.Add(cache.ttl),
	}
	cache.assumedPods[key] = aPod
	return nil
}

func (cache *schedulerCache) AddPod(pod *api.Pod) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(pod)
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAsummed:
		delete(cache.assumedPods, key)
	case !ok:
		// Pod was expired. We should add it back.
		cache.addPod(pod)
	default:
		return fmt.Errorf("pod state wasn't assumed or expired but get added. Pod key: %v", key)
	}
	cache.podStates[key] = podAdded
	return nil
}

func (cache *schedulerCache) UpdatePod(oldPod, newPod *api.Pod) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(oldPod)
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAdded:
		cache.updatePod(oldPod, newPod, key)
	default:
		return fmt.Errorf("pod state wasn't added but get updated. Pod key: %v", key)
	}
	return nil
}

func (cache *schedulerCache) updatePod(oldPod, newPod *api.Pod, key string) {
	cache.deletePod(oldPod)
	cache.addPod(newPod)
}

func (cache *schedulerCache) addPod(pod *api.Pod) {
	n, ok := cache.nodes[pod.Spec.NodeName]
	if !ok {
		n = NewNodeInfo()
		cache.nodes[pod.Spec.NodeName] = n
	}
	n.addPod(pod)
}

func (cache *schedulerCache) deletePod(pod *api.Pod) {
	n := cache.nodes[pod.Spec.NodeName]
	n.removePod(pod)
	if len(n.pods) == 0 {
		delete(cache.nodes, pod.Spec.NodeName)
	}
}

func (cache *schedulerCache) RemovePod(pod *api.Pod) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(pod)
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAdded:
		cache.deletePod(pod)
		delete(cache.podStates, key)
	default:
		return fmt.Errorf("pod state wasn't added but get removed. Pod key: %v", key)
	}
	return nil
}

func (cache *schedulerCache) run() {
	go wait.Until(cache.cleanupExpiredAssumedPods, cache.period, cache.stop)
}

func (cache *schedulerCache) cleanupExpiredAssumedPods() {
	cache.cleanupAssumedPods(time.Now())
}

// cleanupAssumedPods exists for making test deterministic by taking time as input argument.
func (cache *schedulerCache) cleanupAssumedPods(now time.Time) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// The size of assumedPods should be small
	for key, aPod := range cache.assumedPods {
		if now.After(aPod.deadline) {
			err := cache.expirePod(key, aPod)
			if err != nil {
				panic(err)
			}
		}
	}
}

func (cache *schedulerCache) expirePod(key string, aPod assumedPod) error {
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAsummed:
		cache.deletePod(aPod.pod)
		delete(cache.assumedPods, key)
		delete(cache.podStates, key)
		return nil
	default:
		return fmt.Errorf("pod state wasn't assumed but get expired. Pod key: %v", key)
	}
}
