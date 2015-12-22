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

package nodeinfocache

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	clientcache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// New returns a NodeInfoCache implementation.
// It automatically starts a go routine that manages expiration of assumed pods.
// "ttl" is how long the assumed pod will get expired.
// "period" is how long the background goroutine should wait before cleaning up expired pods periodically.
// "stop" is the channel that signals stopping and we would close background goroutines.
func New(ttl, period time.Duration, stop chan struct{}) schedulercache.NodeInfoCache {
	cache := newNodeInfoCache(ttl, period, stop)
	cache.run()
	return cache
}

// mustGetPodKey returns the string key of a pod.
// A pod is ensured to have accessor. We don't want to check the error everytime.
// TODO: We should consider adding a Key() method to api.Pod
func mustGetPodKey(pod *api.Pod) string {
	key, err := clientcache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		panic("api.Pod should have key func: " + err.Error())
	}
	return key
}

type nodeInfoCache struct {
	stop chan struct{}

	// This mutex guards all fields within this cache struct
	mu          sync.Mutex
	ttl         time.Duration
	period      time.Duration
	assumedPods map[string]assumedPod
	podStates   map[string]podState
	// TODO: we should also watch node deletion and clean up node entries
	nodes map[string]*schedulercache.NodeInfo
}

type podState int

const (
	podAsummed podState = iota + 1
	podAdded
	podExpired
)

type assumedPod struct {
	nodeName string
	podInfo  schedulercache.PodInfo
	deadline time.Time
}

func newNodeInfoCache(ttl, period time.Duration, stop chan struct{}) *nodeInfoCache {
	return &nodeInfoCache{
		ttl:    ttl,
		period: period,
		stop:   stop,

		nodes:       make(map[string]*schedulercache.NodeInfo),
		podStates:   make(map[string]podState),
		assumedPods: make(map[string]assumedPod),
	}
}

func (cache *nodeInfoCache) run() {
	go util.Until(cache.cleanupExpiredAssumedPods, cache.period, cache.stop)
}

func (cache *nodeInfoCache) GetNodeInfo(nodeName string, cb func(*schedulercache.NodeInfo)) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	n := cache.getNodeInfo(nodeName)
	cb(n)
}

// GetNodeInfo returns cached NodeInfo. It returns nil if no such node if found for given node name.
func (cache *nodeInfoCache) getNodeInfo(nodeName string) *schedulercache.NodeInfo {
	n, ok := cache.nodes[nodeName]
	if !ok || n.PodNum == 0 {
		return nil
	}
	return n
}

func (cache *nodeInfoCache) AssumePod(pod *api.Pod) error {
	return cache.assumePod(pod, time.Now())
}

// assumePod exists for making test deterministic by taking time as input argument.
func (cache *nodeInfoCache) assumePod(pod *api.Pod, now time.Time) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(pod)
	if _, ok := cache.podStates[key]; ok {
		return fmt.Errorf("pod state wasn't initial but get assumed. Pod key: %v", key)
	}

	cache.addPod(pod)
	cache.podStates[key] = podAsummed
	aPod := assumedPod{
		nodeName: pod.Spec.NodeName,
		podInfo:  schedulercache.ParsePodInfo(pod),
		deadline: now.Add(cache.ttl),
	}
	cache.assumedPods[key] = aPod
	return nil
}

func (cache *nodeInfoCache) AddPod(pod *api.Pod) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(pod)
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAsummed:
		delete(cache.assumedPods, key)
	case ok && state == podExpired:
		// Pod was expired and deleted. We should add it back.
		cache.addPod(pod)
	default:
		return fmt.Errorf("pod state wasn't assumed or expired but get added. Pod key: %v", key)
	}
	cache.podStates[key] = podAdded
	return nil
}

func (cache *nodeInfoCache) UpdatePod(oldPod, newPod *api.Pod) error {
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

func (cache *nodeInfoCache) updatePod(oldPod, newPod *api.Pod, key string) {
	cache.deletePod(oldPod)
	cache.addPod(newPod)
}

func (cache *nodeInfoCache) addPod(pod *api.Pod) {
	n, ok := cache.nodes[pod.Spec.NodeName]
	if !ok {
		n = schedulercache.NewNodeInfo()
		cache.nodes[pod.Spec.NodeName] = n
	}
	n.AddPodInfo(schedulercache.ParsePodInfo(pod))
}

func (cache *nodeInfoCache) deletePod(pod *api.Pod) {
	n := cache.nodes[pod.Spec.NodeName]
	n.RemovePodInfo(schedulercache.ParsePodInfo(pod))
}

func (cache *nodeInfoCache) RemovePod(pod *api.Pod) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := mustGetPodKey(pod)
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podExpired:
	case ok && state == podAsummed:
		delete(cache.assumedPods, key)
		fallthrough
	case ok && state == podAdded:
		cache.deletePod(pod)
	default:
		return fmt.Errorf("pod state wasn't assumed, expired, or added but get removed. Pod key: %v", key)
	}
	delete(cache.podStates, key)
	return nil
}

func (cache *nodeInfoCache) cleanupExpiredAssumedPods() {
	cache.cleanupAssumedPods(time.Now())
}

// cleanupAssumedPods exists for making test deterministic by taking time as input argument.
func (cache *nodeInfoCache) cleanupAssumedPods(now time.Time) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// the size of assumedPods should be small
	for key, aPod := range cache.assumedPods {
		if now.After(aPod.deadline) {
			err := cache.expirePod(key, aPod)
			if err != nil {
				glog.Errorf("cache.expirePod failed: %v", err)
			}
		}
	}
}

func (cache *nodeInfoCache) expirePod(key string, aPod assumedPod) error {
	state, ok := cache.podStates[key]
	switch {
	case ok && state == podAsummed:
		delete(cache.assumedPods, key)
		// This is the same logic as deletePod but we don't keep the api.Pod pointer.
		// Instead we keep and use a local copy of information.
		n := cache.nodes[aPod.nodeName]
		n.RemovePodInfo(aPod.podInfo)
		cache.podStates[key] = podExpired
		return nil
	default:
		return fmt.Errorf("pod state wasn't assumed but get expired. Pod key: %v", key)
	}
}
