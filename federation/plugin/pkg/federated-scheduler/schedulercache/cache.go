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
	"time"
	"sync"

	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

var (
	cleanAssumedPeriod = 1 * time.Second
)
// New returns a Cache implementation.
// It automatically starts a go routine that manages expiration of assumed replicaSets.
// "ttl" is how long the assumed replicaSet will get expired.
// "stop" is the channel that would close the background goroutine.
func New(ttl time.Duration, stop chan struct{}) Cache {
	cache := newSchedulerCache(ttl, cleanAssumedPeriod, stop)
	cache.run()
	return cache
}

type schedulerCache struct {
	stop         chan struct{}
	ttl          time.Duration
	period       time.Duration

	// This mutex guards all fields within this cache struct.
	mu           sync.Mutex
	// a set of assumed replicaSet keys.
	// The key could further be used to get an entry in replicaSetStates.
	assumedReplicaSet map[string]bool
	// a map from replicaSet key to replicaSetState.
	replicaSetStates  map[string]*replicaSetState
	clusters     map[string]*ClusterInfo
}

type replicaSetState struct {
	replicaSet    *v1beta1.ReplicaSet
	// Used by assumedReplicaSet to determinate expiration.
	deadline *time.Time
}

func newSchedulerCache(ttl, period time.Duration, stop chan struct{}) *schedulerCache {
	return &schedulerCache{
		ttl:    ttl,
		period: period,
		stop:   stop,

		clusters:       make(map[string]*ClusterInfo),
		assumedReplicaSet: make(map[string]bool),
		replicaSetStates:   make(map[string]*replicaSetState),
	}
}

func (cache *schedulerCache) GetClusterNameToInfoMap() (map[string]*ClusterInfo, error) {
	clusterNameToInfo := make(map[string]*ClusterInfo)
	cache.mu.Lock()
	defer cache.mu.Unlock()
	for name, info := range cache.clusters {
		clusterNameToInfo[name] = info.Clone()
	}
	return clusterNameToInfo, nil
}

func (cache *schedulerCache) AssumeReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	return cache.assumeReplicaSet(replicaSet, time.Now())
}

func (cache *schedulerCache) assumeReplicaSet(replicaSet *v1beta1.ReplicaSet, now time.Time) error {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key, err := getReplicaSetKey(replicaSet)
	if err != nil {
		return err
	}
	if _, ok := cache.replicaSetStates[key]; ok {
		return fmt.Errorf("replicaSet state wasn't initial but get assumed. ReplicaSet key: %v", key)
	}
	cache.addReplicaSet(replicaSet)
	dl := now.Add(cache.ttl)
	rss := &replicaSetState{
		replicaSet: replicaSet,
		deadline: &dl,
	}
	cache.replicaSetStates[key] = rss
	cache.assumedReplicaSet[key] = true
	return nil
}

func (cache *schedulerCache) AddReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	key, err := getReplicaSetKey(replicaSet)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	_, ok := cache.replicaSetStates[key]
	switch {
	case ok && cache.assumedReplicaSet[key]:
		delete(cache.assumedReplicaSet, key)
		cache.replicaSetStates[key].deadline = nil
	case !ok:
		// ReplicaSet was expired. We should add it back.
		cache.addReplicaSet(replicaSet)
		rss := &replicaSetState{
			replicaSet: replicaSet,
		}
		cache.replicaSetStates[key] = rss
	default:
		return fmt.Errorf("replicaSet was already in added state. ReplicaSet key: %v", key)
	}
	return nil
}

func (cache *schedulerCache) UpdateReplicaSet(oldReplicaSet, newReplicaSet *v1beta1.ReplicaSet) error {
	key, err := getReplicaSetKey(oldReplicaSet)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	_, ok := cache.replicaSetStates[key]
	switch {
	// An assumed replicaSet won't have Update/Remove event. It needs to have Add event
	// before Update event, in which case the state would change from Assumed to Added.
	case ok && !cache.assumedReplicaSet[key]:
		if err := cache.updateReplicaSet(oldReplicaSet, newReplicaSet); err != nil {
			return err
		}
	case ok && cache.assumedReplicaSet[key]:
		//As we have no field to filter scheduled replicasets, so we simply watch all replicasets
		//AddReplicaSet will never be called at this stage
		//So here if assumedReplicaSet info is found, it should be clean
		//We do not support other replicaset update cases at phase I
		delete(cache.assumedReplicaSet, key)
		cache.replicaSetStates[key].deadline = nil
	default:

		return fmt.Errorf("replicaSet state wasn't added but get updated. ReplicaSet key: %v", key)
	}
	return nil
}

func (cache *schedulerCache) updateReplicaSet(oldReplicaSet, newReplicaSet *v1beta1.ReplicaSet) error {
	if err := cache.deleteReplicaSet(oldReplicaSet); err != nil {
		return err
	}
	cache.addReplicaSet(newReplicaSet)
	return nil
}

func (cache *schedulerCache) addReplicaSet(replicaSet *v1beta1.ReplicaSet) {
	annotations := replicaSet.Annotations
	key, _ := annotations[unversioned.TargetClusterKey]
	c, ok := cache.clusters[key]
	if !ok {
		c = NewClusterInfo()
		cache.clusters[key] = c
	}
	c.addReplicaSet(replicaSet)
}

func (cache *schedulerCache) deleteReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	annotations := replicaSet.Annotations
	key, _ := annotations[unversioned.TargetClusterKey]
	c := cache.clusters[key]
	if err := c.removeReplicaSet(replicaSet); err != nil {
		return err
	}
	if len(c.SubReplicaSets()) == 0 {
		delete(cache.clusters, key)
	}
	return nil
}

func (cache *schedulerCache) RemoveReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	key, err := getReplicaSetKey(replicaSet)
	if err != nil {
		return err
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	_, ok := cache.replicaSetStates[key]
	switch {
	// An assumed replicaSet won't have Delete/Remove event. It needs to have Add event
	// before Remove event, in which case the state would change from Assumed to Added.
	case ok && !cache.assumedReplicaSet[key]:
		err := cache.deleteReplicaSet(replicaSet)
		if err != nil {
			return err
		}
		delete(cache.replicaSetStates, key)
	default:
		return fmt.Errorf("replicaSet state wasn't added but get removed. ReplicaSet key: %v", key)
	}
	return nil
}

func (cache *schedulerCache) run() {
	go wait.Until(cache.cleanupExpiredAssumedReplicaSet, cache.period, cache.stop)
}

func (cache *schedulerCache) cleanupExpiredAssumedReplicaSet() {
	cache.cleanupAssumedReplicaSet(time.Now())
}

// cleanupAssumedReplicaSets exists for making test deterministic by taking time as input argument.
func (cache *schedulerCache) cleanupAssumedReplicaSet(now time.Time) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	// The size of assumedReplicaSets should be small
	for key := range cache.assumedReplicaSet {
		rss, ok := cache.replicaSetStates[key]
		if !ok {
			panic("Key found in assumed set but not in replicaSetStates. Potentially a logical error.")
		}
		if now.After(*rss.deadline) {
			if err := cache.expireReplicaSet(key, rss); err != nil {
				glog.Errorf("expireReplicaSet failed for %s: %v", key, err)
			}
		}
	}
}

func (cache *schedulerCache) expireReplicaSet(key string, ps *replicaSetState) error {
	if err := cache.deleteReplicaSet(ps.replicaSet); err != nil {
		return err
	}
	delete(cache.assumedReplicaSet, key)
	delete(cache.replicaSetStates, key)
	return nil
}

func (cache *schedulerCache) List() ([]*v1beta1.ReplicaSet, error) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	var result []*v1beta1.ReplicaSet
	for _, info := range cache.clusters {
		for _, rs := range info.replicaSets {
				result = append(result, rs)
		}
	}
	return result, nil
}
