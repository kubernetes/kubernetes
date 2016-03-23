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
	"time"
	"sync"
	"k8s.io/kubernetes/pkg/api"
)

type schedulerCache struct {
	stop   chan struct{}
	ttl    time.Duration
	period time.Duration

	// This mutex guards all fields within this cache struct.
	mu sync.Mutex
	// a set of assumed pod keys.
	// The key could further be used to get an entry in podStates.
	assumedPods map[string]bool
	// a map from pod key to podState.
	rcStates map[string]*rcState
	clusters     map[string]*ClusterInfo
}

type rcState struct {
	rc *api.ReplicationController
	// Used by assumedPod to determinate expiration.
	deadline *time.Time
}

func newSchedulerCache(ttl, period time.Duration, stop chan struct{}) *schedulerCache {
	return &schedulerCache{
		ttl:    ttl,
		period: period,
		stop:   stop,

		clusters:       make(map[string]*ClusterInfo),
		assumedPods: make(map[string]bool),
		rcStates:   make(map[string]*rcState),
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