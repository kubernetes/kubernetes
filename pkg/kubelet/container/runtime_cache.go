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

package container

import (
	"sync"
	"time"
)

var (
	// TODO(yifan): Maybe set the them as parameters for NewCache().
	defaultCachePeriod    = time.Second * 2
	defaultUpdateInterval = time.Millisecond * 100
)

type RuntimeCache interface {
	GetPods() ([]*Pod, error)
	ForceUpdateIfOlder(time.Time) error
}

// TODO(yifan): This interface can be removed once docker manager has implemented
// all the runtime interfaces, (thus we can pass the runtime directly).
type podsGetter interface {
	GetPods(bool) ([]*Pod, error)
}

// NewRuntimeCache creates a container runtime cache.
func NewRuntimeCache(getter podsGetter) (RuntimeCache, error) {
	return &runtimeCache{
		getter:   getter,
		updating: false,
	}, nil
}

type runtimeCache struct {
	sync.Mutex
	// The underlying container runtime used to update the cache.
	getter podsGetter
	// Last time when cache was updated.
	cacheTime time.Time
	// The content of the cache.
	pods []*Pod
	// Whether the background thread updating the cache is running.
	updating bool
	// Time when the background thread should be stopped.
	updatingThreadStopTime time.Time
}

// GetPods returns the cached result for ListPods if the result is not
// outdated, otherwise it will retrieve the newest result.
// If the cache updating loop has stopped, this function will restart it.
func (r *runtimeCache) GetPods() ([]*Pod, error) {
	r.Lock()
	defer r.Unlock()
	if time.Since(r.cacheTime) > defaultCachePeriod {
		if err := r.updateCache(); err != nil {
			return nil, err
		}
	}
	// Stop refreshing thread if there were no requests within the default cache period
	r.updatingThreadStopTime = time.Now().Add(defaultCachePeriod)
	if !r.updating {
		r.updating = true
		go r.startUpdatingCache()
	}
	return r.pods, nil
}

func (r *runtimeCache) ForceUpdateIfOlder(minExpectedCacheTime time.Time) error {
	r.Lock()
	defer r.Unlock()
	if r.cacheTime.Before(minExpectedCacheTime) {
		return r.updateCache()
	}
	return nil
}

func (r *runtimeCache) updateCache() error {
	pods, err := r.getter.GetPods(false)
	if err != nil {
		return err
	}
	r.pods = pods
	r.cacheTime = time.Now()
	return nil
}

// startUpdateingCache continues to invoke GetPods to get the newest result until
// there are no requests within the default cache period.
func (r *runtimeCache) startUpdatingCache() {
	run := true
	for run {
		time.Sleep(defaultUpdateInterval)
		pods, err := r.getter.GetPods(false)
		cacheTime := time.Now()
		if err != nil {
			continue
		}

		r.Lock()
		if time.Now().After(r.updatingThreadStopTime) {
			r.updating = false
			run = false
		}
		r.pods = pods
		r.cacheTime = cacheTime
		r.Unlock()
	}
}
