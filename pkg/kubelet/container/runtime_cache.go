/*
Copyright 2015 The Kubernetes Authors.

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

//go:generate mockgen -source=runtime_cache.go  -destination=testing/mock_runtime_cache.go -package=testing RuntimeCache
package container

import (
	"context"
	"sync"
	"time"
)

// RuntimeCache is in interface for obtaining cached Pods.
type RuntimeCache interface {
	GetPods(context.Context) ([]*Pod, error)
	ForceUpdateIfOlder(context.Context, time.Time) error
}

type podsGetter interface {
	GetPods(context.Context, bool) ([]*Pod, error)
}

// NewRuntimeCache creates a container runtime cache.
func NewRuntimeCache(getter podsGetter, cachePeriod time.Duration) (RuntimeCache, error) {
	return &runtimeCache{
		getter:      getter,
		cachePeriod: cachePeriod,
	}, nil
}

// runtimeCache caches a list of pods. It records a timestamp (cacheTime) right
// before updating the pods, so the timestamp is at most as new as the pods
// (and can be slightly older). The timestamp always moves forward. Callers are
// expected not to modify the pods returned from GetPods.
type runtimeCache struct {
	sync.Mutex
	// The underlying container runtime used to update the cache.
	getter podsGetter
	// The interval after which the cache should be refreshed.
	cachePeriod time.Duration
	// Last time when cache was updated.
	cacheTime time.Time
	// The content of the cache.
	pods []*Pod
}

// GetPods returns the cached pods if they are not outdated; otherwise, it
// retrieves the latest pods and return them.
func (r *runtimeCache) GetPods(ctx context.Context) ([]*Pod, error) {
	r.Lock()
	defer r.Unlock()
	if time.Since(r.cacheTime) > r.cachePeriod {
		if err := r.updateCache(ctx); err != nil {
			return nil, err
		}
	}
	return r.pods, nil
}

func (r *runtimeCache) ForceUpdateIfOlder(ctx context.Context, minExpectedCacheTime time.Time) error {
	r.Lock()
	defer r.Unlock()
	if r.cacheTime.Before(minExpectedCacheTime) {
		return r.updateCache(ctx)
	}
	return nil
}

func (r *runtimeCache) updateCache(ctx context.Context) error {
	pods, timestamp, err := r.getPodsWithTimestamp(ctx)
	if err != nil {
		return err
	}
	r.pods, r.cacheTime = pods, timestamp
	return nil
}

// getPodsWithTimestamp records a timestamp and retrieves pods from the getter.
func (r *runtimeCache) getPodsWithTimestamp(ctx context.Context) ([]*Pod, time.Time, error) {
	// Always record the timestamp before getting the pods to avoid stale pods.
	timestamp := time.Now()
	pods, err := r.getter.GetPods(ctx, false)
	return pods, timestamp, err
}
