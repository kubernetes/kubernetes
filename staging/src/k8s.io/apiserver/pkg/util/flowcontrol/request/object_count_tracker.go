/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"errors"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	// type deletion (it applies mostly to CRD) is not a very frequent
	// operation so we can afford to prune the cache at a large interval.
	// at the same time, we also want to make sure that the scalability
	// tests hit this code path.
	pruneInterval = 1 * time.Hour

	// the storage layer polls for object count at every 1m interval, we will allow
	// up to 2-3 transient failures to get the latest count for a given resource.
	staleTolerationThreshold = 3 * time.Minute
)

var (
	// ObjectCountNotFoundErr is returned when the object count for
	// a given resource is not being tracked.
	ObjectCountNotFoundErr = errors.New("object count not found for the given resource")

	// ObjectCountStaleErr is returned when the object count for a
	// given resource has gone stale due to transient failures.
	ObjectCountStaleErr = errors.New("object count has gone stale for the given resource")
)

// StorageObjectCountTracker is an interface that is used to keep track of
// of the total number of objects for each resource.
// {group}.{resource} is used as the key name to update and retrieve
// the total number of objects for a given resource.
type StorageObjectCountTracker interface {
	// Set is invoked to update the current number of total
	// objects for the given resource
	Set(string, storage.Stats)

	// Get returns the total number of objects for the given resource.
	// The following errors are returned:
	//  - if the count has gone stale for a given resource due to transient
	//    failures ObjectCountStaleErr is returned.
	//  - if the given resource is not being tracked then
	//    ObjectCountNotFoundErr is returned.
	Get(string) (storage.Stats, error)

	// RunUntil starts all the necessary maintenance.
	RunUntil(stopCh <-chan struct{})
}

// NewStorageObjectCountTracker returns an instance of
// StorageObjectCountTracker interface that can be used to
// keep track of the total number of objects for each resource.
func NewStorageObjectCountTracker() StorageObjectCountTracker {
	return &objectCountTracker{
		clock:  &clock.RealClock{},
		counts: map[string]*timestampedStats{},
	}
}

// timestampedStats stores the count of a given resource with a last updated
// timestamp so we can prune it after it goes stale for certain threshold.
type timestampedStats struct {
	storage.Stats
	lastUpdatedAt time.Time
}

// objectCountTracker implements StorageObjectCountTracker with
// reader/writer mutual exclusion lock.
type objectCountTracker struct {
	clock clock.PassiveClock

	lock   sync.RWMutex
	counts map[string]*timestampedStats
}

func (t *objectCountTracker) Set(groupResource string, stats storage.Stats) {
	now := t.clock.Now()

	// lock for writing
	t.lock.Lock()
	defer t.lock.Unlock()

	if item, ok := t.counts[groupResource]; ok {
		item.Stats = stats
		item.lastUpdatedAt = now
		return
	}

	t.counts[groupResource] = &timestampedStats{
		Stats:         stats,
		lastUpdatedAt: now,
	}
}

func (t *objectCountTracker) Get(groupResource string) (storage.Stats, error) {
	staleThreshold := t.clock.Now().Add(-staleTolerationThreshold)

	t.lock.RLock()
	defer t.lock.RUnlock()

	if item, ok := t.counts[groupResource]; ok {
		if item.lastUpdatedAt.Before(staleThreshold) {
			return item.Stats, ObjectCountStaleErr
		}
		return item.Stats, nil
	}
	return storage.Stats{}, ObjectCountNotFoundErr
}

// RunUntil runs all the necessary maintenance.
func (t *objectCountTracker) RunUntil(stopCh <-chan struct{}) {
	wait.PollUntil(
		pruneInterval,
		func() (bool, error) {
			// always prune at every pruneInterval
			return false, t.prune(pruneInterval)
		}, stopCh)
	klog.InfoS("StorageObjectCountTracker pruner is exiting")
}

func (t *objectCountTracker) prune(threshold time.Duration) error {
	oldestLastUpdatedAtAllowed := t.clock.Now().Add(-threshold)

	// lock for writing
	t.lock.Lock()
	defer t.lock.Unlock()

	for groupResource, count := range t.counts {
		if count.lastUpdatedAt.After(oldestLastUpdatedAtAllowed) {
			continue
		}
		delete(t.counts, groupResource)
	}

	return nil
}
