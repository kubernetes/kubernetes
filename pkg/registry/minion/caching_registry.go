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

package minion

import (
	"sync"
	"sync/atomic"
	"time"
)

type Clock interface {
	Now() time.Time
}

type SystemClock struct{}

func (SystemClock) Now() time.Time {
	return time.Now()
}

type CachingRegistry struct {
	delegate   Registry
	ttl        time.Duration
	minions    []string
	lastUpdate int64
	lock       sync.RWMutex
	clock      Clock
}

func NewCachingRegistry(delegate Registry, ttl time.Duration) (Registry, error) {
	list, err := delegate.List()
	if err != nil {
		return nil, err
	}
	return &CachingRegistry{
		delegate:   delegate,
		ttl:        ttl,
		minions:    list,
		lastUpdate: time.Now().Unix(),
		clock:      SystemClock{},
	}, nil
}

func (r *CachingRegistry) Contains(minion string) (bool, error) {
	if r.expired() {
		if err := r.refresh(false); err != nil {
			return false, err
		}
	}
	// block updates in the middle of a contains.
	r.lock.RLock()
	defer r.lock.RUnlock()
	for _, name := range r.minions {
		if name == minion {
			return true, nil
		}
	}
	return false, nil
}

func (r *CachingRegistry) Delete(minion string) error {
	if err := r.delegate.Delete(minion); err != nil {
		return err
	}
	return r.refresh(true)
}

func (r *CachingRegistry) Insert(minion string) error {
	if err := r.delegate.Insert(minion); err != nil {
		return err
	}
	return r.refresh(true)
}

func (r *CachingRegistry) List() ([]string, error) {
	if r.expired() {
		if err := r.refresh(false); err != nil {
			return r.minions, err
		}
	}
	return r.minions, nil
}

func (r *CachingRegistry) expired() bool {
	var unix int64
	atomic.SwapInt64(&unix, r.lastUpdate)
	return r.clock.Now().Sub(time.Unix(r.lastUpdate, 0)) > r.ttl
}

// refresh updates the current store.  It double checks expired under lock with the assumption
// of optimistic concurrency with the other functions.
func (r *CachingRegistry) refresh(force bool) error {
	r.lock.Lock()
	defer r.lock.Unlock()
	if force || r.expired() {
		var err error
		r.minions, err = r.delegate.List()
		time := r.clock.Now()
		atomic.SwapInt64(&r.lastUpdate, time.Unix())
		return err
	}
	return nil
}
