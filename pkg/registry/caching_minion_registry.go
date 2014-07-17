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

package registry

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

type CachingMinionRegistry struct {
	delegate   MinionRegistry
	ttl        time.Duration
	minions    []string
	lastUpdate int64
	lock       sync.RWMutex
	clock      Clock
}

func NewCachingMinionRegistry(delegate MinionRegistry, ttl time.Duration) (MinionRegistry, error) {
	list, err := delegate.List()
	if err != nil {
		return nil, err
	}
	return &CachingMinionRegistry{
		delegate:   delegate,
		ttl:        ttl,
		minions:    list,
		lastUpdate: time.Now().Unix(),
		clock:      SystemClock{},
	}, nil
}

func (c *CachingMinionRegistry) List() ([]string, error) {
	if c.expired() {
		err := c.refresh(false)
		if err != nil {
			return c.minions, err
		}
	}
	return c.minions, nil
}

func (c *CachingMinionRegistry) Insert(minion string) error {
	err := c.delegate.Insert(minion)
	if err != nil {
		return err
	}
	return c.refresh(true)
}

func (c *CachingMinionRegistry) Delete(minion string) error {
	err := c.delegate.Delete(minion)
	if err != nil {
		return err
	}
	return c.refresh(true)
}

func (c *CachingMinionRegistry) Contains(minion string) (bool, error) {
	if c.expired() {
		err := c.refresh(false)
		if err != nil {
			return false, err
		}
	}

	// block updates in the middle of a contains.
	c.lock.RLock()
	defer c.lock.RUnlock()
	for _, name := range c.minions {
		if name == minion {
			return true, nil
		}
	}
	return false, nil
}

// refresh updates the current store.  It double checks expired under lock with the assumption
// of optimistic concurrency with the other functions.
func (c *CachingMinionRegistry) refresh(force bool) error {
	c.lock.Lock()
	defer c.lock.Unlock()
	if force || c.expired() {
		var err error
		c.minions, err = c.delegate.List()
		time := c.clock.Now()
		atomic.SwapInt64(&c.lastUpdate, time.Unix())
		return err
	}
	return nil
}

func (c *CachingMinionRegistry) expired() bool {
	var unix int64
	atomic.SwapInt64(&unix, c.lastUpdate)
	return c.clock.Now().Sub(time.Unix(c.lastUpdate, 0)) > c.ttl
}
