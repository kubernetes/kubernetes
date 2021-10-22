/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cache

import (
	"strconv"
	"sync"
	"testing"
	"time"
)

const (
	testCacheTimeout = 100 * time.Millisecond
)

func (c *TimeoutCache) getForTesting(key interface{}) (*cacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.cache[key]
	return r, ok
}

// TestCacheExpire attempts to add an entry to the cache and verifies that it
// was added successfully. It then makes sure that on timeout, it's removed and
// the associated callback is called.
func TestCacheExpire(t *testing.T) {
	const k, v = 1, "1"
	c := NewTimeoutCache(testCacheTimeout)

	callbackChan := make(chan struct{})
	c.Add(k, v, func() { close(callbackChan) })

	if gotV, ok := c.getForTesting(k); !ok || gotV.item != v {
		t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", gotV.item, ok, v, true)
	}

	select {
	case <-callbackChan:
	case <-time.After(testCacheTimeout * 2):
		t.Fatalf("timeout waiting for callback")
	}

	if _, ok := c.getForTesting(k); ok {
		t.Fatalf("After Add(), after timeout, from cache got: _, %v, want _, %v", ok, false)
	}
}

// TestCacheRemove attempts to remove an existing entry from the cache and
// verifies that the entry is removed and the associated callback is not
// invoked.
func TestCacheRemove(t *testing.T) {
	const k, v = 1, "1"
	c := NewTimeoutCache(testCacheTimeout)

	callbackChan := make(chan struct{})
	c.Add(k, v, func() { close(callbackChan) })

	if got, ok := c.getForTesting(k); !ok || got.item != v {
		t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.item, ok, v, true)
	}

	time.Sleep(testCacheTimeout / 2)

	gotV, gotOK := c.Remove(k)
	if !gotOK || gotV != v {
		t.Fatalf("After Add(), before timeout, Remove() got: %v, %v, want %v, %v", gotV, gotOK, v, true)
	}

	if _, ok := c.getForTesting(k); ok {
		t.Fatalf("After Add(), before timeout, after Remove(), from cache got: _, %v, want _, %v", ok, false)
	}

	select {
	case <-callbackChan:
		t.Fatalf("unexpected callback after retrieve")
	case <-time.After(testCacheTimeout * 2):
	}
}

// TestCacheClearWithoutCallback attempts to clear all entries from the cache
// and verifies that the associated callbacks are not invoked.
func TestCacheClearWithoutCallback(t *testing.T) {
	var values []string
	const itemCount = 3
	for i := 0; i < itemCount; i++ {
		values = append(values, strconv.Itoa(i))
	}
	c := NewTimeoutCache(testCacheTimeout)

	done := make(chan struct{})
	defer close(done)
	callbackChan := make(chan struct{}, itemCount)

	for i, v := range values {
		callbackChanTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackChanTemp) })
		go func() {
			select {
			case <-callbackChanTemp:
				callbackChan <- struct{}{}
			case <-done:
			}
		}()
	}

	for i, v := range values {
		if got, ok := c.getForTesting(i); !ok || got.item != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.item, ok, v, true)
		}
	}

	time.Sleep(testCacheTimeout / 2)
	c.Clear(false)

	for i := range values {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Remove(), from cache got: _, %v, want _, %v", ok, false)
		}
	}

	select {
	case <-callbackChan:
		t.Fatalf("unexpected callback after Clear")
	case <-time.After(testCacheTimeout * 2):
	}
}

// TestCacheClearWithCallback attempts to clear all entries from the cache and
// verifies that the associated callbacks are invoked.
func TestCacheClearWithCallback(t *testing.T) {
	var values []string
	const itemCount = 3
	for i := 0; i < itemCount; i++ {
		values = append(values, strconv.Itoa(i))
	}
	c := NewTimeoutCache(time.Hour)

	testDone := make(chan struct{})
	defer close(testDone)

	var wg sync.WaitGroup
	wg.Add(itemCount)
	for i, v := range values {
		callbackChanTemp := make(chan struct{})
		c.Add(i, v, func() { close(callbackChanTemp) })
		go func() {
			defer wg.Done()
			select {
			case <-callbackChanTemp:
			case <-testDone:
			}
		}()
	}

	allGoroutineDone := make(chan struct{}, itemCount)
	go func() {
		wg.Wait()
		close(allGoroutineDone)
	}()

	for i, v := range values {
		if got, ok := c.getForTesting(i); !ok || got.item != v {
			t.Fatalf("After Add(), before timeout, from cache got: %v, %v, want %v, %v", got.item, ok, v, true)
		}
	}

	time.Sleep(testCacheTimeout / 2)
	c.Clear(true)

	for i := range values {
		if _, ok := c.getForTesting(i); ok {
			t.Fatalf("After Add(), before timeout, after Remove(), from cache got: _, %v, want _, %v", ok, false)
		}
	}

	select {
	case <-allGoroutineDone:
	case <-time.After(testCacheTimeout * 2):
		t.Fatalf("timeout waiting for all callbacks")
	}
}

// TestCacheRetrieveTimeoutRace simulates the case where an entry's timer fires
// around the same time that Remove() is called for it. It verifies that there
// is no deadlock.
func TestCacheRetrieveTimeoutRace(t *testing.T) {
	c := NewTimeoutCache(time.Nanosecond)

	done := make(chan struct{})
	go func() {
		for i := 0; i < 1000; i++ {
			// Add starts a timer with 1 ns timeout, then remove will race
			// with the timer.
			c.Add(i, strconv.Itoa(i), func() {})
			c.Remove(i)
		}
		close(done)
	}()

	select {
	case <-time.After(time.Second):
		t.Fatalf("Test didn't finish within 1 second. Deadlock")
	case <-done:
	}
}
