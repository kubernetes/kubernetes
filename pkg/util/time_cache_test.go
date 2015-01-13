/*
Copyright 2015 Google Inc. All rights reserved.

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

package util

import (
	"sync"
	"testing"
	"time"

	fuzz "github.com/google/gofuzz"
)

func TestCacheExpire(t *testing.T) {
	calls := map[string]int{}
	ff := func(key string) T { calls[key]++; return key }
	clock := &FakeClock{time.Now()}

	c := NewTimeCache(clock, 60*time.Second, ff)

	c.Get("foo")
	c.Get("bar")
	// This call should hit the cache, so we expect no additional calls
	c.Get("foo")
	// Advance the clock, this call should miss the cache, so expect one more call.
	clock.Time = clock.Time.Add(61 * time.Second)
	c.Get("foo")
	c.Get("bar")

	if e, a := 2, calls["foo"]; e != a {
		t.Errorf("Wrong number of calls for foo: wanted %v, got %v", e, a)
	}
	if e, a := 2, calls["bar"]; e != a {
		t.Errorf("Wrong number of calls for bar: wanted %v, got %v", e, a)
	}
}

func TestCacheNotExpire(t *testing.T) {
	calls := map[string]int{}
	ff := func(key string) T { calls[key]++; return key }
	clock := &FakeClock{time.Now()}

	c := NewTimeCache(clock, 60*time.Second, ff)

	c.Get("foo")
	// This call should hit the cache, so we expect no additional calls to the cloud
	clock.Time = clock.Time.Add(60 * time.Second)
	c.Get("foo")

	if e, a := 1, calls["foo"]; e != a {
		t.Errorf("Wrong number of calls for foo: wanted %v, got %v", e, a)
	}
}

func TestCacheParallel(t *testing.T) {
	ff := func(key string) T { time.Sleep(time.Second); return key }
	clock := &FakeClock{time.Now()}
	c := NewTimeCache(clock, 60*time.Second, ff)

	// Make some keys
	keys := []string{}
	fuzz.New().NilChance(0).NumElements(50, 50).Fuzz(&keys)

	// If we have high parallelism, this will take only a second.
	var wg sync.WaitGroup
	wg.Add(len(keys))
	for _, key := range keys {
		go func(key string) {
			c.Get(key)
			wg.Done()
		}(key)
	}
	wg.Wait()
}

func TestCacheParallelOneCall(t *testing.T) {
	calls := 0
	var callLock sync.Mutex
	ff := func(key string) T {
		time.Sleep(time.Second)
		callLock.Lock()
		defer callLock.Unlock()
		calls++
		return key
	}
	clock := &FakeClock{time.Now()}
	c := NewTimeCache(clock, 60*time.Second, ff)

	// If we have high parallelism, this will take only a second.
	var wg sync.WaitGroup
	wg.Add(50)
	for i := 0; i < 50; i++ {
		go func(key string) {
			c.Get(key)
			wg.Done()
		}("aoeu")
	}
	wg.Wait()

	// And if we wait for existing calls, we should have only one call.
	if e, a := 1, calls; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}
