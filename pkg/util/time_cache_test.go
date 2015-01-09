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
	"testing"
	"time"
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
