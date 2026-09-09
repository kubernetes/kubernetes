/*
Copyright 2017 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/google/uuid"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/utils/clock"
)

func TestSimpleCache(t *testing.T) {
	testCache(newSimpleCache(clock.RealClock{}), t)
}

// Note: the performance profile of this benchmark may not match that in the production.
// When making change to SimpleCache, run test with and without concurrency to better understand the impact.
// This is a tool to test and measure high concurrency of the cache in isolation and not to the Kubernetes usage of the Cache.
func BenchmarkCacheContentions(b *testing.B) {
	for _, numKeys := range []int{1 << 8, 1 << 12, 1 << 16} {
		b.Run(fmt.Sprintf("Simple/keys=%d", numKeys), func(b *testing.B) {
			benchmarkCache(newSimpleCache(clock.RealClock{}), b, numKeys)
		})
		b.Run(fmt.Sprintf("Striped/keys=%d", numKeys), func(b *testing.B) {
			benchmarkCache(newStripedCache(32, fnvHashFunc, func() cache { return newSimpleCache(clock.RealClock{}) }), b, numKeys)
		})
	}
}

func TestStripedCache(t *testing.T) {
	testCache(newStripedCache(32, fnvHashFunc, func() cache { return newSimpleCache(clock.RealClock{}) }), t)
}

func benchmarkCache(cache cache, b *testing.B, numKeys int) {
	keys := []string{}
	for i := 0; i < numKeys; i++ {
		key := uuid.New().String()
		keys = append(keys, key)
	}

	b.ResetTimer()

	b.SetParallelism(500)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			key := keys[rand.Intn(numKeys)]
			_, ok := cache.get(key)
			if ok {
				cache.remove(key)
			} else {
				cache.set(key, &cacheRecord{}, time.Second)
			}
		}
	})
}

func testCache(cache cache, t *testing.T) {
	if result, ok := cache.get("foo"); ok || result != nil {
		t.Errorf("Expected null, false, got %#v, %v", result, ok)
	}

	record1 := &cacheRecord{resp: &authenticator.Response{User: &user.DefaultInfo{Name: "bob"}}}
	record2 := &cacheRecord{resp: &authenticator.Response{User: &user.DefaultInfo{Name: "alice"}}}

	// when empty, record is stored
	cache.set("foo", record1, time.Hour)
	if result, ok := cache.get("foo"); !ok || result != record1 {
		t.Errorf("Expected %#v, true, got %#v, %v", record1, result, ok)
	}

	// newer record overrides
	cache.set("foo", record2, time.Hour)
	if result, ok := cache.get("foo"); !ok || result != record2 {
		t.Errorf("Expected %#v, true, got %#v, %v", record2, result, ok)
	}

	// removing the current value removes
	cache.remove("foo")
	if result, ok := cache.get("foo"); ok || result != nil {
		t.Errorf("Expected null, false, got %#v, %v", result, ok)
	}
}
