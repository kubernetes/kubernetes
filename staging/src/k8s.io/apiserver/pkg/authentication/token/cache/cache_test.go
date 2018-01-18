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
	"math/rand"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/authentication/user"

	"github.com/pborman/uuid"
)

func TestSimpleCache(t *testing.T) {
	testCache(newSimpleCache(4096, clock.RealClock{}), t)
}

func BenchmarkSimpleCache(b *testing.B) {
	benchmarkCache(newSimpleCache(4096, clock.RealClock{}), b)
}

func TestStripedCache(t *testing.T) {
	testCache(newStripedCache(32, fnvKeyFunc, func() cache { return newSimpleCache(128, clock.RealClock{}) }), t)
}

func BenchmarkStripedCache(b *testing.B) {
	benchmarkCache(newStripedCache(32, fnvKeyFunc, func() cache { return newSimpleCache(128, clock.RealClock{}) }), b)
}

func benchmarkCache(cache cache, b *testing.B) {
	keys := []string{}
	for i := 0; i < b.N; i++ {
		key := uuid.NewRandom().String()
		keys = append(keys, key)
	}

	b.ResetTimer()

	b.SetParallelism(500)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			key := keys[rand.Intn(b.N)]
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

	record1 := &cacheRecord{user: &user.DefaultInfo{Name: "bob"}}
	record2 := &cacheRecord{user: &user.DefaultInfo{Name: "alice"}}

	// when empty, record is stored
	cache.set("foo", record1, time.Hour)
	if result, ok := cache.get("foo"); !ok || result != record1 {
		t.Errorf("Expected %#v, true, got %#v, %v", record1, ok)
	}

	// newer record overrides
	cache.set("foo", record2, time.Hour)
	if result, ok := cache.get("foo"); !ok || result != record2 {
		t.Errorf("Expected %#v, true, got %#v, %v", record2, ok)
	}

	// removing the current value removes
	cache.remove("foo")
	if result, ok := cache.get("foo"); ok || result != nil {
		t.Errorf("Expected null, false, got %#v, %v", result, ok)
	}
}
