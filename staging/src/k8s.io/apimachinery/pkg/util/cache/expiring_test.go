/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
)

func TestExpiringCache(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cache := NewExpiring()
	go cache.Run(ctx)

	if result, ok := cache.Get("foo"); ok || result != nil {
		t.Errorf("Expected null, false, got %#v, %v", result, ok)
	}

	record1 := "bob"
	record2 := "alice"

	// when empty, record is stored
	cache.Set("foo", record1, time.Hour)
	if result, ok := cache.Get("foo"); !ok || result != record1 {
		t.Errorf("Expected %#v, true, got %#v, %v", record1, result, ok)
	}

	// newer record overrides
	cache.Set("foo", record2, time.Hour)
	if result, ok := cache.Get("foo"); !ok || result != record2 {
		t.Errorf("Expected %#v, true, got %#v, %v", record2, result, ok)
	}

	// delete the current value
	cache.Delete("foo")
	if result, ok := cache.Get("foo"); ok || result != nil {
		t.Errorf("Expected null, false, got %#v, %v", result, ok)
	}
}

func TestExpiration(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fc := &utilclock.FakeClock{}
	c := NewExpiringWithClock(fc)
	go c.Run(ctx)

	c.Set("a", "a", time.Second)

	fc.Step(500 * time.Millisecond)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("we should have found a key")
	}

	fc.Step(time.Second)
	if _, ok := c.Get("a"); ok {
		t.Fatalf("we should not have found a key")
	}

	c.Set("a", "a", time.Second)

	fc.Step(500 * time.Millisecond)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("we should have found a key")
	}

	// reset should restart the ttl
	c.Set("a", "a", time.Second)

	fc.Step(750 * time.Millisecond)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("we should have found a key")
	}

	// Simulate a race between a reset and cleanup. Assert that del doesn't
	// remove the key.
	c.Set("a", "a", time.Second)

	c.mu.Lock()
	e := c.cache["a"]
	e.generation++
	e.expiry = e.expiry.Add(1 * time.Second)
	c.cache["a"] = e
	c.mu.Unlock()

	fc.Step(1 * time.Second)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("we should have found a key")
	}
}

func BenchmarkExpiringCacheContention(b *testing.B) {
	b.Run("evict_probablility=100%", func(b *testing.B) {
		benchmarkExpiringCacheContention(b, 1)
	})
	b.Run("evict_probablility=10%", func(b *testing.B) {
		benchmarkExpiringCacheContention(b, 0.1)
	})
	b.Run("evict_probablility=1%", func(b *testing.B) {
		benchmarkExpiringCacheContention(b, 0.01)
	})
}

func benchmarkExpiringCacheContention(b *testing.B, prob float64) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	const numKeys = 1 << 16
	cache := NewExpiring()
	go cache.Run(ctx)

	keys := []string{}
	for i := 0; i < numKeys; i++ {
		key := uuid.New().String()
		keys = append(keys, key)
	}

	b.ResetTimer()

	b.SetParallelism(256)
	b.RunParallel(func(pb *testing.PB) {
		rand := rand.New(rand.NewSource(rand.Int63()))
		for pb.Next() {
			i := rand.Int31()
			key := keys[i%numKeys]
			_, ok := cache.Get(key)
			if ok {
				// compare lower bits of sampled i to decide whether we should evict.
				if rand.Float64() < prob {
					cache.Delete(key)
				}
			} else {
				cache.Set(key, struct{}{}, 50*time.Millisecond)
			}
		}
	})
}

func TestStressExpiringCache(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	const numKeys = 1 << 16
	cache := NewExpiring()
	go cache.Run(ctx)

	keys := []string{}
	for i := 0; i < numKeys; i++ {
		key := uuid.New().String()
		keys = append(keys, key)
	}

	var wg sync.WaitGroup
	for i := 0; i < 256; i++ {
		wg.Add(1)
		go func() {
			rand := rand.New(rand.NewSource(rand.Int63()))
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}
				key := keys[rand.Intn(numKeys)]
				if _, ok := cache.Get(key); !ok {
					cache.Set(key, struct{}{}, time.Second)
				}
			}
		}()
	}

	wg.Done()
}
