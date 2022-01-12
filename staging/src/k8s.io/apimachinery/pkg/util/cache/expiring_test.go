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

	testingclock "k8s.io/utils/clock/testing"
)

func TestExpiringCache(t *testing.T) {
	cache := NewExpiring()

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
	fc := &testingclock.FakeClock{}
	c := NewExpiringWithClock(fc)

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

	e := c.cache["a"]
	e.generation++
	e.expiry = e.expiry.Add(1 * time.Second)
	c.cache["a"] = e

	fc.Step(1 * time.Second)
	if _, ok := c.Get("a"); !ok {
		t.Fatalf("we should have found a key")
	}
}

func TestGarbageCollection(t *testing.T) {
	fc := &testingclock.FakeClock{}

	type entry struct {
		key, val string
		ttl      time.Duration
	}

	tests := []struct {
		name string
		now  time.Time
		set  []entry
		want map[string]string
	}{
		{
			name: "two entries just set",
			now:  fc.Now().Add(0 * time.Second),
			set: []entry{
				{"a", "aa", 1 * time.Second},
				{"b", "bb", 2 * time.Second},
			},
			want: map[string]string{
				"a": "aa",
				"b": "bb",
			},
		},
		{
			name: "first entry expired now",
			now:  fc.Now().Add(1 * time.Second),
			set: []entry{
				{"a", "aa", 1 * time.Second},
				{"b", "bb", 2 * time.Second},
			},
			want: map[string]string{
				"b": "bb",
			},
		},
		{
			name: "first entry expired half a second ago",
			now:  fc.Now().Add(1500 * time.Millisecond),
			set: []entry{
				{"a", "aa", 1 * time.Second},
				{"b", "bb", 2 * time.Second},
			},
			want: map[string]string{
				"b": "bb",
			},
		},
		{
			name: "three entries weird order",
			now:  fc.Now().Add(1 * time.Second),
			set: []entry{
				{"c", "cc", 3 * time.Second},
				{"a", "aa", 1 * time.Second},
				{"b", "bb", 2 * time.Second},
			},
			want: map[string]string{
				"b": "bb",
				"c": "cc",
			},
		},
		{
			name: "expire multiple entries in one cycle",
			now:  fc.Now().Add(2500 * time.Millisecond),
			set: []entry{
				{"a", "aa", 1 * time.Second},
				{"b", "bb", 2 * time.Second},
				{"c", "cc", 3 * time.Second},
			},
			want: map[string]string{
				"c": "cc",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := NewExpiringWithClock(fc)
			for _, e := range test.set {
				c.Set(e.key, e.val, e.ttl)
			}

			c.gc(test.now)

			for k, want := range test.want {
				got, ok := c.Get(k)
				if !ok {
					t.Errorf("expected cache to have entry for key=%q but found none", k)
					continue
				}
				if got != want {
					t.Errorf("unexpected value for key=%q: got=%q, want=%q", k, got, want)
				}
			}
			if got, want := c.Len(), len(test.want); got != want {
				t.Errorf("unexpected cache size: got=%d, want=%d", got, want)
			}
		})
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
	const numKeys = 1 << 16
	cache := NewExpiring()

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

	keys := []string{}
	for i := 0; i < numKeys; i++ {
		key := uuid.New().String()
		keys = append(keys, key)
	}

	var wg sync.WaitGroup
	for i := 0; i < 256; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rand := rand.New(rand.NewSource(rand.Int63()))
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}
				key := keys[rand.Intn(numKeys)]
				if _, ok := cache.Get(key); !ok {
					cache.Set(key, struct{}{}, 50*time.Millisecond)
				}
			}
		}()
	}

	wg.Wait()

	// trigger a GC with a set and check the cache size.
	time.Sleep(60 * time.Millisecond)
	cache.Set("trigger", "gc", time.Second)
	if cache.Len() != 1 {
		t.Errorf("unexpected cache size: got=%d, want=1", cache.Len())
	}
}
