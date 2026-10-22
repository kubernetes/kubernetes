/*
Copyright 2023 The Kubernetes Authors.

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

package aes

import (
	"bytes"
	"fmt"
	"testing"
	"time"

	clocktesting "k8s.io/utils/clock/testing"
)

type dataString string

func (d dataString) AuthenticatedData() []byte { return []byte(d) }

func Test_simpleCache(t *testing.T) {
	info1 := []byte{1}
	info2 := []byte{2}
	key1 := dataString("1")
	key2 := dataString("2")
	twi1 := &transformerWithInfo{info: info1}
	twi2 := &transformerWithInfo{info: info2}

	tests := []struct {
		name string
		test func(*testing.T, *simpleCache, *clocktesting.FakeClock)
	}{
		{
			name: "get from empty",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				got := cache.get(info1, key1)
				twiPtrEquals(t, nil, got)
				cacheLenEquals(t, cache, 0)
			},
		},
		{
			name: "get after set",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				cache.set(key1, twi1)
				got := cache.get(info1, key1)
				twiPtrEquals(t, twi1, got)
				cacheLenEquals(t, cache, 1)
			},
		},
		{
			name: "get after set but with different info",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				cache.set(key1, twi1)
				got := cache.get(info2, key1)
				twiPtrEquals(t, nil, got)
				cacheLenEquals(t, cache, 1)
			},
		},
		{
			name: "expired get after set",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				cache.set(key1, twi1)
				clock.Step(time.Hour)
				got := cache.get(info1, key1)
				twiPtrEquals(t, twi1, got)
				cacheLenEquals(t, cache, 1)
			},
		},
		{
			name: "expired get after GC",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				cache.set(key1, twi1)
				clock.Step(time.Hour)
				cacheLenEquals(t, cache, 1)
				cache.set(key2, twi2) // unrelated set to make GC run
				got := cache.get(info1, key1)
				twiPtrEquals(t, nil, got)
				cacheLenEquals(t, cache, 1)
			},
		},
		{
			name: "multiple sets for same key",
			test: func(t *testing.T, cache *simpleCache, clock *clocktesting.FakeClock) {
				cache.set(key1, twi1)
				cacheLenEquals(t, cache, 1)
				cache.set(key1, twi2)
				cacheLenEquals(t, cache, 1)

				got11 := cache.get(info1, key1)
				twiPtrEquals(t, nil, got11)

				got21 := cache.get(info2, key1)
				twiPtrEquals(t, twi2, got21)

				got12 := cache.get(info1, key2)
				twiPtrEquals(t, nil, got12)

				got22 := cache.get(info2, key2)
				twiPtrEquals(t, nil, got22)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			clock := clocktesting.NewFakeClock(time.Now())
			// One shard: these cases exercise GC across keys, which is per shard.
			cache := newSimpleCacheWithShards(clock, 10*time.Second, 1)
			tt.test(t, cache, clock)
		})
	}
}

func Test_simpleCacheSharded(t *testing.T) {
	clock := clocktesting.NewFakeClock(time.Now())
	cache := newSimpleCache(clock, 10*time.Second)
	if len(cache.shards) != cacheShards {
		t.Fatalf("want %d shards, got %d", cacheShards, len(cache.shards))
	}
	twis := map[string]*transformerWithInfo{}
	for i := 0; i < 1000; i++ {
		k := fmt.Sprintf("/registry/events/ns/e-%d", i)
		twis[k] = &transformerWithInfo{info: []byte(k)}
		cache.set(dataString(k), twis[k])
	}
	cacheLenEquals(t, cache, 1000)
	used := 0
	for _, s := range cache.shards {
		if s.Len() > 0 {
			used++
		}
	}
	if used < cacheShards/2 {
		t.Errorf("keys landed in only %d of %d shards", used, cacheShards)
	}
	for k, want := range twis {
		twiPtrEquals(t, want, cache.get(want.info, dataString(k)))
	}
	// Re-setting the same key and info is a no-op: the original transformer stays.
	k := "/registry/events/ns/e-1"
	cache.set(dataString(k), &transformerWithInfo{info: []byte(k)})
	twiPtrEquals(t, twis[k], cache.get([]byte(k), dataString(k)))
	cacheLenEquals(t, cache, 1000)
}

func twiPtrEquals(t *testing.T, want, got *transformerWithInfo) {
	t.Helper()

	if want != got {
		t.Errorf("transformerWithInfo structs are not pointer equivalent")
	}
}

func cacheLenEquals(t *testing.T, cache *simpleCache, want int) {
	t.Helper()

	if got := cache.Len(); want != got {
		t.Errorf("unexpected cache len: want %d, got %d", want, got)
	}
}

// Memory stays bounded: after traffic stops and TTL passes, one set per shard must reclaim
// everything else (an unsharded cache needs one set anywhere; a sharded one needs one per shard).
func Test_simpleCacheShardedGCBoundsMemory(t *testing.T) {
	clock := clocktesting.NewFakeClock(time.Now())
	cache := newSimpleCache(clock, 10*time.Second)
	for i := 0; i < 1000; i++ {
		k := fmt.Sprintf("/registry/events/ns/e-%d", i)
		cache.set(dataString(k), &transformerWithInfo{info: []byte(k)})
	}
	if cache.Len() != 1000 {
		t.Fatalf("want 1000 entries, got %d", cache.Len())
	}
	clock.Step(time.Minute)
	// Expired entries are still served (AllowExpiredGet) until GC, as before sharding.
	k0 := "/registry/events/ns/e-0"
	if cache.get([]byte(k0), dataString(k0)) == nil {
		t.Fatalf("expired entry should still be readable before GC")
	}
	// Touch every shard with one fresh set.
	touched := map[int]bool{}
	for i := 0; len(touched) < cacheShards && i < 1_000_000; i++ {
		k := fmt.Sprintf("/fresh/%d", i)
		s := cache.shard(k)
		for j, sh := range cache.shards {
			if sh == s && !touched[j] {
				touched[j] = true
				cache.set(dataString(k), &transformerWithInfo{info: []byte(k)})
			}
		}
	}
	if got := cache.Len(); got != cacheShards {
		t.Fatalf("after one set per shard, want %d live entries, got %d", cacheShards, got)
	}
}

// Skipping a redundant set must never keep a DIFFERENT info: a set with new info always replaces.
func Test_simpleCacheSkipsOnlyIdenticalInfo(t *testing.T) {
	clock := clocktesting.NewFakeClock(time.Now())
	cache := newSimpleCache(clock, 10*time.Second)
	k := dataString("/registry/events/ns/e")
	v1 := &transformerWithInfo{info: []byte("version-1-xxxxxxxxxxxxxxxxxxxxxx")}
	v2 := &transformerWithInfo{info: []byte("version-2-xxxxxxxxxxxxxxxxxxxxxx")}
	cache.set(k, v1)
	cache.set(k, v2)
	if cache.get(v1.info, k) != nil {
		t.Fatalf("old version still served after a newer set")
	}
	if cache.get(v2.info, k) != v2 {
		t.Fatalf("new version not served")
	}
	// Same info, different pointer: skipped, original kept (equivalent key material).
	v2b := &transformerWithInfo{info: bytes.Clone(v2.info)}
	cache.set(k, v2b)
	if cache.get(v2.info, k) != v2 {
		t.Fatalf("redundant set should be skipped and keep the first transformer")
	}
	// Info that is a prefix of another must not match.
	short := &transformerWithInfo{info: v2.info[:16]}
	if cache.get(short.info, k) != nil {
		t.Fatalf("prefix info matched")
	}
}
