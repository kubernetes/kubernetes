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

// Package kmsv2 transforms values for storage at rest using a Envelope v2 provider
package kmsv2

import (
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/storage/value"
	testingclock "k8s.io/utils/clock/testing"
)

func TestSimpleCacheSetError(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	cache := newSimpleCache(fakeClock, time.Second, "providerName")

	tests := []struct {
		name        string
		key         []byte
		transformer value.Transformer
	}{
		{
			name:        "empty key",
			key:         []byte{},
			transformer: &envelopeTransformer{},
		},
		{
			name:        "nil transformer",
			key:         []byte("key"),
			transformer: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("The code did not panic")
				}
			}()
			cache.set(test.key, test.transformer)
		})
	}
}

func TestKeyFunc(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	cache := newSimpleCache(fakeClock, time.Second, "providerName")

	t.Run("AllocsPerRun test", func(t *testing.T) {
		key, err := generateKey(encryptedDEKSourceMaxSize) // simulate worst case EDEK
		if err != nil {
			t.Fatal(err)
		}

		f := func() {
			out := cache.keyFunc(key)
			if len(out) != sha256.Size {
				t.Errorf("Expected %d bytes, got %d", sha256.Size, len(out))
			}
		}

		// prime the key func
		var wg sync.WaitGroup
		for i := 0; i < 100; i++ {
			wg.Add(1)
			go func() {
				f()
				wg.Done()
			}()
		}
		wg.Wait()

		allocs := testing.AllocsPerRun(100, f)
		if allocs > 1 {
			t.Errorf("Expected 1 allocations, got %v", allocs)
		}
	})
}

func TestSimpleCache(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	cache := newSimpleCache(fakeClock, 5*time.Second, "providerName")
	transformer := &envelopeTransformer{}

	wg := sync.WaitGroup{}
	for i := 0; i < 10; i++ {
		k := fmt.Sprintf("key-%d", i)
		wg.Add(1)
		go func(key string) {
			defer wg.Done()
			cache.set([]byte(key), transformer)
		}(k)
	}
	wg.Wait()

	if cache.cache.Len() != 10 {
		t.Fatalf("Expected 10 items in the cache, got %v", cache.cache.Len())
	}

	for i := 0; i < 10; i++ {
		k := fmt.Sprintf("key-%d", i)
		if cache.get([]byte(k)) != transformer {
			t.Fatalf("Expected to get the transformer for key %v", k)
		}
	}

	// Wait for the cache to expire
	fakeClock.Step(6 * time.Second)

	// expired reads still work until GC runs on write
	for i := 0; i < 10; i++ {
		k := fmt.Sprintf("key-%d", i)
		if cache.get([]byte(k)) != transformer {
			t.Fatalf("Expected to get the transformer for key %v", k)
		}
	}

	// run GC by performing a write
	cache.set([]byte("some-other-unrelated-key"), transformer)

	for i := 0; i < 10; i++ {
		k := fmt.Sprintf("key-%d", i)
		if cache.get([]byte(k)) != nil {
			t.Fatalf("Expected to get nil for key %v", k)
		}
	}
}

func generateKey(length int) (key []byte, err error) {
	key = make([]byte, length)
	if _, err = rand.Read(key); err != nil {
		return nil, err
	}
	return key, nil
}

func TestMetrics(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	cache := newSimpleCache(fakeClock, 5*time.Second, "panda")
	var record sync.Map
	var cacheSize atomic.Uint64
	cache.recordCacheSize = func(providerName string, size int) {
		if providerName != "panda" {
			t.Errorf(`expected "panda" as provider name, got %q`, providerName)
		}
		if _, loaded := record.LoadOrStore(size, nil); loaded {
			t.Errorf("detected duplicated cache size metric for %d", size)
		}
		newSize := uint64(size)
		oldSize := cacheSize.Swap(newSize)
		if oldSize > newSize {
			t.Errorf("cache size decreased from %d to %d", oldSize, newSize)
		}
	}
	transformer := &envelopeTransformer{}

	want := sets.NewInt()
	startCh := make(chan struct{})
	wg := sync.WaitGroup{}
	for i := 0; i < 100; i++ {
		want.Insert(i + 1)
		k := fmt.Sprintf("key-%d", i)
		wg.Add(1)
		go func(key string) {
			defer wg.Done()
			<-startCh
			cache.set([]byte(key), transformer)
		}(k)
	}
	close(startCh)
	wg.Wait()

	got := sets.NewInt()
	record.Range(func(key, value any) bool {
		got.Insert(key.(int))
		if value != nil {
			t.Errorf("expected value to be nil but got %v", value)
		}
		return true
	})
	if !want.Equal(got) {
		t.Errorf("cache size entries missing values: %v", want.SymmetricDifference(got).List())
	}
}
