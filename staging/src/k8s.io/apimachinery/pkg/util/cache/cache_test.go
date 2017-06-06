/*
Copyright 2014 The Kubernetes Authors.

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
	"testing"
)

const (
	maxTestCacheSize int = shardsCount * 2
)

func ExpectEntry(t *testing.T, cache Cache, index uint64, expectedValue interface{}) bool {
	elem, found := cache.Get(index)
	if !found {
		t.Errorf("Expected to find entry with key %d", index)
		return false
	} else if elem != expectedValue {
		t.Errorf("Expected to find %v, got %v", expectedValue, elem)
		return false
	}
	return true
}

func TestBasic(t *testing.T) {
	cache := NewCache(maxTestCacheSize)
	cache.Add(1, "xxx")
	ExpectEntry(t, cache, 1, "xxx")
}

func TestOverflow(t *testing.T) {
	cache := NewCache(maxTestCacheSize)
	for i := 0; i < maxTestCacheSize+1; i++ {
		cache.Add(uint64(i), "xxx")
	}
	foundIndexes := make([]uint64, 0)
	for i := 0; i < maxTestCacheSize+1; i++ {
		_, found := cache.Get(uint64(i))
		if found {
			foundIndexes = append(foundIndexes, uint64(i))
		}
	}
	if len(foundIndexes) != maxTestCacheSize {
		t.Errorf("Expect to find %d elements, got %d %v", maxTestCacheSize, len(foundIndexes), foundIndexes)
	}
}

func TestOverwrite(t *testing.T) {
	cache := NewCache(maxTestCacheSize)
	cache.Add(1, "xxx")
	ExpectEntry(t, cache, 1, "xxx")
	cache.Add(1, "yyy")
	ExpectEntry(t, cache, 1, "yyy")
}

// TestEvict this test will fail sporatically depending on what add()
// selects for the randomKey to be evicted.  Ensure that randomKey
// is never the key we most recently added.  Since the chance of failure
// on each evict is 50%, if we do it 7 times, it should catch the problem
// if it exists >99% of the time.
func TestEvict(t *testing.T) {
	cache := NewCache(shardsCount)
	var found bool
	for retry := 0; retry < 7; retry++ {
		cache.Add(uint64(shardsCount), "xxx")
		found = ExpectEntry(t, cache, uint64(shardsCount), "xxx")
		if !found {
			break
		}
		cache.Add(0, "xxx")
		found = ExpectEntry(t, cache, 0, "xxx")
		if !found {
			break
		}
	}
}
