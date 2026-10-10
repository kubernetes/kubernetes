/*
Copyright The Kubernetes Authors.

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

package nodememo

import "sync"

// MemoLRU bounds how many distinct keys keep a NodeMemo. A memo is only worth its bookkeeping while
// the same key comes back, and a cluster schedules pods of many different shapes, so each caller
// keeps at most size of them and the least recently used one is dropped whole - its aggregate goes
// with it, and the next pod of that shape pays one cold pass.
//
// Entries are only ever touched from the scheduling goroutine of one profile, but the lock is kept
// so that a caller from another extension point, or a second profile sharing a plugin instance by
// mistake, cannot corrupt them.
type MemoLRU[K comparable, V any] struct {
	mu    sync.Mutex
	size  int
	order []K
	items map[K]V
}

// NewMemoLRU returns an LRU holding at most size entries. A size below one is treated as one, so
// that a caller cannot end up with an LRU that evicts the entry it just created.
func NewMemoLRU[K comparable, V any](size int) *MemoLRU[K, V] {
	if size < 1 {
		size = 1
	}
	return &MemoLRU[K, V]{size: size, items: make(map[K]V)}
}

// DefaultLRUSize is how many distinct keys a caller keeps a memo for unless it has an opinion of its
// own. A memo is only worth its bookkeeping while the same key comes back, so the bound is what keeps
// a cluster that schedules many different pod shapes from holding a per node contribution for every
// shape it has ever seen. Ten covers the shapes a rolling update of a handful of workloads produces at
// once; raising it trades memory - an entry holds one contribution per node that has any - for fewer
// cold passes on clusters with more concurrent shapes.
const DefaultLRUSize = 10

// GetOrCreate returns the entry of key, creating it with create when the key is new or was evicted.
// create runs under the LRU lock, so it must not call back into the same LRU.
func (c *MemoLRU[K, V]) GetOrCreate(key K, create func() V) V {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.items[key]; ok {
		for i, k := range c.order {
			if k == key {
				c.order = append(c.order[:i], c.order[i+1:]...)
				break
			}
		}
		c.order = append(c.order, key)
		return v
	}
	v := create()
	if len(c.order) >= c.size {
		evicted := c.order[0]
		delete(c.items, evicted)
		c.order = c.order[1:]
	}
	c.items[key] = v
	c.order = append(c.order, key)
	return v
}

// Get returns the entry of key without creating one and reports whether it is there. It does not
// count as a use for the eviction order: it exists for callers that want to inspect an entry a pass
// has already created, which in practice means tests asserting on the memo counters.
func (c *MemoLRU[K, V]) Get(key K) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.items[key]
	return v, ok
}

// Keys returns the keys that currently hold an entry, in no particular order. It exists for callers
// that want to look at what a pass left behind without knowing the key it was built from, which in
// practice means tests asserting on the memo counters.
func (c *MemoLRU[K, V]) Keys() []K {
	c.mu.Lock()
	defer c.mu.Unlock()
	keys := make([]K, 0, len(c.items))
	for key := range c.items {
		keys = append(keys, key)
	}
	return keys
}

// Len reports how many keys currently hold an entry.
func (c *MemoLRU[K, V]) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.items)
}
