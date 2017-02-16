package ccache

import (
	"sync"
	"time"
)

type bucket struct {
	sync.RWMutex
	lookup map[string]*Item
}

func (b *bucket) get(key string) *Item {
	b.RLock()
	defer b.RUnlock()
	return b.lookup[key]
}

func (b *bucket) set(key string, value interface{}, duration time.Duration) (*Item, *Item) {
	expires := time.Now().Add(duration).UnixNano()
	item := newItem(key, value, expires)
	b.Lock()
	defer b.Unlock()
	existing := b.lookup[key]
	b.lookup[key] = item
	return item, existing
}

func (b *bucket) delete(key string) *Item {
	b.Lock()
	defer b.Unlock()
	item := b.lookup[key]
	delete(b.lookup, key)
	return item
}

func (b *bucket) clear() {
	b.Lock()
	defer b.Unlock()
	b.lookup = make(map[string]*Item)
}
