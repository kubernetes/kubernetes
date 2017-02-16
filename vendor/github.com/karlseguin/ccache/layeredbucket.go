package ccache

import (
	"sync"
	"time"
)

type layeredBucket struct {
	sync.RWMutex
	buckets map[string]*bucket
}

func (b *layeredBucket) get(primary, secondary string) *Item {
	bucket := b.getSecondaryBucket(primary)
	if bucket == nil {
		return nil
	}
	return bucket.get(secondary)
}

func (b *layeredBucket) getSecondaryBucket(primary string) *bucket {
	b.RLock()
	bucket, exists := b.buckets[primary]
	b.RUnlock()
	if exists == false {
		return nil
	}
	return bucket
}

func (b *layeredBucket) set(primary, secondary string, value interface{}, duration time.Duration) (*Item, *Item) {
	b.Lock()
	bkt, exists := b.buckets[primary]
	if exists == false {
		bkt = &bucket{lookup: make(map[string]*Item)}
		b.buckets[primary] = bkt
	}
	b.Unlock()
	item, existing := bkt.set(secondary, value, duration)
	item.group = primary
	return item, existing
}

func (b *layeredBucket) delete(primary, secondary string) *Item {
	b.RLock()
	bucket, exists := b.buckets[primary]
	b.RUnlock()
	if exists == false {
		return nil
	}
	return bucket.delete(secondary)
}

func (b *layeredBucket) deleteAll(primary string, deletables chan *Item) bool {
	b.RLock()
	bucket, exists := b.buckets[primary]
	b.RUnlock()
	if exists == false {
		return false
	}

	bucket.Lock()
	defer bucket.Unlock()

	if l := len(bucket.lookup); l == 0 {
		return false
	}
	for key, item := range bucket.lookup {
		delete(bucket.lookup, key)
		deletables <- item
	}
	return true
}

func (b *layeredBucket) clear() {
	b.Lock()
	defer b.Unlock()
	for _, bucket := range b.buckets {
		bucket.clear()
	}
	b.buckets = make(map[string]*bucket)
}
