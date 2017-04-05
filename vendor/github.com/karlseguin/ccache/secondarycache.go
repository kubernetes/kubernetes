package ccache

import "time"

type SecondaryCache struct {
	bucket *bucket
	pCache *LayeredCache
}

// Get the secondary key.
// The semantics are the same as for LayeredCache.Get
func (s *SecondaryCache) Get(secondary string) *Item {
	return s.bucket.get(secondary)
}

// Set the secondary key to a value.
// The semantics are the same as for LayeredCache.Set
func (s *SecondaryCache) Set(secondary string, value interface{}, duration time.Duration) *Item {
	item, existing := s.bucket.set(secondary, value, duration)
	if existing != nil {
		s.pCache.deletables <- existing
	}
	s.pCache.promote(item)
	return item
}

// Fetch or set a secondary key.
// The semantics are the same as for LayeredCache.Fetch
func (s *SecondaryCache) Fetch(secondary string, duration time.Duration, fetch func() (interface{}, error)) (*Item, error) {
	item := s.Get(secondary)
	if item != nil {
		return item, nil
	}
	value, err := fetch()
	if err != nil {
		return nil, err
	}
	return s.Set(secondary, value, duration), nil
}

// Delete a secondary key.
// The semantics are the same as for LayeredCache.Delete
func (s *SecondaryCache) Delete(secondary string) bool {
	item := s.bucket.delete(secondary)
	if item != nil {
		s.pCache.deletables <- item
		return true
	}
	return false
}

// Replace a secondary key.
// The semantics are the same as for LayeredCache.Replace
func (s *SecondaryCache) Replace(secondary string, value interface{}) bool {
	item := s.Get(secondary)
	if item == nil {
		return false
	}
	s.Set(secondary, value, item.TTL())
	return true
}

// Track a secondary key.
// The semantics are the same as for LayeredCache.TrackingGet
func (c *SecondaryCache) TrackingGet(secondary string) TrackedItem {
	item := c.Get(secondary)
	if item == nil {
		return NilTracked
	}
	item.track()
	return item
}
