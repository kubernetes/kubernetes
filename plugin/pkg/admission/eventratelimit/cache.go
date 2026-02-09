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

package eventratelimit

import (
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/utils/lru"
)

// cache is an interface for caching the limits of a particular type
type cache interface {
	// get the rate limiter associated with the specified key
	get(key interface{}) flowcontrol.RateLimiter
}

// singleCache is a cache that only stores a single, constant item
type singleCache struct {
	// the single rate limiter held by the cache
	rateLimiter flowcontrol.RateLimiter
}

func (c *singleCache) get(key interface{}) flowcontrol.RateLimiter {
	return c.rateLimiter
}

// lruCache is a least-recently-used cache
type lruCache struct {
	// factory to use to create new rate limiters
	rateLimiterFactory func() flowcontrol.RateLimiter
	// the actual LRU cache
	cache *lru.Cache
}

func (c *lruCache) get(key interface{}) flowcontrol.RateLimiter {
	value, found := c.cache.Get(key)
	if !found {
		rateLimter := c.rateLimiterFactory()
		c.cache.Add(key, rateLimter)
		return rateLimter
	}
	return value.(flowcontrol.RateLimiter)
}
