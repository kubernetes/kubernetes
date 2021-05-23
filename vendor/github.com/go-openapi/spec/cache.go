// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"sync"
)

// ResolutionCache a cache for resolving urls
type ResolutionCache interface {
	Get(string) (interface{}, bool)
	Set(string, interface{})
}

type simpleCache struct {
	lock  sync.RWMutex
	store map[string]interface{}
}

func (s *simpleCache) ShallowClone() ResolutionCache {
	store := make(map[string]interface{}, len(s.store))
	s.lock.RLock()
	for k, v := range s.store {
		store[k] = v
	}
	s.lock.RUnlock()

	return &simpleCache{
		store: store,
	}
}

// Get retrieves a cached URI
func (s *simpleCache) Get(uri string) (interface{}, bool) {
	s.lock.RLock()
	v, ok := s.store[uri]

	s.lock.RUnlock()
	return v, ok
}

// Set caches a URI
func (s *simpleCache) Set(uri string, data interface{}) {
	s.lock.Lock()
	s.store[uri] = data
	s.lock.Unlock()
}

var (
	// resCache is a package level cache for $ref resolution and expansion.
	// It is initialized lazily by methods that have the need for it: no
	// memory is allocated unless some expander methods are called.
	//
	// It is initialized with JSON schema and swagger schema,
	// which do not mutate during normal operations.
	//
	// All subsequent utilizations of this cache are produced from a shallow
	// clone of this initial version.
	resCache  *simpleCache
	onceCache sync.Once

	_ ResolutionCache = &simpleCache{}
)

// initResolutionCache initializes the URI resolution cache. To be wrapped in a sync.Once.Do call.
func initResolutionCache() {
	resCache = defaultResolutionCache()
}

func defaultResolutionCache() *simpleCache {
	return &simpleCache{store: map[string]interface{}{
		"http://swagger.io/v2/schema.json":       MustLoadSwagger20Schema(),
		"http://json-schema.org/draft-04/schema": MustLoadJSONSchemaDraft04(),
	}}
}

func cacheOrDefault(cache ResolutionCache) ResolutionCache {
	onceCache.Do(initResolutionCache)

	if cache != nil {
		return cache
	}

	// get a shallow clone of the base cache with swagger and json schema
	return resCache.ShallowClone()
}
