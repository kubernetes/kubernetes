/*
Copyright 2024 The Kubernetes Authors.

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

package cel

import (
	"sync"

	"k8s.io/utils/keymutex"
	"k8s.io/utils/lru"
)

// Cache is a thread-safe LRU cache for a compiled CEL expression.
type Cache struct {
	compileMutex keymutex.KeyMutex
	cacheMutex   sync.RWMutex
	cache        *lru.Cache
	compiler     *compiler
}

// NewCache creates a cache. The maximum number of entries determines
// how many entries are cached at most before dropping the oldest
// entry.
//
// The features are used to get a suitable compiler.
func NewCache(maxCacheEntries int, features Features) *Cache {
	return &Cache{
		compileMutex: keymutex.NewHashed(0),
		cache:        lru.New(maxCacheEntries),
		compiler:     GetCompiler(features),
	}
}

// GetOrCompile checks whether the cache already has a compilation result
// and returns that if available. Otherwise it compiles, stores successful
// results and returns the new result.
//
// Cost estimation is disabled.
func (c *Cache) GetOrCompile(expression string) CompilationResult {
	return c.getOrCompile(expression, "selector", Options{DisableCostEstimation: true})
}

// GetOrCompileDerivedAttribute checks whether the cache already has a
// compilation result for a derived attribute expression and returns that
// if available. Otherwise it compiles, stores successful results and returns
// the new result with DerivedAttribute compilation option enabled.
func (c *Cache) GetOrCompileDerivedAttribute(expression string) CompilationResult {
	return c.getOrCompile(expression, "derived", Options{
		DisableCostEstimation: true,
		DerivedAttribute:      true,
	})
}

func (c *Cache) getOrCompile(expression string, cacheScope string, opts Options) CompilationResult {
	cacheKey := cacheScope + ":" + expression
	// Compiling a CEL expression is expensive enough that it is cheaper
	// to lock a mutex than doing it several times in parallel.
	c.compileMutex.LockKey(cacheKey)
	//nolint:errcheck // Only returns an error for unknown keys, which isn't the case here.
	defer c.compileMutex.UnlockKey(cacheKey)

	cached := c.get(cacheKey)
	if cached != nil {
		return *cached
	}

	expr := c.compiler.CompileCELExpression(expression, opts)
	if expr.Error == nil {
		c.add(cacheKey, &expr)
	}
	return expr
}

func (c *Cache) add(expression string, expr *CompilationResult) {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()
	c.cache.Add(expression, expr)
}

func (c *Cache) get(expression string) *CompilationResult {
	c.cacheMutex.RLock()
	defer c.cacheMutex.RUnlock()
	expr, found := c.cache.Get(expression)
	if !found {
		return nil
	}
	return expr.(*CompilationResult)
}
