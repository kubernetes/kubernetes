/*
Copyright 2022 The Kubernetes Authors.

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

// Package cache provides a cache mechanism based on etags to lazily
// build, and/or cache results from expensive operation such that those
// operations are not repeated unnecessarily. The operations can be
// created as a tree, and replaced dynamically as needed.
//
// # Dependencies and types of caches
//
// This package uses a source/transform/sink model of caches to build
// the dependency tree, and can be used as follows:
//   - [NewSource]: A source cache that recomputes the content every time.
//   - [NewStaticSource]: A source cache that always produces the
//     same content, it is only called once.
//   - [NewTransformer]: A cache that transforms data from one format to
//     another. It's only refreshed when the source changes.
//   - [NewMerger]: A cache that aggregates multiple caches into one.
//     It's only refreshed when the source changes.
//   - [Replaceable]: A cache adapter that can be atomically
//     replaced with a new one, and saves the previous results in case an
//     error pops-up.
//
// # Atomicity
//
// Most of the operations are not atomic/thread-safe, except for
// [Replaceable.Replace] which can be performed while the objects are
// being read. Specifically, `Get` methods are NOT thread-safe. Never
// call `Get()` without a lock on a multi-threaded environment, since
// it's usually performing updates to caches that will require write
// operations.
//
// # Etags
//
// Etags in this library is a cache version identifier. It doesn't
// necessarily strictly match to the semantics of http `etags`, but are
// somewhat inspired from it and function with the same principles.
// Hashing the content is a good way to guarantee that your function is
// never going to be called spuriously. In Kubernetes world, this could
// be a `resourceVersion`, this can be an actual etag, a hash, a UUID
// (if the cache always changes), or even a made-up string when the
// content of the cache never changes.
package cached

import (
	"fmt"
	"sync/atomic"
)

// Result is the content returned from a call to a cache. It can either
// be created with [NewResultOK] if the call was a success, or
// [NewResultErr] if the call resulted in an error.
type Result[T any] struct {
	Data T
	Etag string
	Err  error
}

// NewResultOK creates a new [Result] for a successful operation.
func NewResultOK[T any](data T, etag string) Result[T] {
	return Result[T]{
		Data: data,
		Etag: etag,
	}
}

// NewResultErr creates a new [Result] when an error has happened.
func NewResultErr[T any](err error) Result[T] {
	return Result[T]{
		Err: err,
	}
}

// Result can be treated as a [Data] if necessary.
func (r Result[T]) Get() Result[T] {
	return r
}

// Data is a cache that performs an action whose result data will be
// cached. It also returns an "etag" identifier to version the cache, so
// that the caller can know if they have the most recent version of the
// cache (and can decide to cache some operation based on that).
//
// The [NewMerger] and [NewTransformer] automatically handle
// that for you by checking if the etag is updated before calling the
// merging or transforming function.
type Data[T any] interface {
	// Returns the cached data, as well as an "etag" to identify the
	// version of the cache, or an error if something happened.
	//
	// # Important note
	//
	// This method is NEVER thread-safe, never assume it is OK to
	// call `Get()` without holding a proper mutex in a
	// multi-threaded environment, especially since `Get()` will
	// usually update the cache and perform write operations.
	Get() Result[T]
}

// T is the source type, V is the destination type.
type merger[K comparable, T, V any] struct {
	mergeFn      func(map[K]Result[T]) Result[V]
	caches       map[K]Data[T]
	cacheResults map[K]Result[T]
	result       Result[V]
}

// NewMerger creates a new merge cache, a cache that merges the result
// of other caches. The function only gets called if any of the
// dependency has changed.
//
// If any of the dependency returned an error before, or any of the
// dependency returned an error this time, or if the mergeFn failed
// before, then the function is reran.
//
// The caches and results are mapped by K so that associated data can be
// retrieved. The map of dependencies can not be modified after
// creation, and a new merger should be created (and probably replaced
// using a [Replaceable]).
//
// Note that this assumes there is no "partial" merge, the merge
// function will remerge all the dependencies together everytime. Since
// the list of dependencies is constant, there is no way to save some
// partial merge information either.
func NewMerger[K comparable, T, V any](mergeFn func(results map[K]Result[T]) Result[V], caches map[K]Data[T]) Data[V] {
	return &merger[K, T, V]{
		mergeFn: mergeFn,
		caches:  caches,
	}
}

func (c *merger[K, T, V]) prepareResults() map[K]Result[T] {
	cacheResults := make(map[K]Result[T], len(c.caches))
	for key, cache := range c.caches {
		cacheResults[key] = cache.Get()
	}
	return cacheResults
}

// Rerun if:
//   - The last run resulted in an error
//   - Any of the dependency previously returned an error
//   - Any of the dependency just returned an error
//   - Any of the dependency's etag changed
func (c *merger[K, T, V]) needsRunning(results map[K]Result[T]) bool {
	if c.cacheResults == nil {
		return true
	}
	if c.result.Err != nil {
		return true
	}
	if len(results) != len(c.cacheResults) {
		panic(fmt.Errorf("invalid number of results: %v (expected %v)", len(results), len(c.cacheResults)))
	}
	for key, oldResult := range c.cacheResults {
		newResult, ok := results[key]
		if !ok {
			panic(fmt.Errorf("unknown cache entry: %v", key))
		}

		if newResult.Etag != oldResult.Etag || newResult.Err != nil || oldResult.Err != nil {
			return true
		}
	}
	return false
}

func (c *merger[K, T, V]) Get() Result[V] {
	cacheResults := c.prepareResults()
	if c.needsRunning(cacheResults) {
		c.cacheResults = cacheResults
		c.result = c.mergeFn(c.cacheResults)
	}
	return c.result
}

type transformerCacheKeyType struct{}

// NewTransformer creates a new cache that transforms the result of
// another cache. The transformFn will only be called if the source
// cache has updated the output, otherwise, the cached result will be
// returned.
//
// If the dependency returned an error before, or it returns an error
// this time, or if the transformerFn failed before, the function is
// reran.
func NewTransformer[T, V any](transformerFn func(Result[T]) Result[V], source Data[T]) Data[V] {
	return NewMerger(func(caches map[transformerCacheKeyType]Result[T]) Result[V] {
		cache, ok := caches[transformerCacheKeyType{}]
		if len(caches) != 1 || !ok {
			panic(fmt.Errorf("invalid cache for transformer cache: %v", caches))
		}
		return transformerFn(cache)
	}, map[transformerCacheKeyType]Data[T]{
		{}: source,
	})
}

// NewSource creates a new cache that generates some data. This
// will always be called since we don't know the origin of the data and
// if it needs to be updated or not.
func NewSource[T any](sourceFn func() Result[T]) Data[T] {
	c := source[T](sourceFn)
	return &c
}

type source[T any] func() Result[T]

func (c *source[T]) Get() Result[T] {
	return (*c)()
}

// NewStaticSource creates a new cache that always generates the
// same data. This will only be called once (lazily).
func NewStaticSource[T any](staticFn func() Result[T]) Data[T] {
	return &static[T]{
		fn: staticFn,
	}
}

type static[T any] struct {
	fn     func() Result[T]
	result *Result[T]
}

func (c *static[T]) Get() Result[T] {
	if c.result == nil {
		result := c.fn()
		c.result = &result
	}
	return *c.result
}

// Replaceable is a cache that carries the result even when the
// cache is replaced. The cache can be replaced atomically (without any
// lock held). This is the type that should typically be stored in
// structs.
type Replaceable[T any] struct {
	cache  atomic.Pointer[Data[T]]
	result *Result[T]
}

// Get retrieves the data from the underlying source. [Replaceable]
// implements the [Data] interface itself. This is a pass-through
// that calls the most recent underlying cache. If the cache fails but
// previously had returned a success, that success will be returned
// instead. If the cache fails but we never returned a success, that
// failure is returned.
//
// # Important note
//
// As all implementations of Get, this implementation is NOT
// thread-safe. Please properly lock a mutex before calling this method
// if you are in a multi-threaded environment, since this method will
// update the cache and perform write operations.
func (c *Replaceable[T]) Get() Result[T] {
	result := (*c.cache.Load()).Get()
	if result.Err != nil && c.result != nil && c.result.Err == nil {
		return *c.result
	}
	c.result = &result
	return *c.result
}

// Replace changes the cache in a thread-safe way.
func (c *Replaceable[T]) Replace(cache Data[T]) {
	c.cache.Swap(&cache)
}
