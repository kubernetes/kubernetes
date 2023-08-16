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
// All the operations in this module are thread-safe.
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
	"sync"
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
	Get() Result[T]
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
//
// Also note that Golang map iteration is not stable. If the mergeFn
// depends on the order iteration to be stable, it will need to
// implement its own sorting or iteration order.
func NewMerger[K comparable, T, V any](mergeFn func(results map[K]Result[T]) Result[V], caches map[K]Data[T]) Data[V] {
	listCaches := make([]Data[T], 0, len(caches))
	// maps from index to key
	indexes := make(map[int]K, len(caches))
	i := 0
	for k := range caches {
		listCaches = append(listCaches, caches[k])
		indexes[i] = k
		i++
	}

	return NewListMerger(func(results []Result[T]) Result[V] {
		if len(results) != len(indexes) {
			panic(fmt.Errorf("invalid result length %d, expected %d", len(results), len(indexes)))
		}
		m := make(map[K]Result[T], len(results))
		for i := range results {
			m[indexes[i]] = results[i]
		}
		return mergeFn(m)
	}, listCaches)
}

type listMerger[T, V any] struct {
	lock         sync.Mutex
	mergeFn      func([]Result[T]) Result[V]
	caches       []Data[T]
	cacheResults []Result[T]
	result       Result[V]
}

// NewListMerger creates a new merge cache that merges the results of
// other caches in list form. The function only gets called if any of
// the dependency has changed.
//
// The benefit of ListMerger over the basic Merger is that caches are
// stored in an ordered list so the order of the cache will be
// preserved in the order of the results passed to the mergeFn.
//
// If any of the dependency returned an error before, or any of the
// dependency returned an error this time, or if the mergeFn failed
// before, then the function is reran.
//
// Note that this assumes there is no "partial" merge, the merge
// function will remerge all the dependencies together everytime. Since
// the list of dependencies is constant, there is no way to save some
// partial merge information either.
func NewListMerger[T, V any](mergeFn func(results []Result[T]) Result[V], caches []Data[T]) Data[V] {
	return &listMerger[T, V]{
		mergeFn: mergeFn,
		caches:  caches,
	}
}

func (c *listMerger[T, V]) prepareResultsLocked() []Result[T] {
	cacheResults := make([]Result[T], len(c.caches))
	ch := make(chan struct {
		int
		Result[T]
	}, len(c.caches))
	for i := range c.caches {
		go func(index int) {
			ch <- struct {
				int
				Result[T]
			}{
				index,
				c.caches[index].Get(),
			}
		}(i)
	}
	for i := 0; i < len(c.caches); i++ {
		res := <-ch
		cacheResults[res.int] = res.Result
	}
	return cacheResults
}

func (c *listMerger[T, V]) needsRunningLocked(results []Result[T]) bool {
	if c.cacheResults == nil {
		return true
	}
	if c.result.Err != nil {
		return true
	}
	if len(results) != len(c.cacheResults) {
		panic(fmt.Errorf("invalid number of results: %v (expected %v)", len(results), len(c.cacheResults)))
	}
	for i, oldResult := range c.cacheResults {
		newResult := results[i]
		if newResult.Etag != oldResult.Etag || newResult.Err != nil || oldResult.Err != nil {
			return true
		}
	}
	return false
}

func (c *listMerger[T, V]) Get() Result[V] {
	c.lock.Lock()
	defer c.lock.Unlock()
	cacheResults := c.prepareResultsLocked()
	if c.needsRunningLocked(cacheResults) {
		c.cacheResults = cacheResults
		c.result = c.mergeFn(c.cacheResults)
	}
	return c.result
}

// NewTransformer creates a new cache that transforms the result of
// another cache. The transformFn will only be called if the source
// cache has updated the output, otherwise, the cached result will be
// returned.
//
// If the dependency returned an error before, or it returns an error
// this time, or if the transformerFn failed before, the function is
// reran.
func NewTransformer[T, V any](transformerFn func(Result[T]) Result[V], source Data[T]) Data[V] {
	return NewListMerger(func(caches []Result[T]) Result[V] {
		if len(caches) != 1 {
			panic(fmt.Errorf("invalid cache for transformer cache: %v", caches))
		}
		return transformerFn(caches[0])
	}, []Data[T]{source})
}

// NewSource creates a new cache that generates some data. This
// will always be called since we don't know the origin of the data and
// if it needs to be updated or not. sourceFn MUST be thread-safe.
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
	once   sync.Once
	fn     func() Result[T]
	result Result[T]
}

func (c *static[T]) Get() Result[T] {
	c.once.Do(func() {
		c.result = c.fn()
	})
	return c.result
}

// Replaceable is a cache that carries the result even when the cache is
// replaced. This is the type that should typically be stored in
// structs.
type Replaceable[T any] struct {
	cache  atomic.Pointer[Data[T]]
	result atomic.Pointer[Result[T]]
}

// Get retrieves the data from the underlying source. [Replaceable]
// implements the [Data] interface itself. This is a pass-through
// that calls the most recent underlying cache. If the cache fails but
// previously had returned a success, that success will be returned
// instead. If the cache fails but we never returned a success, that
// failure is returned.
func (c *Replaceable[T]) Get() Result[T] {
	result := (*c.cache.Load()).Get()

	for {
		cResult := c.result.Load()
		if result.Err != nil && cResult != nil && cResult.Err == nil {
			return *cResult
		}
		if c.result.CompareAndSwap(cResult, &result) {
			return result
		}
	}
}

// Replace changes the cache.
func (c *Replaceable[T]) Replace(cache Data[T]) {
	c.cache.Swap(&cache)
}
