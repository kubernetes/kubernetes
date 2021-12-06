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

package validate

import (
	re "regexp"
	"sync"
	"sync/atomic"
)

// Cache for compiled regular expressions
var (
	cacheMutex = &sync.Mutex{}
	reDict     = atomic.Value{} // map[string]*re.Regexp
)

func compileRegexp(pattern string) (*re.Regexp, error) {
	if cache, ok := reDict.Load().(map[string]*re.Regexp); ok {
		if r := cache[pattern]; r != nil {
			return r, nil
		}
	}

	r, err := re.Compile(pattern)
	if err != nil {
		return nil, err
	}
	cacheRegexp(r)
	return r, nil
}

func mustCompileRegexp(pattern string) *re.Regexp {
	if cache, ok := reDict.Load().(map[string]*re.Regexp); ok {
		if r := cache[pattern]; r != nil {
			return r
		}
	}

	r := re.MustCompile(pattern)
	cacheRegexp(r)
	return r
}

func cacheRegexp(r *re.Regexp) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	if cache, ok := reDict.Load().(map[string]*re.Regexp); !ok || cache[r.String()] == nil {
		newCache := map[string]*re.Regexp{
			r.String(): r,
		}

		for k, v := range cache {
			newCache[k] = v
		}

		reDict.Store(newCache)
	}
}
