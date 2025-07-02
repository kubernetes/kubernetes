// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/http"
	"net/url"
)

// CacheInvalidator describes the interface implemented by types that can
// invalidate cache entries for a target URI when an unsafe request receives a
// non-error response, as required by RFC 9111 §4.4. It may also invalidate
// entries for URIs in Location or Content-Location headers, but only if they
// share the same origin as the target URI.
type CacheInvalidator interface {
	InvalidateCache(reqURL *url.URL, respHeader http.Header, refs ResponseRefs, key string)
}

type cacheInvalidator struct {
	cache ResponseCache
	cke   URLKeyer
}

func NewCacheInvalidator(cache ResponseCache, cke URLKeyer) *cacheInvalidator {
	return &cacheInvalidator{cache, cke}
}

func (r *cacheInvalidator) InvalidateCache(
	reqURL *url.URL,
	respHeader http.Header,
	refs ResponseRefs,
	key string,
) {
	deleted := map[string]struct{}{}
	del := func(k string) {
		if _, ok := deleted[k]; !ok {
			_ = r.cache.Delete(k)
			deleted[k] = struct{}{}
		}
	}
	for h := range refs.ResponseIDs() {
		del(h)
	}
	r.invalidateLocationHeaders(reqURL, respHeader, del)
	del(key)
}

var locationHeaders = [...]string{"Location", "Content-Location"}

func (r *cacheInvalidator) invalidateLocationHeaders(
	reqURL *url.URL,
	respHeader http.Header,
	deleteFn func(string),
) {
	for _, hdr := range locationHeaders {
		loc := respHeader.Get(hdr)
		if loc == "" {
			continue
		}
		locURL, err := url.Parse(loc)
		if err != nil {
			continue
		}
		locURL = reqURL.ResolveReference(locURL)
		if sameOrigin(reqURL, locURL) {
			urlKey := r.cke.URLKey(locURL)
			refs, _ := r.cache.GetRefs(urlKey)
			for h := range refs.ResponseIDs() {
				deleteFn(h)
			}
			deleteFn(urlKey)
		}
	}
}
