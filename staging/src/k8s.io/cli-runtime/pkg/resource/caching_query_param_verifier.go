/*
Copyright 2026 The Kubernetes Authors.

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

package resource

import (
	"container/list"
	"net/url"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
)

const defaultVerifierCacheSize = 10000

// VerifierCache is a bounded, concurrency-safe cache for query parameter
// verifier results. It can be shared by multiple verifier instances.
// The zero value is ready to use. A VerifierCache must not be copied after first use.
type VerifierCache struct {
	mu         sync.Mutex
	maxEntries int
	entries    map[verifierCacheKey]*list.Element
	lru        list.List
}

type verifierCacheKey struct {
	serverRelativeURL string
	gvk               schema.GroupVersionKind
	queryParam        VerifiableQueryParam
}

type verifierCacheEntry struct {
	key verifierCacheKey
	err error
}

// NewVerifierCache creates a bounded verifier cache.
// A non-positive maxEntries value uses the default capacity.
func NewVerifierCache(maxEntries int) *VerifierCache {
	if maxEntries <= 0 {
		maxEntries = defaultVerifierCacheSize
	}

	return &VerifierCache{
		maxEntries: maxEntries,
		entries:    make(map[verifierCacheKey]*list.Element),
	}
}

func (c *VerifierCache) get(key verifierCacheKey) (error, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	element, found := c.entries[key]
	if !found {
		return nil, false
	}

	c.lru.MoveToFront(element)
	return element.Value.(*verifierCacheEntry).err, true
}

func (c *VerifierCache) set(key verifierCacheKey, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.maxEntries <= 0 {
		c.maxEntries = defaultVerifierCacheSize
	}
	if c.entries == nil {
		c.entries = make(map[verifierCacheKey]*list.Element)
	}

	if element, found := c.entries[key]; found {
		element.Value.(*verifierCacheEntry).err = err
		c.lru.MoveToFront(element)
		return
	}

	element := c.lru.PushFront(&verifierCacheEntry{
		key: key,
		err: err,
	})
	c.entries[key] = element

	if len(c.entries) > c.maxEntries {
		oldest := c.lru.Back()
		entry := oldest.Value.(*verifierCacheEntry)
		delete(c.entries, entry.key)
		c.lru.Remove(oldest)
	}
}

var _ Verifier = &cachingQueryParamVerifier{}

type cachingQueryParamVerifier struct {
	next       Verifier
	client     openapi.Client
	queryParam VerifiableQueryParam
	cache      *VerifierCache
}

// NewCachingQueryParamVerifier wraps next with a hash-keyed cache.
//
// The client must be the same stable OpenAPI client used to construct next.
// When next is created with NewQueryParamVerifierV3, pass the same client
// (typically a cached.NewClient) to both constructors. The queryParam must
// match the parameter checked by next. The cache can be shared by verifier
// instances checking the same query parameter.
//
// Results are cached only when next returns nil or a ParamUnsupportedError.
// Transient errors and OpenAPI URLs without a non-empty hash are not cached.
func NewCachingQueryParamVerifier(
	next Verifier,
	client openapi.Client,
	queryParam VerifiableQueryParam,
	cache *VerifierCache,
) Verifier {
	if cache == nil {
		cache = NewVerifierCache(defaultVerifierCacheSize)
	}

	return &cachingQueryParamVerifier{
		next:       next,
		client:     client,
		queryParam: queryParam,
		cache:      cache,
	}
}

func (v *cachingQueryParamVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	key, cacheable := v.cacheKey(gvk)
	if !cacheable {
		return v.next.HasSupport(gvk)
	}

	if err, found := v.cache.get(key); found {
		return err
	}

	err := v.next.HasSupport(gvk)
	if err == nil || IsParamUnsupportedError(err) {
		v.cache.set(key, err)
	}

	return err
}

func (v *cachingQueryParamVerifier) cacheKey(
	gvk schema.GroupVersionKind,
) (verifierCacheKey, bool) {
	if v.client == nil {
		return verifierCacheKey{}, false
	}

	paths, err := v.client.Paths()
	if err != nil {
		return verifierCacheKey{}, false
	}

	gv := gvk.GroupVersion()
	apiPath := "api/" + gv.Version
	if gv.Group != "" {
		apiPath = "apis/" + gv.Group + "/" + gv.Version
	}

	groupVersion, found := paths[apiPath]
	if !found || groupVersion == nil {
		return verifierCacheKey{}, false
	}

	serverRelativeURL := groupVersion.ServerRelativeURL()
	parsedURL, err := url.Parse(serverRelativeURL)
	if err != nil || parsedURL.Query().Get("hash") == "" {
		return verifierCacheKey{}, false
	}

	return verifierCacheKey{
		serverRelativeURL: serverRelativeURL,
		gvk:               gvk,
		queryParam:        v.queryParam,
	}, true
}
