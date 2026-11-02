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
	"errors"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/cached"
	"k8s.io/client-go/openapi/openapitest"
)

var cacheTestGVK = schema.GroupVersionKind{
	Group:   "example.com",
	Version: "v1",
	Kind:    "Widget",
}

type cacheTestGroupVersion struct {
	url string
}

func (g *cacheTestGroupVersion) Schema(string) ([]byte, error) {
	return nil, nil
}

func (g *cacheTestGroupVersion) ServerRelativeURL() string {
	return g.url
}

type cacheTestOpenAPIClient struct {
	hash                string
	pathsErr            error
	missingGroupVersion bool
}

func (c *cacheTestOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	if c.pathsErr != nil {
		return nil, c.pathsErr
	}
	if c.missingGroupVersion {
		return map[string]openapi.GroupVersion{}, nil
	}

	serverURL := "/openapi/v3/apis/example.com/v1"
	if c.hash != "" {
		serverURL += "?hash=" + c.hash
	}

	return map[string]openapi.GroupVersion{
		"apis/example.com/v1": &cacheTestGroupVersion{url: serverURL},
	}, nil
}

type cacheTestVerifier struct {
	mu      sync.Mutex
	calls   int
	results []error
}

func (v *cacheTestVerifier) HasSupport(schema.GroupVersionKind) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	v.calls++
	if len(v.results) == 0 {
		return nil
	}

	result := v.results[0]
	v.results = v.results[1:]
	return result
}

func (v *cacheTestVerifier) callCount() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.calls
}

func TestCachingQueryParamVerifierCachesDefiniteResults(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{name: "supported"},
		{
			name: "unsupported",
			err:  NewParamUnsupportedError(cacheTestGVK, QueryParamFieldValidation),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &cacheTestOpenAPIClient{hash: "schema-a"}
			next := &cacheTestVerifier{results: []error{tt.err}}
			cache := NewVerifierCache(10)

			verifier := NewCachingQueryParamVerifier(
				next, client, QueryParamFieldValidation, cache,
			)

			for range 2 {
				err := verifier.HasSupport(cacheTestGVK)
				if tt.err == nil && err != nil {
					t.Fatalf("expected supported result, got %v", err)
				}
				if tt.err != nil && !IsParamUnsupportedError(err) {
					t.Fatalf("expected ParamUnsupportedError, got %v", err)
				}
			}

			if got := next.callCount(); got != 1 {
				t.Fatalf("underlying verifier called %d times, want 1", got)
			}
		})
	}
}

func TestCachingQueryParamVerifierDoesNotCacheTransientErrors(t *testing.T) {
	client := &cacheTestOpenAPIClient{hash: "schema-a"}
	next := &cacheTestVerifier{
		results: []error{
			errors.New("temporary OpenAPI failure"),
			nil,
		},
	}

	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, NewVerifierCache(10),
	)

	if err := verifier.HasSupport(cacheTestGVK); err == nil {
		t.Fatal("expected the first call to return a transient error")
	}

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatalf("expected retry to succeed, got %v", err)
	}

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatalf("expected cached successful result, got %v", err)
	}

	if got := next.callCount(); got != 2 {
		t.Fatalf("underlying verifier called %d times, want 2", got)
	}
}

func TestCachingQueryParamVerifierUsesNewHashAfterSchemaChange(t *testing.T) {
	client := &cacheTestOpenAPIClient{hash: "schema-a"}
	next := &cacheTestVerifier{}
	cache := NewVerifierCache(10)

	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, cache,
	)

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatal(err)
	}

	client.hash = "schema-b"

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatal(err)
	}

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatal(err)
	}

	if got := next.callCount(); got != 2 {
		t.Fatalf("underlying verifier called %d times, want 2", got)
	}
}

func TestCachingQueryParamVerifierBypassesCacheWithoutHash(t *testing.T) {
	client := &cacheTestOpenAPIClient{}
	next := &cacheTestVerifier{}

	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, NewVerifierCache(10),
	)

	for range 3 {
		if err := verifier.HasSupport(cacheTestGVK); err != nil {
			t.Fatal(err)
		}
	}

	if got := next.callCount(); got != 3 {
		t.Fatalf("underlying verifier called %d times, want 3", got)
	}
}

func TestCachingQueryParamVerifierBypassesCacheOnPathsError(t *testing.T) {
	client := &cacheTestOpenAPIClient{
		hash:     "schema-a",
		pathsErr: errors.New("discovery unavailable"),
	}
	next := &cacheTestVerifier{}

	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, NewVerifierCache(10),
	)

	for range 2 {
		if err := verifier.HasSupport(cacheTestGVK); err != nil {
			t.Fatal(err)
		}
	}

	if got := next.callCount(); got != 2 {
		t.Fatalf("underlying verifier called %d times, want 2", got)
	}
}

func TestCachingQueryParamVerifierBypassesCacheForMissingGroupVersion(t *testing.T) {
	client := &cacheTestOpenAPIClient{
		hash:                "schema-a",
		missingGroupVersion: true,
	}
	next := &cacheTestVerifier{}

	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, NewVerifierCache(10),
	)

	for range 2 {
		if err := verifier.HasSupport(cacheTestGVK); err != nil {
			t.Fatal(err)
		}
	}

	if got := next.callCount(); got != 2 {
		t.Fatalf("underlying verifier called %d times, want 2", got)
	}
}

func TestCachingQueryParamVerifierSeparatesQueryParameters(t *testing.T) {
	client := &cacheTestOpenAPIClient{hash: "schema-a"}
	cache := NewVerifierCache(10)

	supported := &cacheTestVerifier{}
	unsupported := &cacheTestVerifier{
		results: []error{
			NewParamUnsupportedError(cacheTestGVK, "dryRun"),
		},
	}

	fieldValidationVerifier := NewCachingQueryParamVerifier(
		supported, client, QueryParamFieldValidation, cache,
	)
	dryRunVerifier := NewCachingQueryParamVerifier(
		unsupported, client, "dryRun", cache,
	)

	for range 2 {
		if err := fieldValidationVerifier.HasSupport(cacheTestGVK); err != nil {
			t.Fatalf("fieldValidation should be supported, got %v", err)
		}
		if err := dryRunVerifier.HasSupport(cacheTestGVK); !IsParamUnsupportedError(err) {
			t.Fatalf("dryRun should be unsupported, got %v", err)
		}
	}

	if got := supported.callCount(); got != 1 {
		t.Fatalf("fieldValidation verifier called %d times, want 1", got)
	}
	if got := unsupported.callCount(); got != 1 {
		t.Fatalf("dryRun verifier called %d times, want 1", got)
	}
}

func TestVerifierCacheEvictsLeastRecentlyUsedEntry(t *testing.T) {
	cache := NewVerifierCache(2)

	keyA := verifierCacheKey{serverRelativeURL: "schema-a"}
	keyB := verifierCacheKey{serverRelativeURL: "schema-b"}
	keyC := verifierCacheKey{serverRelativeURL: "schema-c"}

	cache.set(keyA, nil)
	cache.set(keyB, nil)

	if _, found := cache.get(keyA); !found {
		t.Fatal("expected key A to be present")
	}

	cache.set(keyC, nil)

	if _, found := cache.get(keyB); found {
		t.Fatal("expected least recently used key B to be evicted")
	}
	if _, found := cache.get(keyA); !found {
		t.Fatal("expected recently used key A to remain")
	}
	if _, found := cache.get(keyC); !found {
		t.Fatal("expected new key C to be present")
	}
}

func TestVerifierCacheIsSafeForConcurrentAccess(t *testing.T) {
	const goroutines = 100

	cache := NewVerifierCache(32)
	var wg sync.WaitGroup

	for i := range goroutines {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			key := verifierCacheKey{
				serverRelativeURL: strconv.Itoa(i),
				gvk:               cacheTestGVK,
				queryParam:        QueryParamFieldValidation,
			}

			cache.set(key, nil)
			cache.get(key)
		}(i)
	}

	wg.Wait()

	cache.mu.Lock()
	defer cache.mu.Unlock()

	if got := len(cache.entries); got > cache.maxEntries {
		t.Fatalf("cache contains %d entries, maximum is %d", got, cache.maxEntries)
	}
}

func TestCachingQueryParamVerifierSharesCacheAcrossInstances(t *testing.T) {
	client := &cacheTestOpenAPIClient{hash: "shared-schema"}
	cache := NewVerifierCache(10)

	firstNext := &cacheTestVerifier{}
	first := NewCachingQueryParamVerifier(
		firstNext, client, QueryParamFieldValidation, cache,
	)

	if err := first.HasSupport(cacheTestGVK); err != nil {
		t.Fatalf("first verifier returned unexpected error: %v", err)
	}

	secondNext := &cacheTestVerifier{}
	second := NewCachingQueryParamVerifier(
		secondNext, client, QueryParamFieldValidation, cache,
	)

	if err := second.HasSupport(cacheTestGVK); err != nil {
		t.Fatalf("second verifier returned unexpected error: %v", err)
	}

	if got := firstNext.callCount(); got != 1 {
		t.Fatalf("first underlying verifier called %d times, want 1", got)
	}
	if got := secondNext.callCount(); got != 0 {
		t.Fatalf("second underlying verifier called %d times, want 0 because the shared cache should be used", got)
	}
}

func TestVerifierCacheZeroValue(t *testing.T) {
	var cache VerifierCache
	key := verifierCacheKey{
		serverRelativeURL: "/openapi/v3/apis/example.com/v1?hash=zero-value",
		gvk:               cacheTestGVK,
		queryParam:        QueryParamFieldValidation,
	}

	cache.set(key, nil)

	if _, found := cache.get(key); !found {
		t.Fatal("expected the zero-value cache to retain the inserted entry")
	}
	if cache.maxEntries != defaultVerifierCacheSize {
		t.Fatalf("cache capacity = %d, want default %d", cache.maxEntries, defaultVerifierCacheSize)
	}
}

func TestCachingQueryParamVerifierConcurrentHits(t *testing.T) {
	const goroutines = 100

	client := &cacheTestOpenAPIClient{hash: "concurrent-schema"}
	next := &cacheTestVerifier{}
	verifier := NewCachingQueryParamVerifier(
		next, client, QueryParamFieldValidation, NewVerifierCache(10),
	)

	if err := verifier.HasSupport(cacheTestGVK); err != nil {
		t.Fatalf("failed to prime cache: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, goroutines)

	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := verifier.HasSupport(cacheTestGVK); err != nil {
				errs <- err
			}
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent cache hit returned an error: %v", err)
	}

	if got := next.callCount(); got != 1 {
		t.Fatalf("underlying verifier called %d times, want 1", got)
	}
}

// benchmarkHashedOpenAPIClient adds a stable document hash to test URLs.
type benchmarkHashedOpenAPIClient struct {
	openapi.Client
	hash  string
	once  sync.Once
	paths map[string]openapi.GroupVersion
	err   error
}

func (c *benchmarkHashedOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	c.once.Do(func() {
		paths, err := c.Client.Paths()
		if err != nil {
			c.err = err
			return
		}

		c.paths = make(map[string]openapi.GroupVersion, len(paths))
		for path, groupVersion := range paths {
			parsed, err := url.Parse(groupVersion.ServerRelativeURL())
			if err != nil {
				c.err = err
				return
			}

			query := parsed.Query()
			query.Set("hash", c.hash)
			parsed.RawQuery = query.Encode()

			c.paths[path] = &benchmarkHashedGroupVersion{
				GroupVersion:      groupVersion,
				serverRelativeURL: parsed.String(),
			}
		}
	})

	return c.paths, c.err
}

type benchmarkHashedGroupVersion struct {
	openapi.GroupVersion
	serverRelativeURL string
}

func (g *benchmarkHashedGroupVersion) ServerRelativeURL() string {
	return g.serverRelativeURL
}

func BenchmarkCachingQueryParamVerifier(b *testing.B) {
	gvk := schema.GroupVersionKind{
		Group:   "batch",
		Version: "v1",
		Kind:    "Job",
	}
	dynamicClient := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme())
	client := &benchmarkHashedOpenAPIClient{
		Client: cached.NewClient(openapitest.NewEmbeddedFileClient()),
		hash:   "benchmark-schema-hash",
	}

	b.Run("uncached", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			verifier := NewQueryParamVerifierV3(
				dynamicClient, client, QueryParamFieldValidation,
			)
			if err := verifier.HasSupport(gvk); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("shared-hash-keyed-cache", func(b *testing.B) {
		cache := NewVerifierCache(100)
		first := NewCachingQueryParamVerifier(
			NewQueryParamVerifierV3(dynamicClient, client, QueryParamFieldValidation),
			client,
			QueryParamFieldValidation,
			cache,
		)
		if err := first.HasSupport(gvk); err != nil {
			b.Fatal(err)
		}

		b.ReportAllocs()
		b.ResetTimer()

		for b.Loop() {
			verifier := NewCachingQueryParamVerifier(
				NewQueryParamVerifierV3(dynamicClient, client, QueryParamFieldValidation),
				client,
				QueryParamFieldValidation,
				cache,
			)
			if err := verifier.HasSupport(gvk); err != nil {
				b.Fatal(err)
			}
		}
	})
}
