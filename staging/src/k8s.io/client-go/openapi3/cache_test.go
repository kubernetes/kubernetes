/*
Copyright The Kubernetes Authors.

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

package openapi3

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

var (
	appsGV  = schema.GroupVersion{Group: "apps", Version: "v1"}
	batchGV = schema.GroupVersion{Group: "batch", Version: "v1"}
)

// instrumentedGroupVersion serves a fixed document and records how often the
// schema bytes were read, so a test can tell a cache hit from a refetch.
type instrumentedGroupVersion struct {
	mu          sync.Mutex
	locator     string
	body        []byte
	err         error
	schemaCalls int
}

func (g *instrumentedGroupVersion) Schema(contentType string) ([]byte, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.schemaCalls++
	if contentType != runtime.ContentTypeJSON {
		return nil, fmt.Errorf("unsupported content type %s", contentType)
	}
	if g.err != nil {
		return nil, g.err
	}
	return g.body, nil
}

func (g *instrumentedGroupVersion) ServerRelativeURL() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.locator
}

func (g *instrumentedGroupVersion) calls() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.schemaCalls
}

// cacheCarryingClient is an openapi.Client that hands out a shared parse
// cache, the way the kubectl factory client does.
type cacheCarryingClient struct {
	openapi.Client
	cache *Cache
}

func (c cacheCarryingClient) OpenAPIV3Cache() *Cache { return c.cache }

// syntheticGV builds a GroupVersion document holding one PATCH path and one
// component schema per kind, plus a status subresource path that carries the
// same Group/Version/Kind extension as its parent.
func syntheticGV(t testing.TB, gv schema.GroupVersion, kinds []string) []byte {
	t.Helper()
	paths := map[string]interface{}{}
	schemas := map[string]interface{}{}
	for _, kind := range kinds {
		gvk := map[string]string{"group": gv.Group, "version": gv.Version, "kind": kind}
		patch := map[string]interface{}{
			"x-kubernetes-group-version-kind": gvk,
			"requestBody": map[string]interface{}{
				"content": map[string]interface{}{
					"application/strategic-merge-patch+json": map[string]interface{}{},
				},
			},
			"parameters": []interface{}{
				map[string]interface{}{"name": "fieldValidation", "in": "query"},
			},
		}
		prefix := "/apis/" + gv.Group + "/" + gv.Version
		if gv.Group == "" {
			prefix = "/api/" + gv.Version
		}
		base := fmt.Sprintf("%s/namespaces/{namespace}/%ss/{name}", prefix, strings.ToLower(kind))
		paths[base] = map[string]interface{}{"patch": patch}
		paths[base+"/status"] = map[string]interface{}{"patch": patch}
		schemas[fmt.Sprintf("io.k8s.api.%s.%s.%s", gv.Group, gv.Version, kind)] = map[string]interface{}{
			"type":                            "object",
			"x-kubernetes-group-version-kind": []map[string]string{gvk},
			"properties": map[string]interface{}{
				"metadata": map[string]interface{}{"type": "object"},
				"spec":     map[string]interface{}{"type": "object"},
			},
		}
	}
	data, err := json.Marshal(map[string]interface{}{
		"openapi":    "3.0.0",
		"info":       map[string]interface{}{"title": gv.String(), "version": gv.Version},
		"paths":      paths,
		"components": map[string]interface{}{"schemas": schemas},
	})
	require.NoError(t, err)
	return data
}

func gvPath(gv schema.GroupVersion) string {
	if gv.Group == "" {
		return "api/" + gv.Version
	}
	return "apis/" + gv.Group + "/" + gv.Version
}

func TestCacheSharedAcrossRoots(t *testing.T) {
	doc := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/apps/v1?hash=abc",
		body:    syntheticGV(t, appsGV, []string{"Deployment"}),
	}
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): doc},
	}

	cache := NewCache()
	first := NewRootWithCache(client, cache)
	second := NewRootWithCache(client, cache)

	fromFirst, err := first.GVSpec(appsGV)
	require.NoError(t, err)
	fromSecond, err := second.GVSpec(appsGV)
	require.NoError(t, err)
	if fromFirst != fromSecond {
		t.Fatal("two Roots sharing a Cache each unmarshaled the document")
	}

	// A Root with its own cache must not be affected by the shared one.
	isolated := NewRootWithCache(client, nil)
	fromIsolated, err := isolated.GVSpec(appsGV)
	require.NoError(t, err)
	if fromIsolated == fromFirst {
		t.Fatal("a Root given a nil Cache shared the parse of an unrelated Root")
	}
}

func TestNewRootUsesCacheFromClient(t *testing.T) {
	doc := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/apps/v1?hash=abc",
		body:    syntheticGV(t, appsGV, []string{"Deployment"}),
	}
	client := cacheCarryingClient{
		Client: openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): doc},
		},
		cache: NewCache(),
	}

	// Separate callers each build their own Root from one client, exactly as
	// apply, diff and explain do.
	fromApply, err := NewRoot(client).GVSpec(appsGV)
	require.NoError(t, err)
	fromDiff, err := NewRoot(client).GVSpec(appsGV)
	require.NoError(t, err)
	if fromApply != fromDiff {
		t.Fatal("Roots built from one cache-carrying client did not share a parse")
	}

	// Without a cache-carrying client each Root parses for itself.
	plain := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): doc},
	}
	firstPlain, err := NewRoot(plain).GVSpec(appsGV)
	require.NoError(t, err)
	secondPlain, err := NewRoot(plain).GVSpec(appsGV)
	require.NoError(t, err)
	if firstPlain == secondPlain {
		t.Fatal("Roots built from a plain client unexpectedly shared a parse")
	}
}

func TestGVKIndex(t *testing.T) {
	doc := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/apps/v1?hash=abc",
		body:    syntheticGV(t, appsGV, []string{"Deployment", "StatefulSet"}),
	}
	root := NewRoot(openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): doc},
	})
	indexer, ok := root.(GVKIndexer)
	require.True(t, ok, "NewRoot must return a Root implementing GVKIndexer")

	deployment := appsGV.WithKind("Deployment")

	t.Run("a kind on several paths yields every path in path order", func(t *testing.T) {
		paths, err := indexer.GVKPatchPaths(deployment)
		require.NoError(t, err)
		// The resource path and its status subresource share the extension.
		require.Len(t, paths, 2)
		if paths[0].Patch.RequestBody == nil {
			t.Fatal("the first indexed path is missing its PATCH request body")
		}
		if _, found := paths[0].Patch.RequestBody.Content["application/strategic-merge-patch+json"]; !found {
			t.Fatal("the first indexed path does not accept a strategic merge patch")
		}
	})

	t.Run("component schemas are indexed by kind", func(t *testing.T) {
		schemas, err := indexer.GVKSchemas(deployment)
		require.NoError(t, err)
		require.Len(t, schemas, 1)
		if _, found := schemas[0].Properties["spec"]; !found {
			t.Fatal("the indexed component schema lost its properties")
		}
	})

	t.Run("an unknown kind is empty rather than an error", func(t *testing.T) {
		paths, err := indexer.GVKPatchPaths(appsGV.WithKind("DoesNotExist"))
		require.NoError(t, err)
		require.Empty(t, paths)
		schemas, err := indexer.GVKSchemas(appsGV.WithKind("DoesNotExist"))
		require.NoError(t, err)
		require.Empty(t, schemas)
	})

	t.Run("the index is reused and shares the parsed document", func(t *testing.T) {
		spec, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		paths, err := indexer.GVKPatchPaths(deployment)
		require.NoError(t, err)
		// The index points into the one cached parse rather than a second one.
		if paths[0] != spec.Paths.Paths["/apis/apps/v1/namespaces/{namespace}/deployments/{name}"] {
			t.Fatal("the index does not point into the cached document")
		}
	})

	t.Run("a changed locator rebuilds the index", func(t *testing.T) {
		doc.mu.Lock()
		doc.locator = "/openapi/v3/apis/apps/v1?hash=def"
		doc.body = syntheticGV(t, appsGV, []string{"Deployment"})
		doc.mu.Unlock()

		schemas, err := indexer.GVKSchemas(appsGV.WithKind("StatefulSet"))
		require.NoError(t, err)
		require.Empty(t, schemas, "the index still answers from the superseded document")
	})

	t.Run("an unparsable document reports the parse error", func(t *testing.T) {
		broken := &instrumentedGroupVersion{locator: "?hash=bad", body: []byte(`{`)}
		brokenRoot := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): broken},
		}).(GVKIndexer)
		_, err := brokenRoot.GVKPatchPaths(deployment)
		require.Error(t, err)
	})

	t.Run("a missing GroupVersion is reported, not indexed", func(t *testing.T) {
		emptyRoot := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{},
		}).(GVKIndexer)
		_, err := emptyRoot.GVKPatchPaths(deployment)
		require.ErrorAs(t, err, new(*GroupVersionNotFoundError))
	})
}

// TestApplyShapedWorkloadParsesDocumentOnce reproduces the access pattern that
// issue #143862 measured: kubectl apply of 20 objects in one GroupVersion,
// where each object asks whether its kind supports strategic merge patch and
// then asks for its component schema, with a few patch retries on top. Before
// the parse cache this unmarshaled the GroupVersion document on every one of
// those lookups.
func TestApplyShapedWorkloadParsesDocumentOnce(t *testing.T) {
	const objects = 20

	kinds := make([]string, 0, objects)
	for i := 0; i < objects; i++ {
		kinds = append(kinds, fmt.Sprintf("Kind%02d", i))
	}
	doc := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/apps/v1?hash=abc",
		body:    syntheticGV(t, appsGV, kinds),
	}

	// One Root for the whole apply, as ApplyOptions builds and every Patcher
	// then shares.
	root := NewRoot(openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{gvPath(appsGV): doc},
	})
	indexer, ok := root.(GVKIndexer)
	require.True(t, ok)

	var parsed interface{}
	lookups := 0
	for i, kind := range kinds {
		gvk := appsGV.WithKind(kind)

		// Every object performs the two lookups the Patcher performs.
		spec, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		lookups++
		if parsed == nil {
			parsed = spec
		} else if parsed != spec {
			t.Fatalf("object %d received a freshly unmarshaled document", i)
		}

		paths, err := indexer.GVKPatchPaths(gvk)
		require.NoError(t, err)
		require.Len(t, paths, 2, "kind %s", kind)

		schemas, err := indexer.GVKSchemas(gvk)
		require.NoError(t, err)
		require.Len(t, schemas, 1, "kind %s", kind)

		// A couple of objects hit a conflict and retry the patch.
		if i%10 == 0 {
			retried, err := root.GVSpec(appsGV)
			require.NoError(t, err)
			lookups++
			if retried != spec {
				t.Fatalf("the retry for object %d unmarshaled the document again", i)
			}
		}
	}

	require.GreaterOrEqual(t, lookups, 22, "the workload should exercise repeated lookups")
	// Identical pointers across every lookup prove the document was
	// unmarshaled exactly once for the whole apply.
	require.NotNil(t, parsed)
	// The bytes are still fetched per lookup; it is the parse that is shared.
	require.Greater(t, doc.calls(), lookups)
}

func TestCacheConcurrentLookups(t *testing.T) {
	doc := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/apps/v1?hash=abc",
		body:    syntheticGV(t, appsGV, []string{"Deployment", "StatefulSet"}),
	}
	batch := &instrumentedGroupVersion{
		locator: "/openapi/v3/apis/batch/v1?hash=abc",
		body:    syntheticGV(t, batchGV, []string{"Job"}),
	}
	cache := NewCache()
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			gvPath(appsGV):  doc,
			gvPath(batchGV): batch,
		},
	}

	const goroutines = 16
	var wg sync.WaitGroup
	errs := make(chan error, goroutines*4)
	// Every goroutine builds its own Root over the shared cache, so the
	// cache sees concurrent readers, writers and index builds.
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			root := NewRootWithCache(client, cache)
			indexer := root.(GVKIndexer)
			gv, kind := appsGV, "Deployment"
			if i%2 == 0 {
				gv, kind = batchGV, "Job"
			}
			if _, err := root.GVSpec(gv); err != nil {
				errs <- err
			}
			if _, err := root.GVSpecAsMap(gv); err != nil {
				errs <- err
			}
			if _, err := indexer.GVKPatchPaths(gv.WithKind(kind)); err != nil {
				errs <- err
			}
			if _, err := indexer.GVKSchemas(gv.WithKind(kind)); err != nil {
				errs <- err
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	// All goroutines must have converged on one parse per GroupVersion.
	appsSpec, err := NewRootWithCache(client, cache).GVSpec(appsGV)
	require.NoError(t, err)
	batchSpec, err := NewRootWithCache(client, cache).GVSpec(batchGV)
	require.NoError(t, err)
	if appsSpec == batchSpec {
		t.Fatal("distinct GroupVersions converged on one entry")
	}
}

// benchDocument is a document of roughly the shape kubectl works with: a few
// hundred kinds, each contributing two paths and one component schema.
func benchDocument(b *testing.B) ([]byte, []string) {
	b.Helper()
	kinds := make([]string, 0, 200)
	for i := 0; i < 200; i++ {
		kinds = append(kinds, fmt.Sprintf("Kind%03d", i))
	}
	return syntheticGV(b, appsGV, kinds), kinds
}

// BenchmarkGVSpecReusedRoot measures the lookup cost once a Root has parsed
// the document, which is what a caller reusing one Root pays per object.
func BenchmarkGVSpecReusedRoot(b *testing.B) {
	body, _ := benchDocument(b)
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			gvPath(appsGV): &instrumentedGroupVersion{locator: "?hash=abc", body: body},
		},
	}
	root := NewRoot(client)
	if _, err := root.GVSpec(appsGV); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := root.GVSpec(appsGV); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGVSpecFreshRoot measures the same lookup when every caller builds
// its own Root, which is the cost the cache removes.
func BenchmarkGVSpecFreshRoot(b *testing.B) {
	body, _ := benchDocument(b)
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			gvPath(appsGV): &instrumentedGroupVersion{locator: "?hash=abc", body: body},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := NewRoot(client).GVSpec(appsGV); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGVKPatchPathsIndexed measures an indexed kind lookup.
func BenchmarkGVKPatchPathsIndexed(b *testing.B) {
	body, kinds := benchDocument(b)
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			gvPath(appsGV): &instrumentedGroupVersion{locator: "?hash=abc", body: body},
		},
	}
	indexer := NewRoot(client).(GVKIndexer)
	if _, err := indexer.GVKPatchPaths(appsGV.WithKind(kinds[0])); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gvk := appsGV.WithKind(kinds[i%len(kinds)])
		if _, err := indexer.GVKPatchPaths(gvk); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGVKPatchPathsScan measures the same lookup done by scanning every
// path in the document, which is what callers did before the index.
func BenchmarkGVKPatchPathsScan(b *testing.B) {
	body, kinds := benchDocument(b)
	client := openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			gvPath(appsGV): &instrumentedGroupVersion{locator: "?hash=abc", body: body},
		},
	}
	root := NewRoot(client)
	doc, err := root.GVSpec(appsGV)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gvk := appsGV.WithKind(kinds[i%len(kinds)])
		found := 0
		for _, path := range doc.Paths.Paths {
			if path == nil || path.Patch == nil {
				continue
			}
			if indexed, ok := singleGVKExtension(path.Patch.Extensions); ok && indexed == gvk {
				found++
			}
		}
		if found == 0 {
			b.Fatalf("scan did not find %s", gvk)
		}
	}
}
