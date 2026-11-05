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
	"fmt"
	"maps"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

type fakeGroupVersion struct {
	spec        []byte
	err         error
	relativeURL string
}

func (f fakeGroupVersion) Schema(contentType string) ([]byte, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.spec, nil
}

func (f fakeGroupVersion) ServerRelativeURL() string {
	return f.relativeURL
}

type countingGroupVersion struct {
	delegate    openapi.GroupVersion
	relativeURL string
	schemaCalls *atomic.Int32
}

func (c *countingGroupVersion) Schema(contentType string) ([]byte, error) {
	c.schemaCalls.Add(1)
	return c.delegate.Schema(contentType)
}

func (c *countingGroupVersion) ServerRelativeURL() string {
	if c.relativeURL != "" {
		return c.relativeURL
	}
	return c.delegate.ServerRelativeURL()
}

type dynamicFakeClient struct {
	mu       sync.Mutex
	pathsMap map[string]openapi.GroupVersion
	err      error
}

func (d *dynamicFakeClient) Paths() (map[string]openapi.GroupVersion, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.err != nil {
		return nil, d.err
	}
	out := make(map[string]openapi.GroupVersion, len(d.pathsMap))
	maps.Copy(out, d.pathsMap)
	return out, nil
}

func TestResources_SchemaForGVK(t *testing.T) {
	res := NewResources(openapitest.NewEmbeddedFileClient())

	t.Run("existing built-in GVKs return schema and component schemas map", func(t *testing.T) {
		gvks := []schema.GroupVersionKind{
			{Group: "apps", Version: "v1", Kind: "Deployment"},
			{Group: "apps", Version: "v1", Kind: "StatefulSet"},
			{Group: "batch", Version: "v1", Kind: "Job"},
			{Group: "", Version: "v1", Kind: "Pod"},
			{Group: "", Version: "v1", Kind: "ConfigMap"},
			{Group: "", Version: "v1", Kind: "Service"},
		}
		for _, gvk := range gvks {
			s, schemas, err := res.SchemaForGVK(gvk)
			require.NoError(t, err, "unexpected error for %v", gvk)
			require.NotNil(t, s, "expected non-nil schema for %v", gvk)
			require.NotEmpty(t, schemas, "expected non-empty schemas map for %v", gvk)
		}
	})

	t.Run("unknown Kind in existing GroupVersion returns nil schema without error", func(t *testing.T) {
		s, schemas, err := res.SchemaForGVK(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "DoesNotExist"})
		require.NoError(t, err)
		assert.Nil(t, s)
		assert.NotEmpty(t, schemas)
	})

	t.Run("unknown GroupVersion returns GroupVersionNotFoundError", func(t *testing.T) {
		gv := schema.GroupVersion{Group: "not", Version: "found"}
		_, _, err := res.SchemaForGVK(gv.WithKind("Foo"))
		var notFoundErr *GroupVersionNotFoundError
		require.ErrorAs(t, err, &notFoundErr)
		assert.Equal(t, gv, notFoundErr.gv)
	})

	t.Run("document with nil Components returns error", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0","paths":{}}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=1",
		}
		r := NewResources(fakeClient)
		_, _, err := r.SchemaForGVK(schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"})
		require.ErrorContains(t, err, "OpenAPI V3 Components is nil")
	})
}

func TestResources_GetConsumes(t *testing.T) {
	res := NewResources(openapitest.NewEmbeddedFileClient())

	t.Run("built-in resources return sorted PATCH media types and defensive copy", func(t *testing.T) {
		for _, gvk := range []schema.GroupVersionKind{
			{Group: "apps", Version: "v1", Kind: "Deployment"},
			{Group: "", Version: "v1", Kind: "Pod"},
			{Group: "", Version: "v1", Kind: "ConfigMap"},
		} {
			consumes, err := res.GetConsumes(gvk, "PATCH")
			require.NoError(t, err)
			assert.True(t, slices.Contains(consumes, "application/strategic-merge-patch+json"), "expected %v to support strategic-merge-patch", gvk)
			assert.True(t, slices.Contains(consumes, "application/merge-patch+json"), "expected %v to support merge-patch", gvk)
			assert.True(t, slices.IsSorted(consumes), "expected sorted media types")

			// Lowercase method works identically.
			consumesLower, err := res.GetConsumes(gvk, "patch")
			require.NoError(t, err)
			assert.Equal(t, consumes, consumesLower)

			// Mutating returned slice does not mutate cached slice.
			consumes[0] = "mutated"
			consumesAgain, err := res.GetConsumes(gvk, "PATCH")
			require.NoError(t, err)
			assert.NotEqual(t, "mutated", consumesAgain[0])
		}
	})

	t.Run("CRD without strategic-merge-patch omits strategic-merge-patch", func(t *testing.T) {
		crdSpec := `{
			"openapi": "3.0.0",
			"paths": {
				"/apis/example.com/v1/namespaces/{namespace}/widgets/{name}": {
					"patch": {
						"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Widget"},
						"requestBody": {
							"content": {
								"application/json-patch+json": {},
								"application/merge-patch+json": {},
								"application/apply-patch+yaml": {}
							}
						}
					}
				}
			}
		}`
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(crdSpec),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=crd1",
		}
		r := NewResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

		consumes, err := r.GetConsumes(gvk, "PATCH")
		require.NoError(t, err)
		assert.Equal(t, []string{
			"application/apply-patch+yaml",
			"application/json-patch+json",
			"application/merge-patch+json",
		}, consumes)

		// Unknown HTTP method on existing GVK returns nil, nil.
		consumesPut, err := r.GetConsumes(gvk, "PUT")
		require.NoError(t, err)
		assert.Nil(t, consumesPut)
	})

	t.Run("non-patchable List kind returns nil without error", func(t *testing.T) {
		consumes, err := res.GetConsumes(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ReplicationControllerList"}, "PATCH")
		require.NoError(t, err)
		assert.Nil(t, consumes)
	})

	t.Run("missing Paths returns error", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0"}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=empty",
		}
		r := NewResources(fakeClient)
		_, err := r.GetConsumes(schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}, "PATCH")
		require.ErrorContains(t, err, "does not exist for OpenAPI V3")
	})
}

func TestResources_SupportsQueryParam(t *testing.T) {
	res := NewResources(openapitest.NewEmbeddedFileClient())

	t.Run("built-in kinds support fieldValidation on PATCH, POST, and PUT", func(t *testing.T) {
		for _, gvk := range []schema.GroupVersionKind{
			{Group: "batch", Version: "v1", Kind: "Job"},
			{Group: "apps", Version: "v1", Kind: "Deployment"},
			{Group: "", Version: "v1", Kind: "Namespace"},
			{Group: "", Version: "v1", Kind: "Pod"},
		} {
			for _, op := range []string{"PATCH", "POST", "PUT"} {
				ok, err := res.SupportsQueryParam(gvk, op, "fieldValidation")
				require.NoError(t, err)
				assert.True(t, ok, "expected %v %s to support fieldValidation", gvk, op)
			}

			ok, err := res.SupportsQueryParam(gvk, "PATCH", "UnknownQueryParam")
			require.NoError(t, err)
			assert.False(t, ok)

			ok, err = res.SupportsQueryParam(gvk, "HEAD", "fieldValidation")
			require.NoError(t, err)
			assert.False(t, ok)
		}
	})

	t.Run("path-level and global $ref parameters in Components.Parameters are resolved", func(t *testing.T) {
		docWithGlobalAndPathParams := `{
			"openapi": "3.0.0",
			"paths": {
				"/apis/example.com/v1/widgets/{name}": {
					"parameters": [
						{"name": "pathLevelQuery", "in": "query"}
					],
					"patch": {
						"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Widget"},
						"parameters": [
							{"$ref": "#/parameters/fieldValidationParam"}
						]
					}
				}
			},
			"components": {
				"parameters": {
					"fieldValidationParam": {
						"name": "fieldValidation",
						"in": "query"
					}
				}
			}
		}`
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(docWithGlobalAndPathParams),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=refparam",
		}
		r := NewResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

		ok, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.NoError(t, err)
		assert.True(t, ok)

		ok, err = r.SupportsQueryParam(gvk, "PATCH", "pathLevelQuery")
		require.NoError(t, err)
		assert.True(t, ok)
	})

	t.Run("nil Paths returns invalid OpenAPI v3 document error, empty Paths map returns false", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0"}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=nilpaths",
		}
		r := NewResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
		_, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.ErrorContains(t, err, "invalid OpenAPI v3 document")

		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0","paths":{}}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=emptypaths",
		}
		ok, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.NoError(t, err)
		assert.False(t, ok)
	})
}

func TestResources_CachingAndHashInvalidation(t *testing.T) {
	embedded := openapitest.NewEmbeddedFileClient()
	embeddedPaths, err := embedded.Paths()
	require.NoError(t, err)

	var coreV1Calls atomic.Int32
	var crdCalls atomic.Int32

	crdSpecV1 := `{
		"openapi": "3.0.0",
		"paths": {
			"/apis/example.com/v1/widgets/{name}": {
				"patch": {
					"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Widget"},
					"parameters": [{"name": "fieldValidation", "in": "query"}],
					"requestBody": {"content": {"application/merge-patch+json": {}}}
				}
			}
		},
		"components": {
			"schemas": {
				"com.example.v1.Widget": {
					"type": "object",
					"x-kubernetes-group-version-kind": [{"group": "example.com", "version": "v1", "kind": "Widget"}]
				}
			}
		}
	}`
	crdSpecV2 := `{
		"openapi": "3.0.0",
		"paths": {
			"/apis/example.com/v1/widgets/{name}": {
				"patch": {
					"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Widget"},
					"parameters": [{"name": "fieldValidation", "in": "query"}],
					"requestBody": {"content": {"application/merge-patch+json": {}}}
				}
			},
			"/apis/example.com/v1/gadgets/{name}": {
				"patch": {
					"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Gadget"},
					"parameters": [{"name": "fieldValidation", "in": "query"}],
					"requestBody": {"content": {"application/merge-patch+json": {}}}
				}
			}
		},
		"components": {
			"schemas": {
				"com.example.v1.Widget": {
					"type": "object",
					"x-kubernetes-group-version-kind": [{"group": "example.com", "version": "v1", "kind": "Widget"}]
				},
				"com.example.v1.Gadget": {
					"type": "object",
					"x-kubernetes-group-version-kind": [{"group": "example.com", "version": "v1", "kind": "Gadget"}]
				}
			}
		}
	}`

	client := &dynamicFakeClient{
		pathsMap: map[string]openapi.GroupVersion{
			"api/v1": &countingGroupVersion{
				delegate:    embeddedPaths["api/v1"],
				relativeURL: "/openapi/v3/api/v1?hash=corev1hash",
				schemaCalls: &coreV1Calls,
			},
			"apis/example.com/v1": &countingGroupVersion{
				delegate:    fakeGroupVersion{spec: []byte(crdSpecV1)},
				relativeURL: "/openapi/v3/apis/example.com/v1?hash=crdv1",
				schemaCalls: &crdCalls,
			},
		},
	}

	res := NewResources(client)

	// 1. Perform 60 lookups across 7 distinct core/v1 Kinds (simulating a 20-resource apply):
	// Schema("application/json") must be called only ONCE for core/v1.
	coreKinds := []string{"ServiceAccount", "Secret", "ConfigMap", "Service", "PersistentVolumeClaim", "Pod", "ReplicationController"}
	for i := range 20 {
		kind := coreKinds[i%len(coreKinds)]
		gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: kind}

		ok, err := res.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.NoError(t, err)
		assert.True(t, ok)

		consumes, err := res.GetConsumes(gvk, "PATCH")
		require.NoError(t, err)
		assert.True(t, slices.Contains(consumes, "application/strategic-merge-patch+json"))

		s, _, err := res.SchemaForGVK(gvk)
		require.NoError(t, err)
		require.NotNil(t, s)
	}
	assert.Equal(t, int32(1), coreV1Calls.Load(), "core/v1 should only be fetched and unmarshaled once across all lookups")
	assert.Equal(t, int32(0), crdCalls.Load(), "unused CRD GroupVersion should never be fetched")

	// 2. Query CRD v1 (Widget exists, Gadget does not yet exist).
	widgetGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
	gadgetGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Gadget"}

	s, _, err := res.SchemaForGVK(widgetGVK)
	require.NoError(t, err)
	require.NotNil(t, s)

	s, _, err = res.SchemaForGVK(gadgetGVK)
	require.NoError(t, err)
	assert.Nil(t, s)
	assert.Equal(t, int32(1), crdCalls.Load())

	// 3. Simulate CRD schema update changing ServerRelativeURL ?hash=crdv1 -> ?hash=crdv2.
	client.mu.Lock()
	client.pathsMap["apis/example.com/v1"] = &countingGroupVersion{
		delegate:    fakeGroupVersion{spec: []byte(crdSpecV2)},
		relativeURL: "/openapi/v3/apis/example.com/v1?hash=crdv2",
		schemaCalls: &crdCalls,
	}
	client.mu.Unlock()

	s, _, err = res.SchemaForGVK(gadgetGVK)
	require.NoError(t, err)
	require.NotNil(t, s, "Gadget should be found after ServerRelativeURL hash change")
	assert.Equal(t, int32(2), crdCalls.Load(), "CRD GroupVersion should be re-fetched once after hash change")
	assert.Equal(t, int32(1), coreV1Calls.Load(), "core/v1 cache must remain warm when CRD hash changes")

	// 4. Simulate CRD deletion removing apis/example.com/v1 from Paths().
	client.mu.Lock()
	delete(client.pathsMap, "apis/example.com/v1")
	client.mu.Unlock()

	_, _, err = res.SchemaForGVK(widgetGVK)
	var notFoundErr *GroupVersionNotFoundError
	require.ErrorAs(t, err, &notFoundErr)
}

func TestResources_ConcurrentAccess(t *testing.T) {
	embedded := openapitest.NewEmbeddedFileClient()
	embeddedPaths, err := embedded.Paths()
	require.NoError(t, err)

	var coreV1Calls atomic.Int32
	var appsV1Calls atomic.Int32

	client := &dynamicFakeClient{
		pathsMap: map[string]openapi.GroupVersion{
			"api/v1": &countingGroupVersion{
				delegate:    embeddedPaths["api/v1"],
				relativeURL: "/openapi/v3/api/v1?hash=core",
				schemaCalls: &coreV1Calls,
			},
			"apis/apps/v1": &countingGroupVersion{
				delegate:    embeddedPaths["apis/apps/v1"],
				relativeURL: "/openapi/v3/apis/apps/v1?hash=apps",
				schemaCalls: &appsV1Calls,
			},
		},
	}

	res := NewResources(client)

	const goroutines = 40
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make(chan error, goroutines)

	gvks := []schema.GroupVersionKind{
		{Group: "", Version: "v1", Kind: "Pod"},
		{Group: "", Version: "v1", Kind: "Service"},
		{Group: "", Version: "v1", Kind: "ConfigMap"},
		{Group: "apps", Version: "v1", Kind: "Deployment"},
		{Group: "apps", Version: "v1", Kind: "StatefulSet"},
	}

	for i := range goroutines {
		go func(idx int) {
			defer wg.Done()
			gvk := gvks[idx%len(gvks)]
			if ok, err := res.SupportsQueryParam(gvk, "PATCH", "fieldValidation"); err != nil || !ok {
				errs <- fmt.Errorf("SupportsQueryParam(%v): ok=%v, err=%w", gvk, ok, err)
				return
			}
			consumes, err := res.GetConsumes(gvk, "PATCH")
			if err != nil || !slices.Contains(consumes, "application/strategic-merge-patch+json") {
				errs <- fmt.Errorf("GetConsumes(%v): consumes=%v, err=%w", gvk, consumes, err)
				return
			}
			if s, _, err := res.SchemaForGVK(gvk); err != nil || s == nil {
				errs <- fmt.Errorf("SchemaForGVK(%v): s=%v, err=%w", gvk, s, err)
				return
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	assert.Equal(t, int32(1), coreV1Calls.Load(), "core/v1 must only be fetched once under concurrent access")
	assert.Equal(t, int32(1), appsV1Calls.Load(), "apps/v1 must only be fetched once under concurrent access")
}

func TestResources_ErrorAndEdgeCases(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

	t.Run("client Paths() error propagates to all methods", func(t *testing.T) {
		fakeClient := openapitest.FakeClient{ForcedErr: fmt.Errorf("paths failure")}
		r := NewResources(fakeClient)

		_, _, err := r.SchemaForGVK(gvk)
		require.ErrorContains(t, err, "paths failure")

		_, err = r.GetConsumes(gvk, "PATCH")
		require.ErrorContains(t, err, "paths failure")

		_, err = r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.ErrorContains(t, err, "paths failure")
	})

	t.Run("GroupVersion Schema() error and invalid JSON error propagate", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			err:         fmt.Errorf("schema download error"),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=err",
		}
		r := NewResources(fakeClient)
		_, _, err := r.SchemaForGVK(gvk)
		require.ErrorContains(t, err, "schema download error")

		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{invalid json`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=badjson",
		}
		_, _, err = r.SchemaForGVK(gvk)
		require.Error(t, err)
	})

	t.Run("NewRoot and NewResources share the underlying root struct implementing both interfaces", func(t *testing.T) {
		client := openapitest.NewEmbeddedFileClient()

		rt := NewRoot(client)
		resFromRoot, ok := rt.(Resources)
		require.True(t, ok, "expected NewRoot to implement Resources")
		s, _, err := resFromRoot.SchemaForGVK(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"})
		require.NoError(t, err)
		require.NotNil(t, s)

		res := NewResources(client)
		rootFromRes, ok := res.(Root)
		require.True(t, ok, "expected NewResources to implement Root")
		gvs, err := rootFromRes.GroupVersions()
		require.NoError(t, err)
		require.NotEmpty(t, gvs)
	})

	t.Run("GroupVersionNotFoundError formats message", func(t *testing.T) {
		err := &GroupVersionNotFoundError{gv: schema.GroupVersion{Group: "example.com", Version: "v1"}}
		assert.Contains(t, err.Error(), "example.com/v1")
	})

	t.Run("extension helpers handle nil, malformed, and typed extensions", func(t *testing.T) {
		_, ok := extractSingleGVKFromExtension(nil)
		assert.False(t, ok)

		_, ok = extractSingleGVKFromExtension(map[string]interface{}{"other": "val"})
		assert.False(t, ok)

		parsedGVK, ok := extractSingleGVKFromExtension(map[string]interface{}{
			groupVersionKindExtensionKey: map[string]string{"group": "apps", "version": "v1", "kind": "Deployment"},
		})
		assert.True(t, ok)
		assert.Equal(t, schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, parsedGVK)

		_, ok = extractSingleGVKFromExtension(map[string]interface{}{
			groupVersionKindExtensionKey: map[string]string{"group": "apps", "version": "v1"},
		})
		assert.False(t, ok)

		_, ok = extractSingleGVKFromExtension(map[string]interface{}{
			groupVersionKindExtensionKey: "not-a-map",
		})
		assert.False(t, ok)

		assert.Nil(t, extractGVKListFromExtension(nil))
		assert.Nil(t, extractGVKListFromExtension(map[string]interface{}{"other": "val"}))
		assert.Nil(t, extractGVKListFromExtension(map[string]interface{}{groupVersionKindExtensionKey: "not-a-list"}))

		gvkList := extractGVKListFromExtension(map[string]interface{}{
			groupVersionKindExtensionKey: []map[string]string{
				{"group": "apps", "version": "v1", "kind": "Deployment"},
				{"group": "apps", "version": "v1"},
			},
		})
		require.Len(t, gvkList, 1)
		assert.Equal(t, schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, gvkList[0])

		entry := buildGVCacheEntry("url", nil)
		require.NotNil(t, entry)
	})
}
