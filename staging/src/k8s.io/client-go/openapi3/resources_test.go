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
	"k8s.io/kube-openapi/pkg/spec3"
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

func TestResources_SchemaForGVKAndSchemasForGV(t *testing.T) {
	res := NewRootAndResources(openapitest.NewEmbeddedFileClient())

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
			s, err := res.SchemaForGVK(gvk)
			require.NoError(t, err, "unexpected error for %v", gvk)
			require.NotNil(t, s, "expected non-nil schema for %v", gvk)

			schemas, err := res.SchemasForGV(gvk.GroupVersion())
			require.NoError(t, err, "unexpected SchemasForGV error for %v", gvk.GroupVersion())
			require.NotEmpty(t, schemas, "expected non-empty schemas map for %v", gvk.GroupVersion())
		}

		appsSchemas, err := res.SchemasForGV(schema.GroupVersion{Group: "apps", Version: "v1"})
		require.NoError(t, err)
		assert.Contains(t, appsSchemas, "io.k8s.api.apps.v1.Deployment")
	})

	t.Run("unknown Kind in existing GroupVersion returns GroupVersionKindNotFoundError", func(t *testing.T) {
		unknownGVK := schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "DoesNotExist"}
		s, err := res.SchemaForGVK(unknownGVK)
		assert.Nil(t, s)
		var gvkErr *GroupVersionKindNotFoundError
		require.ErrorAs(t, err, &gvkErr)
		assert.Equal(t, unknownGVK, gvkErr.GVK)
	})

	t.Run("unknown GroupVersion returns GroupVersionNotFoundError", func(t *testing.T) {
		gv := schema.GroupVersion{Group: "not", Version: "found"}
		_, err := res.SchemaForGVK(gv.WithKind("Foo"))
		var notFoundErr *GroupVersionNotFoundError
		require.ErrorAs(t, err, &notFoundErr)
		assert.Equal(t, gv, notFoundErr.gv)

		_, err = res.SchemasForGV(gv)
		require.ErrorAs(t, err, &notFoundErr)
		assert.Equal(t, gv, notFoundErr.gv)
	})

	t.Run("document with nil Components returns GroupVersionKindNotFoundError and empty SchemasForGV map", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0","paths":{}}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=1",
		}
		r := NewRootAndResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
		_, err := r.SchemaForGVK(gvk)
		var gvkErr *GroupVersionKindNotFoundError
		require.ErrorAs(t, err, &gvkErr)
		assert.Equal(t, gvk, gvkErr.GVK)

		schemas, err := r.SchemasForGV(gvk.GroupVersion())
		require.NoError(t, err)
		require.NotNil(t, schemas)
		assert.Empty(t, schemas)
	})
}

func TestResources_GetConsumes(t *testing.T) {
	res := NewRootAndResources(openapitest.NewEmbeddedFileClient())

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

			// Bodyless GET operation returns nil, nil.
			consumesGet, err := res.GetConsumes(gvk, "GET")
			require.NoError(t, err)
			assert.Nil(t, consumesGet)
		}
	})

	t.Run("CRD without strategic-merge-patch omits strategic-merge-patch and unsupported method returns OperationNotFoundError", func(t *testing.T) {
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
		r := NewRootAndResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

		consumes, err := r.GetConsumes(gvk, "PATCH")
		require.NoError(t, err)
		assert.Equal(t, []string{
			"application/apply-patch+yaml",
			"application/json-patch+json",
			"application/merge-patch+json",
		}, consumes)

		// Unsupported HTTP method on existing GVK returns OperationNotFoundError.
		_, err = r.GetConsumes(gvk, "PUT")
		var opErr *OperationNotFoundError
		require.ErrorAs(t, err, &opErr)
		assert.Equal(t, gvk, opErr.GVK)
		assert.Equal(t, "PUT", opErr.Operation)

		// Unknown GVK in existing GroupVersion returns GroupVersionKindNotFoundError.
		unknownGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Unknown"}
		_, err = r.GetConsumes(unknownGVK, "PATCH")
		var gvkErr *GroupVersionKindNotFoundError
		require.ErrorAs(t, err, &gvkErr)
		assert.Equal(t, unknownGVK, gvkErr.GVK)
	})

	t.Run("schema-only List kind returns OperationNotFoundError for PATCH", func(t *testing.T) {
		listGVK := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ReplicationControllerList"}
		_, err := res.GetConsumes(listGVK, "PATCH")
		var opErr *OperationNotFoundError
		require.ErrorAs(t, err, &opErr)
		assert.Equal(t, listGVK, opErr.GVK)
	})

	t.Run("missing Paths returns invalid OpenAPI v3 document error", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0"}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=empty",
		}
		r := NewRootAndResources(fakeClient)
		_, err := r.GetConsumes(schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}, "PATCH")
		require.ErrorContains(t, err, "invalid OpenAPI v3 document")
	})
}

func TestResources_SupportsQueryParam(t *testing.T) {
	res := NewRootAndResources(openapitest.NewEmbeddedFileClient())

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

			_, err = res.SupportsQueryParam(gvk, "HEAD", "fieldValidation")
			var opErr *OperationNotFoundError
			require.ErrorAs(t, err, &opErr)
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
							{"$ref": "#/parameters/fieldValidationParam"},
							{"$ref": "#/components/parameters/dryRunParam"}
						]
					}
				}
			},
			"components": {
				"parameters": {
					"fieldValidationParam": {
						"name": "fieldValidation",
						"in": "query"
					},
					"dryRunParam": {
						"name": "dryRun",
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
		r := NewRootAndResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

		ok, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.NoError(t, err)
		assert.True(t, ok)

		ok, err = r.SupportsQueryParam(gvk, "PATCH", "dryRun")
		require.NoError(t, err)
		assert.True(t, ok)

		ok, err = r.SupportsQueryParam(gvk, "PATCH", "pathLevelQuery")
		require.NoError(t, err)
		assert.True(t, ok)
	})

	t.Run("nil Paths returns invalid OpenAPI v3 document error, empty Paths map returns GroupVersionKindNotFoundError", func(t *testing.T) {
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0"}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=nilpaths",
		}
		r := NewRootAndResources(fakeClient)
		gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
		_, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.ErrorContains(t, err, "invalid OpenAPI v3 document")

		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{"openapi":"3.0.0","paths":{}}`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=emptypaths",
		}
		_, err = r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		var gvkErr *GroupVersionKindNotFoundError
		require.ErrorAs(t, err, &gvkErr)
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

	res := NewRootAndResources(client)

	// 1. Perform 80 lookups across 7 distinct core/v1 Kinds (simulating a 20-resource apply):
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

		s, err := res.SchemaForGVK(gvk)
		require.NoError(t, err)
		require.NotNil(t, s)

		schemas, err := res.SchemasForGV(gvk.GroupVersion())
		require.NoError(t, err)
		require.NotEmpty(t, schemas)
	}
	assert.Equal(t, int32(1), coreV1Calls.Load(), "core/v1 should only be fetched and unmarshaled once across all lookups")
	assert.Equal(t, int32(0), crdCalls.Load(), "unused CRD GroupVersion should never be fetched")

	// 2. Query CRD v1 (Widget exists, Gadget does not yet exist).
	widgetGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}
	gadgetGVK := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Gadget"}

	s, err := res.SchemaForGVK(widgetGVK)
	require.NoError(t, err)
	require.NotNil(t, s)

	s, err = res.SchemaForGVK(gadgetGVK)
	var gvkErr *GroupVersionKindNotFoundError
	require.ErrorAs(t, err, &gvkErr)
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

	s, err = res.SchemaForGVK(gadgetGVK)
	require.NoError(t, err)
	require.NotNil(t, s, "Gadget should be found after ServerRelativeURL hash change")
	assert.Equal(t, int32(2), crdCalls.Load(), "CRD GroupVersion should be re-fetched once after hash change")
	assert.Equal(t, int32(1), coreV1Calls.Load(), "core/v1 cache must remain warm when CRD hash changes")

	// 4. Empty ServerRelativeURL bypasses caching instead of colliding.
	var emptyURLCalls atomic.Int32
	client.mu.Lock()
	client.pathsMap["apis/example.com/v1"] = &countingGroupVersion{
		delegate:    fakeGroupVersion{spec: []byte(crdSpecV1)},
		relativeURL: "",
		schemaCalls: &emptyURLCalls,
	}
	client.mu.Unlock()
	_, err = res.SchemaForGVK(widgetGVK)
	require.NoError(t, err)
	_, err = res.SchemaForGVK(widgetGVK)
	require.NoError(t, err)
	assert.Equal(t, int32(2), emptyURLCalls.Load(), "empty ServerRelativeURL must not be cached")

	// 5. Simulate CRD deletion removing apis/example.com/v1 from Paths().
	client.mu.Lock()
	delete(client.pathsMap, "apis/example.com/v1")
	client.mu.Unlock()

	_, err = res.SchemaForGVK(widgetGVK)
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

	res := NewRootAndResources(client)

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
			if s, err := res.SchemaForGVK(gvk); err != nil || s == nil {
				errs <- fmt.Errorf("SchemaForGVK(%v): s=%v, err=%w", gvk, s, err)
				return
			}
			if schemas, err := res.SchemasForGV(gvk.GroupVersion()); err != nil || len(schemas) == 0 {
				errs <- fmt.Errorf("SchemasForGV(%v): len=%d, err=%w", gvk.GroupVersion(), len(schemas), err)
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
		r := NewRootAndResources(fakeClient)

		_, err := r.SchemaForGVK(gvk)
		require.ErrorContains(t, err, "paths failure")

		_, err = r.SchemasForGV(gvk.GroupVersion())
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
		r := NewRootAndResources(fakeClient)
		_, err := r.SchemaForGVK(gvk)
		require.ErrorContains(t, err, "schema download error")

		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(`{invalid json`),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=badjson",
		}
		_, err = r.SchemaForGVK(gvk)
		require.Error(t, err)
	})

	t.Run("NewRoot and NewRootAndResources share the underlying root struct implementing both interfaces", func(t *testing.T) {
		client := openapitest.NewEmbeddedFileClient()

		rt := NewRoot(client)
		resFromRoot, ok := rt.(RootAndResources)
		require.True(t, ok, "expected NewRoot to implement RootAndResources")
		s, err := resFromRoot.SchemaForGVK(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"})
		require.NoError(t, err)
		require.NotNil(t, s)

		combined := NewRootAndResources(client)
		gvs, err := combined.GroupVersions()
		require.NoError(t, err)
		require.NotEmpty(t, gvs)
		s2, err := combined.SchemaForGVK(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"})
		require.NoError(t, err)
		require.NotNil(t, s2)
	})

	t.Run("error types format messages", func(t *testing.T) {
		gvErr := &GroupVersionNotFoundError{gv: schema.GroupVersion{Group: "example.com", Version: "v1"}}
		assert.Contains(t, gvErr.Error(), "example.com/v1")

		gvkErr := &GroupVersionKindNotFoundError{GVK: gvk}
		assert.Contains(t, gvkErr.Error(), "Widget")

		opErr := &OperationNotFoundError{GVK: gvk, Operation: "PATCH"}
		assert.Contains(t, opErr.Error(), `"PATCH"`)
		assert.Contains(t, opErr.Error(), "Widget")
	})

	t.Run("extension helpers handle nil, malformed, and typed extensions", func(t *testing.T) {
		_, ok := extractSingleGVKFromExtension(nil)
		assert.False(t, ok)

		_, ok = extractSingleGVKFromExtension(map[string]any{"other": "val"})
		assert.False(t, ok)

		parsedGVK, ok := extractSingleGVKFromExtension(map[string]any{
			groupVersionKindExtensionKey: map[string]string{"group": "apps", "version": "v1", "kind": "Deployment"},
		})
		assert.True(t, ok)
		assert.Equal(t, schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, parsedGVK)

		_, ok = extractSingleGVKFromExtension(map[string]any{
			groupVersionKindExtensionKey: map[string]string{"group": "apps", "version": "v1"},
		})
		assert.False(t, ok)

		_, ok = extractSingleGVKFromExtension(map[string]any{
			groupVersionKindExtensionKey: "not-a-map",
		})
		assert.False(t, ok)

		assert.Nil(t, extractGVKListFromExtension(nil))
		assert.Nil(t, extractGVKListFromExtension(map[string]any{"other": "val"}))
		assert.Nil(t, extractGVKListFromExtension(map[string]any{groupVersionKindExtensionKey: "not-a-list"}))

		gvkList := extractGVKListFromExtension(map[string]any{
			groupVersionKindExtensionKey: []map[string]string{
				{"group": "apps", "version": "v1", "kind": "Deployment"},
				{"group": "apps", "version": "v1"},
			},
		})
		require.Len(t, gvkList, 1)
		assert.Equal(t, schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, gvkList[0])

		entry := buildGVCacheEntry(gvk.GroupVersion(), "url", nil)
		require.NotNil(t, entry)

		entryWithNilPath := buildGVCacheEntry(gvk.GroupVersion(), "url", &spec3.OpenAPI{
			Paths: &spec3.Paths{
				Paths: map[string]*spec3.Path{"/null": nil},
			},
		})
		assert.Empty(t, entryWithNilPath.operationsByGVK)
	})

	t.Run("null schemas, null parameters, and non-map GVK list elements are ignored", func(t *testing.T) {
		docWithNulls := `{
			"openapi": "3.0.0",
			"paths": {
				"/apis/example.com/v1/widgets/{name}": {
					"parameters": [null],
					"patch": {
						"x-kubernetes-group-version-kind": {"group": "example.com", "version": "v1", "kind": "Widget"},
						"parameters": [null, {"name": "fieldValidation", "in": "query"}]
					}
				}
			},
			"components": {
				"schemas": {
					"NullSchema": null,
					"Widget": {
						"type": "object",
						"x-kubernetes-group-version-kind": [
							"not-a-map",
							{"group": "example.com", "version": "v1", "kind": "Widget"}
						]
					}
				}
			}
		}`
		fakeClient := openapitest.NewFakeClient()
		fakeClient.PathsMap["apis/example.com/v1"] = fakeGroupVersion{
			spec:        []byte(docWithNulls),
			relativeURL: "/openapi/v3/apis/example.com/v1?hash=nulls",
		}
		r := NewRootAndResources(fakeClient)

		s, err := r.SchemaForGVK(gvk)
		require.NoError(t, err)
		require.NotNil(t, s)

		ok, err := r.SupportsQueryParam(gvk, "PATCH", "fieldValidation")
		require.NoError(t, err)
		assert.True(t, ok)
	})
}
