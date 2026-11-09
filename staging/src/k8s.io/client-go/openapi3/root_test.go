/*
Copyright 2023 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

func TestOpenAPIV3Root_GroupVersions(t *testing.T) {
	tests := []struct {
		name        string
		paths       map[string]openapi.GroupVersion
		expectedGVs []schema.GroupVersion
		forcedErr   error
	}{
		{
			name:        "OpenAPI V3 Root: No openapi.Paths() equals no GroupVersions.",
			expectedGVs: []schema.GroupVersion{},
		},
		{
			name: "OpenAPI V3 Root: Single openapi.Path equals one GroupVersion.",
			paths: map[string]openapi.GroupVersion{
				"apis/apps/v1": nil,
			},
			expectedGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
			},
		},
		{
			name: "OpenAPI V3 Root: Multiple openapi.Paths equals multiple GroupVersions.",
			paths: map[string]openapi.GroupVersion{
				"apis/apps/v1":       nil,
				"api/v1":             nil,
				"apis/batch/v1beta1": nil,
			},
			// Alphabetical ordering, since GV's are returned sorted.
			expectedGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
				{Group: "batch", Version: "v1beta1"},
				{Group: "", Version: "v1"},
			},
		},
		{
			name: "Multiple GroupVersions, some invalid",
			paths: map[string]openapi.GroupVersion{
				"apis/batch/v1beta1":              nil,
				"api/v1":                          nil,
				"foo/apps/v1":                     nil, // bad prefix
				"apis/networking.k8s.io/v1alpha1": nil,
				"api":                             nil, // No version
				"apis/apps":                       nil, // Missing Version
				"apis/apps/v1":                    nil,
			},
			// Alphabetical ordering, since GV's are returned sorted.
			expectedGVs: []schema.GroupVersion{
				{Group: "apps", Version: "v1"},
				{Group: "batch", Version: "v1beta1"},
				{Group: "networking.k8s.io", Version: "v1alpha1"},
				{Group: "", Version: "v1"},
			},
		},
		{
			name:      "OpenAPI V3 Root: Forced error returns error.",
			forcedErr: fmt.Errorf("openapi client error"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := openapitest.FakeClient{
				PathsMap:  test.paths,
				ForcedErr: test.forcedErr,
			}
			root := NewRoot(fakeClient)
			actualGVs, err := root.GroupVersions()
			if test.forcedErr != nil {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if !reflect.DeepEqual(test.expectedGVs, actualGVs) {
				t.Errorf("expected GroupVersions (%s), got (%s): (%s)\n",
					test.expectedGVs, actualGVs, err)
			}
		})
	}
}

func TestOpenAPIV3Root_GVSpec(t *testing.T) {
	tests := []struct {
		name          string
		gv            schema.GroupVersion
		expectedPaths []string
		err           error
	}{
		{
			name: "OpenAPI V3 for apps/v1 works",
			gv:   schema.GroupVersion{Group: "apps", Version: "v1"},
			expectedPaths: []string{
				"/apis/apps/v1/",
				"/apis/apps/v1/deployments",
				"/apis/apps/v1/replicasets",
				"/apis/apps/v1/daemonsets",
			},
		},
		{
			name: "OpenAPI V3 for networking/v1alpha1 works",
			gv:   schema.GroupVersion{Group: "networking.k8s.io", Version: "v1alpha1"},
			expectedPaths: []string{
				"/apis/networking.k8s.io/v1alpha1/",
			},
		},
		{
			name: "OpenAPI V3 for batch/v1 works",
			gv:   schema.GroupVersion{Group: "batch", Version: "v1"},
			expectedPaths: []string{
				"/apis/batch/v1/",
				"/apis/batch/v1/jobs",
				"/apis/batch/v1/cronjobs",
			},
		},
		{
			name: "OpenAPI V3 spec not found",
			gv:   schema.GroupVersion{Group: "not", Version: "found"},
			err:  &GroupVersionNotFoundError{gv: schema.GroupVersion{Group: "not", Version: "found"}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := openapitest.NewEmbeddedFileClient()
			root := NewRoot(client)
			gvSpec, err := root.GVSpec(test.gv)
			if test.err != nil {
				assert.True(t, reflect.DeepEqual(test.err, err))
				return
			}
			require.NoError(t, err)
			for _, path := range test.expectedPaths {
				if _, found := gvSpec.Paths.Paths[path]; !found {
					assert.True(t, found, "expected path not found (%s)\n", path)
				}
			}
		})
	}
}

func TestOpenAPIV3Root_GVSpecAsMap(t *testing.T) {
	tests := []struct {
		name          string
		gv            schema.GroupVersion
		expectedPaths []string
		err           error
	}{
		{
			name: "OpenAPI V3 for apps/v1 works",
			gv:   schema.GroupVersion{Group: "apps", Version: "v1"},
			expectedPaths: []string{
				"/apis/apps/v1/",
				"/apis/apps/v1/deployments",
				"/apis/apps/v1/replicasets",
				"/apis/apps/v1/daemonsets",
			},
		},
		{
			name: "OpenAPI V3 for networking/v1alpha1 works",
			gv:   schema.GroupVersion{Group: "networking.k8s.io", Version: "v1alpha1"},
			expectedPaths: []string{
				"/apis/networking.k8s.io/v1alpha1/",
			},
		},
		{
			name: "OpenAPI V3 for batch/v1 works",
			gv:   schema.GroupVersion{Group: "batch", Version: "v1"},
			expectedPaths: []string{
				"/apis/batch/v1/",
				"/apis/batch/v1/jobs",
				"/apis/batch/v1/cronjobs",
			},
		},
		{
			name: "OpenAPI V3 spec not found",
			gv:   schema.GroupVersion{Group: "not", Version: "found"},
			err:  &GroupVersionNotFoundError{gv: schema.GroupVersion{Group: "not", Version: "found"}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := NewRoot(openapitest.NewEmbeddedFileClient())
			gvSpecAsMap, err := root.GVSpecAsMap(test.gv)
			if test.err != nil {
				assert.True(t, reflect.DeepEqual(test.err, err))
				return
			}
			require.NoError(t, err)
			for _, path := range test.expectedPaths {
				pathsMap := gvSpecAsMap["paths"]
				if _, found := pathsMap.(map[string]interface{})[path]; !found {
					assert.True(t, found, "expected path not found (%s)\n", path)
				}
			}
		})
	}
}

func TestOpenAPIV3Root_GroupVersionToPath(t *testing.T) {
	tests := []struct {
		name         string
		groupVersion schema.GroupVersion
		expectedPath string
	}{
		{
			name: "OpenAPI V3 Root: Path to GroupVersion apps group",
			groupVersion: schema.GroupVersion{
				Group:   "apps",
				Version: "v1",
			},
			expectedPath: "apis/apps/v1",
		},
		{
			name: "OpenAPI V3 Root: Path to GroupVersion batch group",
			groupVersion: schema.GroupVersion{
				Group:   "batch",
				Version: "v1beta1",
			},
			expectedPath: "apis/batch/v1beta1",
		},
		{
			name: "OpenAPI V3 Root: Path to GroupVersion core group",
			groupVersion: schema.GroupVersion{
				Version: "v1",
			},
			expectedPath: "api/v1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualPath := gvToAPIPath(test.groupVersion)
			assert.Equal(t, test.expectedPath, actualPath, "expected API path (%s), got (%s)",
				test.expectedPath, actualPath)
		})
	}
}

func TestOpenAPIV3Root_PathToGroupVersion(t *testing.T) {
	tests := []struct {
		name        string
		path        string
		expectedGV  schema.GroupVersion
		expectedErr bool
	}{
		{
			name: "OpenAPI V3 Root: Path to GroupVersion apps/v1 group",
			path: "apis/apps/v1",
			expectedGV: schema.GroupVersion{
				Group:   "apps",
				Version: "v1",
			},
		},
		{
			name:        "Group without Version throws error",
			path:        "apis/apps",
			expectedErr: true,
		},
		{
			name: "OpenAPI V3 Root: Path to GroupVersion batch group",
			path: "apis/batch/v1beta1",
			expectedGV: schema.GroupVersion{
				Group:   "batch",
				Version: "v1beta1",
			},
		},
		{
			name: "OpenAPI V3 Root: Path to GroupVersion core group",
			path: "api/v1",
			expectedGV: schema.GroupVersion{
				Version: "v1",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualGV, err := pathToGroupVersion(test.path)
			if test.expectedErr {
				require.Error(t, err, "should have received error for path: %s", test.path)
			} else {
				require.NoError(t, err, "expected no error, got (%v)", err)
				assert.Equal(t, test.expectedGV, actualGV, "expected GroupVersion (%s), got (%s)",
					test.expectedGV, actualGV)
			}
		})
	}
}

// countingGroupVersion records how often the schema bytes are read.
type countingGroupVersion struct {
	url  string
	body []byte
	err  error
}

func (c *countingGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != runtime.ContentTypeJSON {
		return nil, fmt.Errorf("unsupported content type %s", contentType)
	}
	if c.err != nil {
		return nil, c.err
	}
	return c.body, nil
}

func (c *countingGroupVersion) ServerRelativeURL() string {
	return c.url
}

func TestOpenAPIV3Root_GVSpecReusesParseForSameHash(t *testing.T) {
	gv := schema.GroupVersion{Group: "apps", Version: "v1"}
	body := []byte(`{"openapi":"3.0.0","info":{"title":"apps","version":"v1"},"paths":{"/apis/apps/v1/deployments":{}}}`)
	doc := &countingGroupVersion{
		url:  "/openapi/v3/apis/apps/v1?hash=abc",
		body: body,
	}
	root := NewRoot(openapitest.FakeClient{
		PathsMap: map[string]openapi.GroupVersion{
			"apis/apps/v1": doc,
		},
	})

	first, err := root.GVSpec(gv)
	require.NoError(t, err)
	second, err := root.GVSpec(gv)
	require.NoError(t, err)
	if first != second {
		t.Fatal("GVSpec unmarshaled the GroupVersion document again for the same schema hash")
	}
	if _, found := first.Paths.Paths["/apis/apps/v1/deployments"]; !found {
		t.Fatal("cached spec is missing the deployments path")
	}

	firstMap, err := root.GVSpecAsMap(gv)
	require.NoError(t, err)
	secondMap, err := root.GVSpecAsMap(gv)
	require.NoError(t, err)
	if len(firstMap) == 0 {
		t.Fatal("GVSpecAsMap returned an empty document")
	}
	if fmt.Sprintf("%p", firstMap) != fmt.Sprintf("%p", secondMap) {
		t.Fatal("GVSpecAsMap unmarshaled the GroupVersion document again for the same schema hash")
	}

	doc.url = "/openapi/v3/apis/apps/v1?hash=def"
	doc.body = []byte(`{"openapi":"3.0.0","info":{"title":"apps","version":"v1"},"paths":{"/apis/apps/v1/statefulsets":{}}}`)
	updated, err := root.GVSpec(gv)
	require.NoError(t, err)
	if updated == first {
		t.Fatal("GVSpec reused a parse after ServerRelativeURL changed")
	}
	if _, found := updated.Paths.Paths["/apis/apps/v1/statefulsets"]; !found {
		t.Fatal("refetched spec is missing the statefulsets path")
	}
}

func TestOpenAPIV3Root_GVSpecCacheEdges(t *testing.T) {
	appsGV := schema.GroupVersion{Group: "apps", Version: "v1"}
	batchGV := schema.GroupVersion{Group: "batch", Version: "v1"}
	coreGV := schema.GroupVersion{Version: "v1"}
	appsBody := []byte(`{"openapi":"3.0.0","info":{"title":"apps","version":"v1"},"paths":{"/apis/apps/v1/deployments":{}}}`)
	batchBody := []byte(`{"openapi":"3.0.0","info":{"title":"batch","version":"v1"},"paths":{"/apis/batch/v1/jobs":{}}}`)
	coreBody := []byte(`{"openapi":"3.0.0","info":{"title":"core","version":"v1"},"paths":{"/api/v1/pods":{}}}`)

	t.Run("missing groupversion is not cached as a document", func(t *testing.T) {
		doc := &countingGroupVersion{url: "/openapi/v3/apis/apps/v1?hash=abc", body: appsBody}
		client := openapitest.FakeClient{PathsMap: map[string]openapi.GroupVersion{}}
		root := NewRoot(client)
		_, err := root.GVSpec(appsGV)
		require.ErrorAs(t, err, new(*GroupVersionNotFoundError))

		client.PathsMap["apis/apps/v1"] = doc
		parsed, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		if _, found := parsed.Paths.Paths["/apis/apps/v1/deployments"]; !found {
			t.Fatal("lookup after a miss did not parse the document")
		}
	})

	t.Run("empty locator keys by document bytes", func(t *testing.T) {
		doc := &countingGroupVersion{body: appsBody}
		root := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{"apis/apps/v1": doc},
		})
		first, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		second, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		if first != second {
			t.Fatal("identical bytes with an empty ServerRelativeURL were parsed twice")
		}

		doc.body = batchBody
		updated, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		if updated == first {
			t.Fatal("changed bytes with an empty ServerRelativeURL reused the old parse")
		}
		if _, found := updated.Paths.Paths["/apis/batch/v1/jobs"]; !found {
			t.Fatal("content-hash refresh did not parse the new document")
		}
	})

	t.Run("distinct groupversions do not share a parse", func(t *testing.T) {
		apps := &countingGroupVersion{url: "?hash=same", body: appsBody}
		batch := &countingGroupVersion{url: "?hash=same", body: batchBody}
		core := &countingGroupVersion{url: "/openapi/v3/api/v1?hash=core", body: coreBody}
		root := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{
				"apis/apps/v1":  apps,
				"apis/batch/v1": batch,
				"api/v1":        core,
			},
		})
		appsSpec, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		batchSpec, err := root.GVSpec(batchGV)
		require.NoError(t, err)
		coreSpec, err := root.GVSpec(coreGV)
		require.NoError(t, err)
		if appsSpec == batchSpec || appsSpec == coreSpec || batchSpec == coreSpec {
			t.Fatal("different GroupVersions returned the same parsed document")
		}
		if _, found := appsSpec.Paths.Paths["/apis/apps/v1/deployments"]; !found {
			t.Fatal("apps spec lost its path")
		}
		if _, found := batchSpec.Paths.Paths["/apis/batch/v1/jobs"]; !found {
			t.Fatal("batch spec was served from the apps cache")
		}
		if _, found := coreSpec.Paths.Paths["/api/v1/pods"]; !found {
			t.Fatal("core v1 spec was not parsed from api/v1")
		}
	})

	t.Run("schema fetch error is retried", func(t *testing.T) {
		doc := &countingGroupVersion{url: "/openapi/v3/apis/apps/v1?hash=abc", err: fmt.Errorf("discovery unavailable")}
		root := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{"apis/apps/v1": doc},
		})
		_, err := root.GVSpec(appsGV)
		require.EqualError(t, err, "discovery unavailable")

		doc.err = nil
		doc.body = appsBody
		parsed, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		if _, found := parsed.Paths.Paths["/apis/apps/v1/deployments"]; !found {
			t.Fatal("a failed schema fetch was cached")
		}
	})

	t.Run("invalid json stays cached until the hash changes", func(t *testing.T) {
		doc := &countingGroupVersion{url: "/openapi/v3/apis/apps/v1?hash=bad", body: []byte(`{`)}
		root := NewRoot(openapitest.FakeClient{
			PathsMap: map[string]openapi.GroupVersion{"apis/apps/v1": doc},
		})
		_, err := root.GVSpec(appsGV)
		require.Error(t, err)
		_, err = root.GVSpec(appsGV)
		require.Error(t, err)

		doc.url = "/openapi/v3/apis/apps/v1?hash=good"
		doc.body = appsBody
		parsed, err := root.GVSpec(appsGV)
		require.NoError(t, err)
		if _, found := parsed.Paths.Paths["/apis/apps/v1/deployments"]; !found {
			t.Fatal("a new schema hash did not replace the invalid parse")
		}
	})
}
