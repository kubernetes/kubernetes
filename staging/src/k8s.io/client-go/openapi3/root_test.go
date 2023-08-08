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
