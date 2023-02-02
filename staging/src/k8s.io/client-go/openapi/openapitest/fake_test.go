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

package openapitest

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var openAPIV3SpecDir = "../../../../../../api/openapi-spec/v3"

func TestFileOpenAPIClient_Paths(t *testing.T) {
	// Directory with OpenAPI V3 spec files
	tests := map[string]struct {
		path  string
		found bool
	}{
		"apps/v1 path exists": {
			path:  "apis/apps/v1",
			found: true,
		},
		"core/v1 path exists": {
			path:  "api/v1",
			found: true,
		},
		"batch/v1 path exists": {
			path:  "apis/batch/v1",
			found: true,
		},
		"networking/v1alpha1 path exists": {
			path:  "apis/networking.k8s.io/v1alpha1",
			found: true,
		},
		"discovery/v1 path exists": {
			path:  "apis/discovery.k8s.io/v1",
			found: true,
		},
		"fake path does not exists": {
			path:  "does/not/exist",
			found: false,
		},
	}

	fileClient := NewFileClient(openAPIV3SpecDir)
	paths, err := fileClient.Paths()
	require.NoError(t, err)
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			_, found := paths[tc.path]
			if tc.found {
				require.True(t, found)
			} else {
				require.False(t, found)
			}
		})
	}
}

func TestFileOpenAPIClient_GroupVersions(t *testing.T) {
	tests := map[string]struct {
		path     string
		filename string
	}{
		"apps/v1 groupversion spec validation": {
			path:     "apis/apps/v1",
			filename: "apis__apps__v1_openapi.json",
		},
		"core/v1 groupversion spec validation": {
			path:     "api/v1",
			filename: "api__v1_openapi.json",
		},
		"networking/v1alpha1 groupversion spec validation": {
			path:     "apis/networking.k8s.io/v1alpha1",
			filename: "apis__networking.k8s.io__v1alpha1_openapi.json",
		},
	}

	fileClient := NewFileClient(openAPIV3SpecDir)
	paths, err := fileClient.Paths()
	require.NoError(t, err)
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			gv, found := paths[tc.path]
			require.True(t, found)
			actualBytes, err := gv.Schema("application/json")
			require.NoError(t, err)
			expectedBytes, err := ioutil.ReadFile(
				filepath.Join(openAPIV3SpecDir, tc.filename))
			require.NoError(t, err)
			assert.Equal(t, expectedBytes, actualBytes)
		})
	}
}

func TestFileOpenAPIClient_apiFilePath(t *testing.T) {
	tests := map[string]struct {
		filename string
		expected string
		isError  bool
	}{
		"apps/v1 filename": {
			filename: "apis__apps__v1_openapi.json",
			expected: "apis/apps/v1",
		},
		"core/v1 filename": {
			filename: "api__v1_openapi.json",
			expected: "api/v1",
		},
		"logs filename": {
			filename: "logs_openapi.json",
			expected: "logs",
		},
		"api filename": {
			filename: "api_openapi.json",
			expected: "api",
		},
		"unversioned autoscaling filename": {
			filename: "apis__autoscaling_openapi.json",
			expected: "apis/autoscaling",
		},
		"networking/v1alpha1 filename": {
			filename: "apis__networking.k8s.io__v1alpha1_openapi.json",
			expected: "apis/networking.k8s.io/v1alpha1",
		},
		"batch/v1beta1 filename": {
			filename: "apis__batch__v1beta1_openapi.json",
			expected: "apis/batch/v1beta1",
		},
		"non-JSON suffix is invalid": {
			filename: "apis__networking.k8s.io__v1alpha1_openapi.yaml",
			isError:  true,
		},
		"missing final openapi before suffix is invalid": {
			filename: "apis__apps__v1_something.json",
			isError:  true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			actual, err := apiFilepath(tc.filename)
			if !tc.isError {
				require.NoError(t, err)
				assert.Equal(t, tc.expected, actual)
			} else {
				require.Error(t, err)
			}
		})
	}
}
