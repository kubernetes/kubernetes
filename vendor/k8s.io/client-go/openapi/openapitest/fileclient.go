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
	"embed"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"k8s.io/client-go/openapi"
)

//go:embed testdata/*_openapi.json
var f embed.FS

// NewFileClient returns a test client implementing the openapi.Client
// interface, which serves a subset of hard-coded GroupVersion
// Open API V3 specifications files. The subset of specifications is
// located in the "testdata" subdirectory.
func NewFileClient(t *testing.T) openapi.Client {
	if t == nil {
		panic("non-nil testing.T required; this package is only for use in tests")
	}
	return &fileClient{t: t}
}

type fileClient struct {
	t     *testing.T
	init  sync.Once
	paths map[string]openapi.GroupVersion
	err   error
}

// fileClient implements the openapi.Client interface.
var _ openapi.Client = &fileClient{}

// Paths returns a map of api path string to openapi.GroupVersion or
// an error. The OpenAPI V3 GroupVersion specifications are hard-coded
// in the "testdata" subdirectory. The api path is derived from the
// spec filename. Example:
//
//	apis__apps__v1_openapi.json -> apis/apps/v1
//
// The file contents are read only once. All files must parse correctly
// into an api path, or an error is returned.
func (t *fileClient) Paths() (map[string]openapi.GroupVersion, error) {
	t.init.Do(func() {
		t.paths = map[string]openapi.GroupVersion{}
		entries, err := f.ReadDir("testdata")
		if err != nil {
			t.err = err
			t.t.Error(err)
		}
		for _, e := range entries {
			// this reverses the transformation done in hack/update-openapi-spec.sh
			path := strings.ReplaceAll(strings.TrimSuffix(e.Name(), "_openapi.json"), "__", "/")
			t.paths[path] = &fileGroupVersion{t: t.t, filename: filepath.Join("testdata", e.Name())}
		}
	})
	return t.paths, t.err
}

type fileGroupVersion struct {
	t        *testing.T
	init     sync.Once
	filename string
	data     []byte
	err      error
}

// fileGroupVersion implements the openapi.GroupVersion interface.
var _ openapi.GroupVersion = &fileGroupVersion{}

// Schema returns the OpenAPI V3 specification for the GroupVersion as
// unstructured bytes, or an error if the contentType is not
// "application/json" or there is an error reading the spec file. The
// file is read only once. The embedded file is located in the "testdata"
// subdirectory.
func (t *fileGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != "application/json" {
		return nil, errors.New("openapitest only supports 'application/json' contentType")
	}
	t.init.Do(func() {
		t.data, t.err = f.ReadFile(t.filename)
		if t.err != nil {
			t.t.Error(t.err)
		}
	})
	return t.data, t.err
}
