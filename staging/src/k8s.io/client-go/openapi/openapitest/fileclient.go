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
	"io/fs"
	"os"
	"strings"

	"k8s.io/client-go/openapi"
)

//go:embed testdata/*_openapi.json
var embedded embed.FS

// NewFileClient returns a test client implementing the openapi.Client
// interface, which serves Open API V3 specifications files from the
// given path, as prepared in `api/openapi-spec/v3`.
func NewFileClient(path string) openapi.Client {
	return &fileClient{f: os.DirFS(path)}
}

// NewEmbeddedFileClient returns a test client that uses the embedded
// `testdata` openapi files.
func NewEmbeddedFileClient() openapi.Client {
	f, err := fs.Sub(embedded, "testdata")
	if err != nil {
		panic(err)
	}
	return &fileClient{f: f}
}

type fileClient struct {
	f fs.FS
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
func (f *fileClient) Paths() (map[string]openapi.GroupVersion, error) {
	paths := map[string]openapi.GroupVersion{}
	entries, err := fs.ReadDir(f.f, ".")
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		// this reverses the transformation done in hack/update-openapi-spec.sh
		path := strings.ReplaceAll(strings.TrimSuffix(e.Name(), "_openapi.json"), "__", "/")
		paths[path] = &fileGroupVersion{f: f.f, filename: e.Name()}
	}
	return paths, nil
}

type fileGroupVersion struct {
	f        fs.FS
	filename string
}

// fileGroupVersion implements the openapi.GroupVersion interface.
var _ openapi.GroupVersion = &fileGroupVersion{}

// Schema returns the OpenAPI V3 specification for the GroupVersion as
// unstructured bytes, or an error if the contentType is not
// "application/json" or there is an error reading the spec file. The
// file is read only once.
func (f *fileGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != "application/json" {
		return nil, errors.New("openapitest only supports 'application/json' contentType")
	}
	return fs.ReadFile(f.f, f.filename)
}

// ServerRelativeURL returns an empty string.
func (f *fileGroupVersion) ServerRelativeURL() string {
	return f.filename
}
