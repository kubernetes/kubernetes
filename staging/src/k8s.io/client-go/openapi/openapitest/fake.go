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
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/client-go/openapi"
)

// NewFileClient returns a pointer to a testing
// FileOpenAPIClient, which parses the OpenAPI GroupVersion
// files in the passed directory path. Example GroupVersion
// OpenAPI filename for apps/v1 would look like:
//
//	apis__apps__v1_openapi.json
//
// The current directory housing these hard-coded GroupVersion
// OpenAPI V3 specification files is:
//
//	<K8S_ROOT>/api/openapi-spec/v3
//
// An example to invoke this function for the test files:
//
//	NewFileClient("../../../api/openapi-spec/v3")
//
// This function will search passed directory for files
// with the suffix "_openapi.json". IMPORTANT: If any file in
// the directory does NOT parse correctly, this function will
// panic.
func NewFileClient(fullDirPath string) openapi.Client {
	_, err := os.Stat(fullDirPath)
	if err != nil {
		panic(fmt.Sprintf("Unable to find test file directory: %s\n", fullDirPath))
	}
	files, err := ioutil.ReadDir(fullDirPath)
	if err != nil {
		panic(fmt.Sprintf("Error reading test file directory: %s (%s)\n", err, fullDirPath))
	}
	values := map[string]openapi.GroupVersion{}
	for _, fileInfo := range files {
		filename := fileInfo.Name()
		apiFilename, err := apiFilepath(filename)
		if err != nil {
			panic(fmt.Sprintf("Error translating file to apipath: %s (%s)\n", err, filename))
		}
		fullFilename := filepath.Join(fullDirPath, filename)
		gvFile := fileOpenAPIGroupVersion{filepath: fullFilename}
		values[apiFilename] = gvFile
	}
	return &fileOpenAPIClient{values: values}
}

// fileOpenAPIClient is a testing version implementing the
// openapi.Client interface. This struct stores the hard-coded
// values returned by this file client.
type fileOpenAPIClient struct {
	values map[string]openapi.GroupVersion
}

// fileOpenAPIClient implements the openapi.Client interface.
var _ openapi.Client = &fileOpenAPIClient{}

// Paths returns the hard-coded map of the api server relative URL
// path string to the GroupVersion swagger bytes. An example Path
// string for apps/v1 GroupVersion is:
//
//	apis/apps/v1
func (f fileOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	return f.values, nil
}

// fileOpenAPIGroupVersion is a testing version implementing the
// openapi.GroupVersion interface. This struct stores the full
// filepath to the file storing the hard-coded GroupVersion bytes.
type fileOpenAPIGroupVersion struct {
	filepath string
}

// FileOpenAPIGroupVersion implements the openapi.GroupVersion interface.
var _ openapi.GroupVersion = &fileOpenAPIGroupVersion{}

// Schemas returns the GroupVersion bytes at the stored filepath, or
// an error if one is returned from reading the file. Panics if the
// passed contentType string is not "application/json".
func (f fileOpenAPIGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != "application/json" {
		panic("FileOpenAPI only supports 'application/json' contentType")
	}
	return ioutil.ReadFile(f.filepath)
}

// apiFilepath is a helper function to parse a openapi filename
// and transform it to the corresponding api relative url. This function
// is the inverse of the filenaming for OpenAPI V3 specs in the
// hack/update-openapi-spec.sh
//
// Examples:
//
//		apis__apps__v1_openapi.json -> apis/apps/v1
//		apis__networking.k8s.io__v1alpha1_openapi.json -> apis/networking.k8s.io/v1alpha1
//		api__v1_openapi.json -> api/v1
//	 logs_openapi.json -> logs
func apiFilepath(filename string) (string, error) {
	if !strings.HasSuffix(filename, "_openapi.json") {
		errStr := fmt.Sprintf("Unable to parse openapi v3 spec filename: %s", filename)
		return "", errors.New(errStr)
	}
	filename = strings.TrimSuffix(filename, "_openapi.json")
	filepath := strings.ReplaceAll(filename, "__", "/")
	return filepath, nil
}
