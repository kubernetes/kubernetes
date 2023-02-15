/*
Copyright 2022 The Kubernetes Authors.

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

package v2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
)

var apiGroupsPath = "apis/discovery.k8s.io/v1"

var apiGroupsGVR = schema.GroupVersionResource{
	Group:    "discovery.k8s.io",
	Version:  "v1",
	Resource: "apigroups",
}

func TestExplainErrors(t *testing.T) {
	var buf bytes.Buffer

	// Validate error when "Paths()" returns error.
	err := PrintModelDescription(nil, &buf, &forceErrorClient{}, apiGroupsGVR, false, "unknown-format")
	require.ErrorContains(t, err, "failed to fetch list of groupVersions")

	// Validate error when GVR is not found in returned paths map.
	emptyClient := &emptyPathsClient{}
	err = PrintModelDescription(nil, &buf, emptyClient, schema.GroupVersionResource{
		Group:    "test0.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "could not locate schema")

	// Validate error when GroupVersion "Schema()" call returns error.
	fakeClient := &fakeOpenAPIClient{values: make(map[string]openapi.GroupVersion)}
	fakeClient.values["apis/test1.example.com/v1"] = &forceErrorGV{}
	err = PrintModelDescription(nil, &buf, fakeClient, schema.GroupVersionResource{
		Group:    "test1.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "failed to fetch openapi schema ")

	// Validate error when returned bytes from GroupVersion "Schema" are invalid.
	fakeClient.values["apis/test2.example.com/v1"] = &parseErrorGV{}
	err = PrintModelDescription(nil, &buf, fakeClient, schema.GroupVersionResource{
		Group:    "test2.example.com",
		Version:  "v1",
		Resource: "doesntmatter",
	}, false, "unknown-format")
	require.ErrorContains(t, err, "failed to parse openapi schema")

	// Validate error when render template is not recognized.
	client := openapitest.NewFileClient(t)
	err = PrintModelDescription(nil, &buf, client, apiGroupsGVR, false, "unknown-format")
	require.ErrorContains(t, err, "unrecognized format: unknown-format")
}

// Shows that the correct GVR is fetched from the open api client when
// given to explain
func TestExplainOpenAPIClient(t *testing.T) {
	var buf bytes.Buffer

	fileClient := openapitest.NewFileClient(t)
	paths, err := fileClient.Paths()
	require.NoError(t, err)
	gv, found := paths[apiGroupsPath]
	require.True(t, found)
	discoveryBytes, err := gv.Schema("application/json")
	require.NoError(t, err)

	var doc map[string]interface{}
	err = json.Unmarshal(discoveryBytes, &doc)
	require.NoError(t, err)

	gen := NewGenerator()
	err = gen.AddTemplate("Context", "{{ toJson . }}")
	require.NoError(t, err)

	expectedContext := TemplateContext{
		Document:  doc,
		GVR:       apiGroupsGVR,
		Recursive: false,
		FieldPath: nil,
	}

	err = printModelDescriptionWithGenerator(gen, nil, &buf, fileClient, apiGroupsGVR, false, "Context")
	require.NoError(t, err)

	var actualContext TemplateContext
	err = json.Unmarshal(buf.Bytes(), &actualContext)
	require.NoError(t, err)
	require.Equal(t, expectedContext, actualContext)
}

// forceErrorClient always returns an error for "Paths()".
type forceErrorClient struct{}

func (f *forceErrorClient) Paths() (map[string]openapi.GroupVersion, error) {
	return nil, fmt.Errorf("Always fails")
}

// emptyPathsClient returns an empty map for "Paths()".
type emptyPathsClient struct{}

func (f *emptyPathsClient) Paths() (map[string]openapi.GroupVersion, error) {
	return map[string]openapi.GroupVersion{}, nil
}

// fakeOpenAPIClient returns hard-coded map for "Paths()".
type fakeOpenAPIClient struct {
	values map[string]openapi.GroupVersion
}

func (f *fakeOpenAPIClient) Paths() (map[string]openapi.GroupVersion, error) {
	return f.values, nil
}

// forceErrorGV always returns an error for "Schema()".
type forceErrorGV struct{}

func (f *forceErrorGV) Schema(contentType string) ([]byte, error) {
	return nil, fmt.Errorf("Always fails")
}

// parseErrorGV always returns invalid JSON for "Schema()".
type parseErrorGV struct{}

func (f *parseErrorGV) Schema(contentType string) ([]byte, error) {
	return []byte(`<some invalid json!>`), nil
}
