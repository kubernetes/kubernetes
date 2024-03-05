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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/openapi"
)

// FakeClient implements openapi.Client interface, with hard-coded
// return values, including the possibility to force errors.
type FakeClient struct {
	// Hard-coded paths to return from Paths() function.
	PathsMap map[string]openapi.GroupVersion
	// Hard-coded returned error.
	ForcedErr error
}

// Validate FakeClient implements openapi.Client interface.
var _ openapi.Client = &FakeClient{}

// NewFakeClient returns a fake openapi client with an empty PathsMap.
func NewFakeClient() *FakeClient {
	return &FakeClient{PathsMap: make(map[string]openapi.GroupVersion)}
}

// Paths returns stored PathsMap field, creating an empty one if
// it does not already exist. If ForcedErr is set, this function
// returns the error instead.
func (f FakeClient) Paths() (map[string]openapi.GroupVersion, error) {
	if f.ForcedErr != nil {
		return nil, f.ForcedErr
	}
	return f.PathsMap, nil
}

// FakeGroupVersion implements openapi.GroupVersion with hard-coded
// return GroupVersion specification bytes. If ForcedErr is set, then
// "Schema()" function returns the error instead of the GVSpec.
type FakeGroupVersion struct {
	// Hard-coded GroupVersion specification
	GVSpec []byte
	// Hard-coded returned error.
	ForcedErr error
}

// FileOpenAPIGroupVersion implements the openapi.GroupVersion interface.
var _ openapi.GroupVersion = &FakeGroupVersion{}

// Schema returns the hard-coded byte slice, including creating an
// empty slice if it has not been set yet. If the ForcedErr is set,
// this function returns the error instead of the GVSpec field. If
// content type other than application/json is passed, and error is
// returned.
func (f FakeGroupVersion) Schema(contentType string) ([]byte, error) {
	if contentType != runtime.ContentTypeJSON {
		return nil, fmt.Errorf("application/json is only content type supported: %s", contentType)
	}
	if f.ForcedErr != nil {
		return nil, f.ForcedErr
	}
	return f.GVSpec, nil
}

// ServerRelativeURL returns an empty string.
func (f FakeGroupVersion) ServerRelativeURL() string {
	panic("unimplemented")
}
