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
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/spec3"
)

// Root interface defines functions implemented against the root
// OpenAPI V3 document. The root OpenAPI V3 document maps the
// API Server relative url for all GroupVersions to the relative
// url for the OpenAPI relative url. Example for single GroupVersion
// apps/v1:
//
//	"apis/apps/v1": {
//	    "ServerRelativeURL": "/openapi/v3/apis/apps/v1?hash=<HASH>"
//	}
type Root interface {
	// GroupVersions returns every GroupVersion for which there is an
	// OpenAPI V3 GroupVersion document. Returns an error for problems
	// retrieving or parsing the OpenAPI V3 root document.
	GroupVersions() ([]schema.GroupVersion, error)
	// GVSpec returns the specification for all the resources in a
	// GroupVersion as a pointer to a spec3.OpenAPI struct.
	// Returns an error for problems retrieving or parsing the root
	// document or GroupVersion OpenAPI V3 document.
	GVSpec(gv schema.GroupVersion) (*spec3.OpenAPI, error)
	// GVSpecAsMap returns the specification for all the resources in a
	// GroupVersion as unstructured bytes. Returns an error for
	// problems retrieving or parsing the root or GroupVersion
	// OpenAPI V3 document.
	GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error)
}

// root implements the Root interface, and encapsulates the
// fields to retrieve, store the parsed OpenAPI V3 root document.
type root struct {
	// OpenAPI client to retrieve the OpenAPI V3 documents.
	client openapi.Client
}

// Validate root implements the Root interface.
var _ Root = &root{}

// NewRoot returns a structure implementing the Root interface,
// created with the passed rest client.
func NewRoot(client openapi.Client) Root {
	return &root{client: client}
}

func (r *root) GroupVersions() ([]schema.GroupVersion, error) {
	paths, err := r.client.Paths()
	if err != nil {
		return nil, err
	}
	// Example GroupVersion API path: "apis/apps/v1"
	gvs := make([]schema.GroupVersion, 0, len(paths))
	for gvAPIPath := range paths {
		gv, err := pathToGroupVersion(gvAPIPath)
		if err != nil {
			// Ignore paths which do not parse to GroupVersion
			continue
		}
		gvs = append(gvs, gv)
	}
	// Sort GroupVersions alphabetically
	sort.Slice(gvs, func(i, j int) bool {
		return gvs[i].String() < gvs[j].String()
	})
	return gvs, nil
}

func (r *root) GVSpec(gv schema.GroupVersion) (*spec3.OpenAPI, error) {
	openAPISchemaBytes, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	// Unmarshal the downloaded Group/Version bytes into the spec3.OpenAPI struct.
	var parsedV3Schema spec3.OpenAPI
	err = json.Unmarshal(openAPISchemaBytes, &parsedV3Schema)
	return &parsedV3Schema, err
}

func (r *root) GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error) {
	gvOpenAPIBytes, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	// GroupVersion bytes into unstructured map[string] -> empty interface.
	var gvMap map[string]interface{}
	err = json.Unmarshal(gvOpenAPIBytes, &gvMap)
	return gvMap, err
}

// retrieveGVBytes returns the schema for a passed GroupVersion as an
// unstructured slice of bytes or an error if there is a problem downloading
// or if the passed GroupVersion is not supported.
func (r *root) retrieveGVBytes(gv schema.GroupVersion) ([]byte, error) {
	paths, err := r.client.Paths()
	if err != nil {
		return nil, err
	}
	apiPath := gvToAPIPath(gv)
	gvOpenAPI, found := paths[apiPath]
	if !found {
		return nil, &GroupVersionNotFoundError{gv: gv}
	}
	return gvOpenAPI.Schema(runtime.ContentTypeJSON)
}

// gvToAPIPath maps the passed GroupVersion to a relative api
// server url. Example:
//
//	GroupVersion{Group: "apps", Version: "v1"} -> "apis/apps/v1".
func gvToAPIPath(gv schema.GroupVersion) string {
	var resourcePath string
	if len(gv.Group) == 0 {
		resourcePath = fmt.Sprintf("api/%s", gv.Version)
	} else {
		resourcePath = fmt.Sprintf("apis/%s/%s", gv.Group, gv.Version)
	}
	return resourcePath
}

// pathToGroupVersion is a helper function parsing the passed relative
// url into a GroupVersion.
//
//	Example: apis/apps/v1 -> GroupVersion{Group: "apps", Version: "v1"}
//	Example: api/v1 -> GroupVersion{Group: "", Version: "v1"}
func pathToGroupVersion(path string) (schema.GroupVersion, error) {
	var gv schema.GroupVersion
	parts := strings.Split(path, "/")
	if len(parts) < 2 {
		return gv, fmt.Errorf("Unable to parse api relative path: %s", path)
	}
	apiPrefix := parts[0]
	if apiPrefix == "apis" {
		// Example: apis/apps (without version)
		if len(parts) < 3 {
			return gv, fmt.Errorf("Group without Version not allowed")
		}
		gv.Group = parts[1]
		gv.Version = parts[2]
	} else if apiPrefix == "api" {
		gv.Version = parts[1]
	} else {
		return gv, fmt.Errorf("Unable to parse api relative path: %s", path)
	}
	return gv, nil
}

// Encapsulates GroupVersion not found as one of the paths
// at OpenAPI V3 endpoint.
type GroupVersionNotFoundError struct {
	gv schema.GroupVersion
}

func (r *GroupVersionNotFoundError) Error() string {
	return fmt.Sprintf("GroupVersion (%v) not found as OpenAPI V3 path", r.gv)
}
