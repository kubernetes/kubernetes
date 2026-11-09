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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
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
	//
	// The returned document is cached and shared with every other caller
	// that asks for the same GroupVersion, so it must be treated as
	// read-only. Callers that need to modify it must deep copy it first.
	GVSpec(gv schema.GroupVersion) (*spec3.OpenAPI, error)
	// GVSpecAsMap returns the specification for all the resources in a
	// GroupVersion as unstructured bytes. Returns an error for
	// problems retrieving or parsing the root or GroupVersion
	// OpenAPI V3 document.
	//
	// The returned map is cached and shared with every other caller that
	// asks for the same GroupVersion, so it must be treated as read-only.
	// Writing to it races with other readers and corrupts the cache;
	// callers that need to modify it must deep copy it first.
	GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error)
}

// GVKIndexer is implemented by the Root returned from NewRoot and
// NewRootWithCache. It answers Group/Version/Kind lookups from an index built
// once per GroupVersion document, so a caller that repeatedly asks about kinds
// in one GroupVersion neither re-unmarshals the document nor rescans every
// path and component schema in it.
//
// It is a separate interface rather than part of Root so that existing
// implementations of Root keep compiling. Discover it with a type assertion:
//
//	if indexer, ok := root.(openapi3.GVKIndexer); ok {
//	    paths, err := indexer.GVKPatchPaths(gvk)
//	}
//
// Both methods return slices because a single GroupVersionKind can be declared
// by several paths (a resource and its subresources) and by several component
// schemas. An empty result means the document does not declare that kind. The
// returned slices and the values they point at are shared and must be treated
// as read-only.
type GVKIndexer interface {
	// GVKPatchPaths returns every path in the GroupVersion document whose
	// PATCH operation declares gvk, ordered by path.
	GVKPatchPaths(gvk schema.GroupVersionKind) ([]*spec3.Path, error)
	// GVKSchemas returns every component schema in the GroupVersion
	// document that declares gvk, ordered by schema name.
	GVKSchemas(gvk schema.GroupVersionKind) ([]*spec.Schema, error)
}

// root implements the Root interface, and encapsulates the
// fields to retrieve, store the parsed OpenAPI V3 root document.
type root struct {
	// OpenAPI client to retrieve the OpenAPI V3 documents.
	client openapi.Client

	// cache holds the parsed documents and their GVK indexes. It is never
	// nil and may be shared with other root instances.
	cache *Cache
}

// Validate root implements the Root and GVKIndexer interfaces.
var (
	_ Root       = &root{}
	_ GVKIndexer = &root{}
)

// NewRoot returns a structure implementing the Root interface,
// created with the passed rest client.
//
// Parsed documents are cached, so a caller that reuses one Root unmarshals
// each GroupVersion document at most once per schema hash. If the passed
// client implements CacheProvider, the cache it supplies is used, which lets
// every Root built from one client share a single set of parsed documents.
func NewRoot(client openapi.Client) Root {
	var cache *Cache
	if provider, ok := client.(CacheProvider); ok {
		cache = provider.OpenAPIV3Cache()
	}
	return NewRootWithCache(client, cache)
}

// NewRootWithCache returns a structure implementing the Root interface that
// stores its parsed documents in the passed cache, which may be shared with
// other Root instances to avoid parsing the same document more than once
// across callers. A nil cache gives this Root a private one.
func NewRootWithCache(client openapi.Client, cache *Cache) Root {
	if cache == nil {
		cache = NewCache()
	}
	return &root{client: client, cache: cache}
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
	openAPISchemaBytes, locator, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	// Unmarshal once per schema hash. The result is shared, so callers must
	// not mutate it.
	return r.cache.spec(gv, locator, openAPISchemaBytes)
}

func (r *root) GVSpecAsMap(gv schema.GroupVersion) (map[string]interface{}, error) {
	gvOpenAPIBytes, locator, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	return r.cache.specAsMap(gv, locator, gvOpenAPIBytes)
}

// GVKPatchPaths implements GVKIndexer.
func (r *root) GVKPatchPaths(gvk schema.GroupVersionKind) ([]*spec3.Path, error) {
	gv := gvk.GroupVersion()
	openAPISchemaBytes, locator, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	return r.cache.patchPaths(gv, locator, openAPISchemaBytes, gvk)
}

// GVKSchemas implements GVKIndexer.
func (r *root) GVKSchemas(gvk schema.GroupVersionKind) ([]*spec.Schema, error) {
	gv := gvk.GroupVersion()
	openAPISchemaBytes, locator, err := r.retrieveGVBytes(gv)
	if err != nil {
		return nil, err
	}
	return r.cache.schemas(gv, locator, openAPISchemaBytes, gvk)
}

// retrieveGVBytes returns the schema for a passed GroupVersion as an
// unstructured slice of bytes, the cache key for that schema, or an error
// if there is a problem downloading or if the passed GroupVersion is not
// supported. The cache key is ServerRelativeURL, which includes the schema
// hash. An empty locator falls back to a hash of the bytes.
func (r *root) retrieveGVBytes(gv schema.GroupVersion) ([]byte, string, error) {
	paths, err := r.client.Paths()
	if err != nil {
		return nil, "", err
	}
	apiPath := gvToAPIPath(gv)
	gvOpenAPI, found := paths[apiPath]
	if !found {
		return nil, "", &GroupVersionNotFoundError{gv: gv}
	}
	data, err := gvOpenAPI.Schema(runtime.ContentTypeJSON)
	if err != nil {
		return nil, "", err
	}
	key := gvOpenAPI.ServerRelativeURL()
	if key == "" {
		sum := sha256.Sum256(data)
		key = gv.String() + ":" + hex.EncodeToString(sum[:])
	}
	return data, key, nil
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
