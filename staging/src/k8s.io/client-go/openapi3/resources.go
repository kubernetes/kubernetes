/*
Copyright The Kubernetes Authors.

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
	"net/http"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"

// Resources provides GVK-indexed lookups over OpenAPI V3 GroupVersion
// specifications, cached per GroupVersion and invalidated automatically when
// a GroupVersion's ServerRelativeURL (?hash=...) changes.
type Resources interface {
	// SchemaForGVK returns the OpenAPI V3 schema for the given GVK and the
	// GroupVersion's Components.Schemas map (used to resolve $ref references).
	// If the GroupVersion exists in OpenAPI V3 but does not define a schema
	// for the Kind, it returns (nil, schemas, nil).
	SchemaForGVK(gvk schema.GroupVersionKind) (*spec.Schema, map[string]*spec.Schema, error)
	// GetConsumes returns the sorted list of request body media types accepted
	// by the given HTTP operation (for example, "PATCH", "POST", "PUT") for the
	// specified GVK.
	GetConsumes(gvk schema.GroupVersionKind, operation string) ([]string, error)
	// SupportsQueryParam reports whether the given HTTP operation (for example,
	// "PATCH") for the specified GVK accepts the named query parameter (for
	// example, "fieldValidation").
	SupportsQueryParam(gvk schema.GroupVersionKind, operation, param string) (bool, error)
}

// Validate root implements the Resources interface.
var _ Resources = &root{}

// NewResources returns a Resources implementation backed by the provided
// OpenAPI V3 client, caching GVK-indexed GroupVersion metadata keyed by
// ServerRelativeURL.
func NewResources(client openapi.Client) Resources {
	return &root{client: client}
}

// operationInfo holds the indexed request media types and query parameters
// supported by a single (GVK, HTTP method) pair across all paths in a
// GroupVersion's OpenAPI V3 document.
type operationInfo struct {
	consumesSet map[string]struct{}
	consumes    []string
	queryParams map[string]struct{}
}

// gvCacheEntry stores the lightweight GVK-indexed metadata extracted from a
// single GroupVersion's OpenAPI V3 document so the full spec3.OpenAPI wrapper
// can be discarded after indexing.
type gvCacheEntry struct {
	serverRelativeURL string
	hasComponents     bool
	hasPaths          bool
	hasPathItems      bool
	schemas           map[string]*spec.Schema
	schemasByGVK      map[schema.GroupVersionKind]*spec.Schema
	operationsByGVK   map[schema.GroupVersionKind]map[string]*operationInfo
}

// SchemaForGVK returns the OpenAPI V3 schema for the given GVK and the
// GroupVersion's Components.Schemas map for resolving $ref references.
// If the GroupVersion exists but has no schema for the Kind, it returns
// (nil, schemas, nil). If the GroupVersion is not present in OpenAPI V3
// discovery, it returns *GroupVersionNotFoundError.
func (r *root) SchemaForGVK(gvk schema.GroupVersionKind) (*spec.Schema, map[string]*spec.Schema, error) {
	entry, err := r.getGVEntry(gvk.GroupVersion())
	if err != nil {
		return nil, nil, err
	}
	if !entry.hasComponents {
		return nil, nil, fmt.Errorf("OpenAPI V3 Components is nil")
	}
	return entry.schemasByGVK[gvk], entry.schemas, nil
}

// GetConsumes returns a sorted copy of the request body media types accepted
// by the specified HTTP operation (case-insensitive, e.g. "PATCH") for gvk.
// Returns (nil, nil) if the GroupVersion exists with non-empty paths but does
// not define the GVK or operation, or *GroupVersionNotFoundError if the
// GroupVersion is absent from OpenAPI V3 discovery.
func (r *root) GetConsumes(gvk schema.GroupVersionKind, operation string) ([]string, error) {
	entry, err := r.getGVEntry(gvk.GroupVersion())
	if err != nil {
		return nil, err
	}
	if !entry.hasPaths || !entry.hasPathItems {
		return nil, fmt.Errorf("gvk group: %s, version: %s, kind: %s does not exist for OpenAPI V3", gvk.Group, gvk.Version, gvk.Kind)
	}
	ops, ok := entry.operationsByGVK[gvk]
	if !ok {
		return nil, nil
	}
	opInfo, ok := ops[strings.ToUpper(operation)]
	if !ok {
		return nil, nil
	}
	return slices.Clone(opInfo.consumes), nil
}

// SupportsQueryParam reports whether the specified HTTP operation
// (case-insensitive, e.g. "PATCH") for gvk accepts the named query parameter.
// Returns *GroupVersionNotFoundError if the GroupVersion is absent from
// OpenAPI V3 discovery.
func (r *root) SupportsQueryParam(gvk schema.GroupVersionKind, operation, param string) (bool, error) {
	entry, err := r.getGVEntry(gvk.GroupVersion())
	if err != nil {
		return false, err
	}
	if !entry.hasPaths {
		return false, fmt.Errorf("invalid OpenAPI v3 document")
	}
	ops, ok := entry.operationsByGVK[gvk]
	if !ok {
		return false, nil
	}
	opInfo, ok := ops[strings.ToUpper(operation)]
	if !ok {
		return false, nil
	}
	_, supported := opInfo.queryParams[param]
	return supported, nil
}

// getGVEntry returns the cached GVK index for gv, fetching, unmarshaling, and
// indexing the GroupVersion's OpenAPI V3 document only on cache miss or when
// its ServerRelativeURL (?hash=...) changes.
func (r *root) getGVEntry(gv schema.GroupVersion) (*gvCacheEntry, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	paths, err := r.client.Paths()
	if err != nil {
		return nil, err
	}
	apiPath := gvToAPIPath(gv)
	gvOpenAPI, found := paths[apiPath]
	if !found {
		delete(r.cache, gv)
		return nil, &GroupVersionNotFoundError{gv: gv}
	}
	url := gvOpenAPI.ServerRelativeURL()

	if entry, ok := r.cache[gv]; ok && entry.serverRelativeURL == url {
		return entry, nil
	}

	openAPISchemaBytes, err := gvOpenAPI.Schema(runtime.ContentTypeJSON)
	if err != nil {
		return nil, err
	}
	var parsedV3Schema spec3.OpenAPI
	if err := json.Unmarshal(openAPISchemaBytes, &parsedV3Schema); err != nil {
		return nil, err
	}
	entry := buildGVCacheEntry(url, &parsedV3Schema)
	if r.cache == nil {
		r.cache = make(map[schema.GroupVersion]*gvCacheEntry)
	}
	r.cache[gv] = entry
	return entry, nil
}

// buildGVCacheEntry indexes schemas and HTTP operations by GVK from a parsed
// OpenAPI V3 GroupVersion specification.
func buildGVCacheEntry(url string, parsedSpec *spec3.OpenAPI) *gvCacheEntry {
	entry := &gvCacheEntry{
		serverRelativeURL: url,
		schemasByGVK:      make(map[schema.GroupVersionKind]*spec.Schema),
		operationsByGVK:   make(map[schema.GroupVersionKind]map[string]*operationInfo),
	}
	if parsedSpec == nil {
		return entry
	}

	if parsedSpec.Components != nil {
		entry.hasComponents = true
		entry.schemas = parsedSpec.Components.Schemas
		for _, s := range parsedSpec.Components.Schemas {
			if s == nil {
				continue
			}
			for _, gvk := range extractGVKListFromExtension(s.Extensions) {
				entry.schemasByGVK[gvk] = s
			}
		}
	}

	if parsedSpec.Paths != nil {
		entry.hasPaths = true
		entry.hasPathItems = parsedSpec.Paths.Paths != nil
		for _, path := range parsedSpec.Paths.Paths {
			if path == nil {
				continue
			}
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodGet, path.Get)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodPut, path.Put)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodPost, path.Post)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodDelete, path.Delete)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodOptions, path.Options)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodHead, path.Head)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodPatch, path.Patch)
			indexOperation(entry, parsedSpec.Components, path.Parameters, http.MethodTrace, path.Trace)
		}
		for _, ops := range entry.operationsByGVK {
			for _, opInfo := range ops {
				if len(opInfo.consumesSet) > 0 {
					opInfo.consumes = make([]string, 0, len(opInfo.consumesSet))
					for ct := range opInfo.consumesSet {
						opInfo.consumes = append(opInfo.consumes, ct)
					}
					slices.Sort(opInfo.consumes)
				}
				opInfo.consumesSet = nil
			}
		}
	}

	return entry
}

// indexOperation records requestBody content types and query parameters for a
// single OpenAPI V3 operation carrying an x-kubernetes-group-version-kind
// extension.
func indexOperation(entry *gvCacheEntry, components *spec3.Components, pathParams []*spec3.Parameter, method string, op *spec3.Operation) {
	if op == nil {
		return
	}
	gvk, ok := extractSingleGVKFromExtension(op.Extensions)
	if !ok {
		return
	}
	ops, exists := entry.operationsByGVK[gvk]
	if !exists {
		ops = make(map[string]*operationInfo)
		entry.operationsByGVK[gvk] = ops
	}
	opInfo, exists := ops[method]
	if !exists {
		opInfo = &operationInfo{
			consumesSet: make(map[string]struct{}),
			queryParams: make(map[string]struct{}),
		}
		ops[method] = opInfo
	}
	if op.RequestBody != nil && op.RequestBody.Content != nil {
		for contentType := range op.RequestBody.Content {
			opInfo.consumesSet[contentType] = struct{}{}
		}
	}
	addQueryParams(opInfo.queryParams, components, pathParams)
	addQueryParams(opInfo.queryParams, components, op.Parameters)
}

// addQueryParams adds inline and "#/parameters/..."-referenced query parameter
// names from params into dst.
func addQueryParams(dst map[string]struct{}, components *spec3.Components, params []*spec3.Parameter) {
	for _, param := range params {
		if param == nil {
			continue
		}
		if param.In == "query" && param.Name != "" {
			dst[param.Name] = struct{}{}
		}
		if ref := param.Ref.String(); strings.HasPrefix(ref, "#/parameters/") && components != nil {
			k := strings.TrimPrefix(ref, "#/parameters/")
			if globalParam, ok := components.Parameters[k]; ok && globalParam != nil {
				if globalParam.In == "query" && globalParam.Name != "" {
					dst[globalParam.Name] = struct{}{}
				}
			}
		}
	}
}

// extractSingleGVKFromExtension extracts a single GroupVersionKind object from
// an operation's x-kubernetes-group-version-kind extension.
func extractSingleGVKFromExtension(ext spec.Extensions) (schema.GroupVersionKind, bool) {
	if ext == nil {
		return schema.GroupVersionKind{}, false
	}
	raw, ok := ext[groupVersionKindExtensionKey]
	if !ok || raw == nil {
		return schema.GroupVersionKind{}, false
	}
	if m, ok := raw.(map[string]interface{}); ok {
		group, _ := m["group"].(string)
		version, vOk := m["version"].(string)
		kind, kOk := m["kind"].(string)
		if vOk && kOk && kind != "" {
			return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}, true
		}
	}
	var gvkMap map[string]string
	if err := ext.GetObject(groupVersionKindExtensionKey, &gvkMap); err != nil {
		return schema.GroupVersionKind{}, false
	}
	if gvkMap["version"] == "" || gvkMap["kind"] == "" {
		return schema.GroupVersionKind{}, false
	}
	return schema.GroupVersionKind{
		Group:   gvkMap["group"],
		Version: gvkMap["version"],
		Kind:    gvkMap["kind"],
	}, true
}

// extractGVKListFromExtension extracts the list of GroupVersionKinds from a
// component schema's x-kubernetes-group-version-kind extension.
func extractGVKListFromExtension(ext spec.Extensions) []schema.GroupVersionKind {
	if ext == nil {
		return nil
	}
	raw, ok := ext[groupVersionKindExtensionKey]
	if !ok || raw == nil {
		return nil
	}
	if list, ok := raw.([]interface{}); ok {
		gvks := make([]schema.GroupVersionKind, 0, len(list))
		for _, item := range list {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			group, _ := m["group"].(string)
			version, vOk := m["version"].(string)
			kind, kOk := m["kind"].(string)
			if vOk && kOk && kind != "" {
				gvks = append(gvks, schema.GroupVersionKind{Group: group, Version: version, Kind: kind})
			}
		}
		return gvks
	}
	var gvkList []map[string]string
	if err := ext.GetObject(groupVersionKindExtensionKey, &gvkList); err != nil {
		return nil
	}
	gvks := make([]schema.GroupVersionKind, 0, len(gvkList))
	for _, gvkMap := range gvkList {
		if gvkMap["version"] != "" && gvkMap["kind"] != "" {
			gvks = append(gvks, schema.GroupVersionKind{
				Group:   gvkMap["group"],
				Version: gvkMap["version"],
				Kind:    gvkMap["kind"],
			})
		}
	}
	return gvks
}
