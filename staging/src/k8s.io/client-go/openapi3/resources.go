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
// specifications. GroupVersion documents are fetched and indexed lazily on
// demand, cached per GroupVersion keyed by ServerRelativeURL (?hash=...), and
// safe for concurrent use by multiple goroutines.
type Resources interface {
	// SchemaForGVK returns the OpenAPI V3 component schema for gvk.
	// Callers must not mutate the returned schema.
	//
	// Error cases:
	//   - *GroupVersionNotFoundError if gvk's GroupVersion is absent from OpenAPI V3 discovery.
	//   - *GroupVersionKindNotFoundError if the GroupVersion exists in OpenAPI V3 but defines no component schema for gvk.
	//   - another error if retrieving or parsing the OpenAPI V3 discovery or GroupVersion document fails.
	SchemaForGVK(gvk schema.GroupVersionKind) (*spec.Schema, error)

	// SchemasForGV returns the GroupVersion's OpenAPI V3 Components.Schemas map,
	// keyed by component schema name (for example, "io.k8s.api.apps.v1.Deployment",
	// matching "#/components/schemas/<name>" $ref targets). Returns an empty non-nil
	// map if the GroupVersion defines no component schemas. Callers must not mutate
	// the returned map or schemas.
	//
	// Error cases:
	//   - *GroupVersionNotFoundError if gv is absent from OpenAPI V3 discovery.
	//   - another error if retrieving or parsing the OpenAPI V3 discovery or GroupVersion document fails.
	SchemasForGV(gv schema.GroupVersion) (map[string]*spec.Schema, error)

	// GetConsumes returns the sorted list of request body media types accepted
	// by operation (case-insensitive HTTP method, for example "PATCH", "POST",
	// "PUT") for gvk. If gvk supports operation without a requestBody (for
	// example, "GET" or "DELETE"), it returns (nil, nil).
	//
	// Error cases:
	//   - *GroupVersionNotFoundError if gvk's GroupVersion is absent from OpenAPI V3 discovery.
	//   - *GroupVersionKindNotFoundError if the GroupVersion exists in OpenAPI V3 but does not contain gvk.
	//   - *OperationNotFoundError if gvk exists in the GroupVersion document but does not support operation.
	//   - another error if retrieving or parsing the OpenAPI V3 discovery or GroupVersion document fails.
	GetConsumes(gvk schema.GroupVersionKind, operation string) ([]string, error)

	// SupportsQueryParam reports whether operation (case-insensitive HTTP method,
	// for example "PATCH") for gvk accepts the named query parameter (for example,
	// "fieldValidation"). Returns (false, nil) when gvk supports operation without
	// accepting param.
	//
	// Error cases:
	//   - *GroupVersionNotFoundError if gvk's GroupVersion is absent from OpenAPI V3 discovery.
	//   - *GroupVersionKindNotFoundError if the GroupVersion exists in OpenAPI V3 but does not contain gvk.
	//   - *OperationNotFoundError if gvk exists in the GroupVersion document but does not support operation.
	//   - another error if retrieving or parsing the OpenAPI V3 discovery or GroupVersion document fails.
	SupportsQueryParam(gvk schema.GroupVersionKind, operation, param string) (bool, error)
}

// RootAndResources combines the Root and Resources interfaces over OpenAPI V3
// GroupVersion documents, sharing a single underlying client and cache.
type RootAndResources interface {
	Root
	Resources
}

// GroupVersionKindNotFoundError indicates that the GroupVersion OpenAPI V3
// document was retrieved and parsed, but does not contain the requested
// GroupVersionKind.
type GroupVersionKindNotFoundError struct {
	GVK schema.GroupVersionKind
}

func (e *GroupVersionKindNotFoundError) Error() string {
	return fmt.Sprintf("GroupVersionKind (%v) not found in OpenAPI V3 document", e.GVK)
}

// OperationNotFoundError indicates that the GroupVersionKind exists in the
// GroupVersion OpenAPI V3 document, but does not support the requested HTTP
// operation.
type OperationNotFoundError struct {
	GVK       schema.GroupVersionKind
	Operation string
}

func (e *OperationNotFoundError) Error() string {
	return fmt.Sprintf("operation %q not supported for GroupVersionKind (%v) in OpenAPI V3 document", e.Operation, e.GVK)
}

// Validate root implements the RootAndResources interface.
var _ RootAndResources = &root{}

// NewRootAndResources returns a RootAndResources implementation backed by the
// provided OpenAPI V3 client, caching GVK-indexed GroupVersion metadata keyed
// by ServerRelativeURL.
func NewRootAndResources(client openapi.Client) RootAndResources {
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
// and Paths tree can be discarded after indexing.
type gvCacheEntry struct {
	serverRelativeURL string
	// schemas maps OpenAPI component schema names under components.schemas
	// (e.g. "io.k8s.api.apps.v1.Deployment", matching "#/components/schemas/<name>"
	// $ref targets) to their parsed schemas.
	schemas      map[string]*spec.Schema
	schemasByGVK map[schema.GroupVersionKind]*spec.Schema
	// operationsByGVK maps each GVK to a map keyed by uppercase HTTP method
	// (e.g. "GET", "PATCH", "POST", "PUT", "DELETE").
	operationsByGVK map[schema.GroupVersionKind]map[string]*operationInfo
}

// SchemaForGVK returns the OpenAPI V3 component schema for gvk, or
// *GroupVersionNotFoundError if gvk's GroupVersion is absent from OpenAPI V3
// discovery, or *GroupVersionKindNotFoundError if the GroupVersion defines no
// component schema for gvk.
func (r *root) SchemaForGVK(gvk schema.GroupVersionKind) (*spec.Schema, error) {
	entry, err := r.getGVEntry(gvk.GroupVersion())
	if err != nil {
		return nil, err
	}
	s, ok := entry.schemasByGVK[gvk]
	if !ok {
		return nil, &GroupVersionKindNotFoundError{GVK: gvk}
	}
	return s, nil
}

// SchemasForGV returns the GroupVersion's Components.Schemas map (keyed by
// component schema name, matching "#/components/schemas/<name>"), or
// *GroupVersionNotFoundError if gv is absent from OpenAPI V3 discovery.
// Callers must not mutate the returned map or schemas.
func (r *root) SchemasForGV(gv schema.GroupVersion) (map[string]*spec.Schema, error) {
	entry, err := r.getGVEntry(gv)
	if err != nil {
		return nil, err
	}
	if entry.schemas == nil {
		return map[string]*spec.Schema{}, nil
	}
	return entry.schemas, nil
}

// GetConsumes returns a sorted copy of the request body media types accepted
// by operation (case-insensitive, e.g. "PATCH") for gvk. Returns (nil, nil) if
// gvk supports operation without a requestBody, or *GroupVersionNotFoundError,
// *GroupVersionKindNotFoundError, or *OperationNotFoundError if the
// GroupVersion, GVK, or operation is not present.
func (r *root) GetConsumes(gvk schema.GroupVersionKind, operation string) ([]string, error) {
	opInfo, err := r.lookupOperation(gvk, operation)
	if err != nil {
		return nil, err
	}
	return slices.Clone(opInfo.consumes), nil
}

// SupportsQueryParam reports whether operation (case-insensitive, e.g. "PATCH")
// for gvk accepts param. Returns (false, nil) if gvk supports operation without
// accepting param, or *GroupVersionNotFoundError,
// *GroupVersionKindNotFoundError, or *OperationNotFoundError if the
// GroupVersion, GVK, or operation is not present.
func (r *root) SupportsQueryParam(gvk schema.GroupVersionKind, operation, param string) (bool, error) {
	opInfo, err := r.lookupOperation(gvk, operation)
	if err != nil {
		return false, err
	}
	_, supported := opInfo.queryParams[param]
	return supported, nil
}

func (r *root) lookupOperation(gvk schema.GroupVersionKind, operation string) (*operationInfo, error) {
	entry, err := r.getGVEntry(gvk.GroupVersion())
	if err != nil {
		return nil, err
	}
	ops, hasOps := entry.operationsByGVK[gvk]
	if !hasOps {
		if _, hasSchema := entry.schemasByGVK[gvk]; !hasSchema {
			return nil, &GroupVersionKindNotFoundError{GVK: gvk}
		}
		return nil, &OperationNotFoundError{GVK: gvk, Operation: operation}
	}
	opInfo, ok := ops[strings.ToUpper(operation)]
	if !ok {
		return nil, &OperationNotFoundError{GVK: gvk, Operation: operation}
	}
	return opInfo, nil
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

	if url != "" {
		if entry, ok := r.cache[gv]; ok && entry.serverRelativeURL == url {
			return entry, nil
		}
	}

	openAPISchemaBytes, err := gvOpenAPI.Schema(runtime.ContentTypeJSON)
	if err != nil {
		return nil, err
	}
	var parsedV3Schema spec3.OpenAPI
	if err := json.Unmarshal(openAPISchemaBytes, &parsedV3Schema); err != nil {
		return nil, err
	}
	if parsedV3Schema.Paths == nil {
		return nil, fmt.Errorf("invalid OpenAPI v3 document: missing paths")
	}
	entry := buildGVCacheEntry(gv, url, &parsedV3Schema)
	if url != "" {
		if r.cache == nil {
			r.cache = make(map[schema.GroupVersion]*gvCacheEntry)
		}
		r.cache[gv] = entry
	}
	return entry, nil
}

// buildGVCacheEntry indexes schemas and HTTP operations by GVK from a parsed
// OpenAPI V3 GroupVersion specification.
func buildGVCacheEntry(gv schema.GroupVersion, url string, parsedSpec *spec3.OpenAPI) *gvCacheEntry {
	entry := &gvCacheEntry{
		serverRelativeURL: url,
		schemasByGVK:      make(map[schema.GroupVersionKind]*spec.Schema),
		operationsByGVK:   make(map[schema.GroupVersionKind]map[string]*operationInfo),
	}
	if parsedSpec == nil {
		return entry
	}

	if parsedSpec.Components != nil {
		entry.schemas = parsedSpec.Components.Schemas
		for _, s := range parsedSpec.Components.Schemas {
			if s == nil {
				continue
			}
			for _, itemGVK := range extractGVKListFromExtension(s.Extensions) {
				if itemGVK.GroupVersion() == gv {
					entry.schemasByGVK[itemGVK] = s
				}
			}
		}
	}

	if parsedSpec.Paths != nil {
		for _, path := range parsedSpec.Paths.Paths {
			if path == nil {
				continue
			}
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodGet, path.Get)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodPut, path.Put)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodPost, path.Post)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodDelete, path.Delete)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodOptions, path.Options)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodHead, path.Head)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodPatch, path.Patch)
			indexOperation(gv, entry, parsedSpec.Components, path.Parameters, http.MethodTrace, path.Trace)
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
// extension belonging to gv.
func indexOperation(gv schema.GroupVersion, entry *gvCacheEntry, components *spec3.Components, pathParams []*spec3.Parameter, method string, op *spec3.Operation) {
	if op == nil {
		return
	}
	gvk, ok := extractSingleGVKFromExtension(op.Extensions)
	if !ok || gvk.GroupVersion() != gv {
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

// addQueryParams adds inline and "#/components/parameters/..." (or legacy
// "#/parameters/...") referenced query parameter names from params into dst.
func addQueryParams(dst map[string]struct{}, components *spec3.Components, params []*spec3.Parameter) {
	for _, param := range params {
		if param == nil {
			continue
		}
		if param.In == "query" && param.Name != "" {
			dst[param.Name] = struct{}{}
		}
		if ref := param.Ref.String(); ref != "" && components != nil {
			k, ok := strings.CutPrefix(ref, "#/components/parameters/")
			if !ok {
				k, ok = strings.CutPrefix(ref, "#/parameters/")
			}
			if ok {
				if globalParam := components.Parameters[k]; globalParam != nil && globalParam.In == "query" && globalParam.Name != "" {
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
	if m, ok := raw.(map[string]any); ok {
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
	if list, ok := raw.([]any); ok {
		gvks := make([]schema.GroupVersionKind, 0, len(list))
		for _, item := range list {
			m, ok := item.(map[string]any)
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
