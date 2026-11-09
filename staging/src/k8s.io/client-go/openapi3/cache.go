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
	"sort"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// gvkExtensionKey is the OpenAPI vendor extension that records which
// Group/Version/Kind an operation or component schema belongs to.
const gvkExtensionKey = "x-kubernetes-group-version-kind"

// cacheKey identifies one parsed GroupVersion document. The locator is the
// document's ServerRelativeURL, which embeds the schema hash, so an entry
// stays valid exactly as long as the served document is unchanged.
type cacheKey struct {
	gv      schema.GroupVersion
	locator string
}

// parsedGV holds everything derived from one GroupVersion document: the
// parsed spec, the same document as an unstructured map, and a lazily built
// Group/Version/Kind index. Each is computed at most once per locator.
type parsedGV struct {
	spec    *spec3.OpenAPI
	specErr error
	parsed  bool

	asMap  map[string]interface{}
	mapErr error
	mapped bool

	indexed    bool
	patchPaths map[schema.GroupVersionKind][]*spec3.Path
	schemas    map[schema.GroupVersionKind][]*spec.Schema
}

// Cache stores parsed OpenAPI V3 GroupVersion documents so that repeated
// lookups do not unmarshal the same document again. Entries are keyed by
// GroupVersion together with the document locator (ServerRelativeURL, which
// embeds the schema hash); when a GroupVersion locator changes, the superseded
// entry is dropped, so at most one parse per GroupVersion is retained.
//
// A Cache is safe for concurrent use and may be shared by any number of Root
// instances via NewRootWithCache. Share a Cache only among Roots that talk to
// the same API server: documents from different servers do not corrupt each
// other, but they evict each other and so defeat the cache.
//
// A Cache holds its parsed documents for as long as it is reachable and
// performs no time-based expiry, so scope it to the work that needs it (a
// command invocation, say) rather than to the lifetime of a long-running
// process.
type Cache struct {
	mu      sync.Mutex
	entries map[cacheKey]*parsedGV
	current map[schema.GroupVersion]string
}

// NewCache returns an empty Cache ready to be shared between Root instances.
func NewCache() *Cache {
	return &Cache{}
}

// CacheProvider may be implemented by an openapi.Client to hand every Root
// built from that client the same parse cache. NewRoot checks for it, so
// callers that already share one client automatically share one cache without
// changing how they construct a Root.
type CacheProvider interface {
	// OpenAPIV3Cache returns the cache Roots built from this client should
	// use. Returning nil is allowed and gives each Root a private cache.
	OpenAPIV3Cache() *Cache
}

// entryFor returns the entry for gv at locator, dropping any entry this
// GroupVersion previously had at a different locator. The caller must hold
// c.mu.
func (c *Cache) entryFor(gv schema.GroupVersion, locator string) *parsedGV {
	if c.entries == nil {
		c.entries = make(map[cacheKey]*parsedGV)
		c.current = make(map[schema.GroupVersion]string)
	}
	if previous, ok := c.current[gv]; ok && previous != locator {
		delete(c.entries, cacheKey{gv: gv, locator: previous})
	}
	c.current[gv] = locator
	key := cacheKey{gv: gv, locator: locator}
	entry := c.entries[key]
	if entry == nil {
		entry = &parsedGV{}
		c.entries[key] = entry
	}
	return entry
}

// spec returns the parsed document for gv, unmarshaling data only the first
// time this locator is seen. A parse failure is recorded and returned for the
// life of the entry, since re-parsing identical bytes cannot succeed.
func (c *Cache) spec(gv schema.GroupVersion, locator string, data []byte) (*spec3.OpenAPI, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := c.entryFor(gv, locator)
	if !entry.parsed {
		var parsed spec3.OpenAPI
		entry.specErr = json.Unmarshal(data, &parsed)
		if entry.specErr == nil {
			entry.spec = &parsed
		}
		entry.parsed = true
	}
	return entry.spec, entry.specErr
}

// specAsMap returns the document for gv as an unstructured map, unmarshaling
// data only the first time this locator is seen.
func (c *Cache) specAsMap(gv schema.GroupVersion, locator string, data []byte) (map[string]interface{}, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := c.entryFor(gv, locator)
	if !entry.mapped {
		var asMap map[string]interface{}
		entry.mapErr = json.Unmarshal(data, &asMap)
		if entry.mapErr == nil {
			entry.asMap = asMap
		}
		entry.mapped = true
	}
	return entry.asMap, entry.mapErr
}

// patchPaths returns the PATCH-capable paths that declare gvk, building the
// index on first use. The returned slice is shared and must not be modified.
func (c *Cache) patchPaths(gv schema.GroupVersion, locator string, data []byte, gvk schema.GroupVersionKind) ([]*spec3.Path, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := c.entryFor(gv, locator)
	if err := indexLocked(entry, data); err != nil {
		return nil, err
	}
	return entry.patchPaths[gvk], nil
}

// schemas returns the component schemas that declare gvk, building the index
// on first use. The returned slice is shared and must not be modified.
func (c *Cache) schemas(gv schema.GroupVersion, locator string, data []byte, gvk schema.GroupVersionKind) ([]*spec.Schema, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry := c.entryFor(gv, locator)
	if err := indexLocked(entry, data); err != nil {
		return nil, err
	}
	return entry.schemas[gvk], nil
}

// indexLocked makes sure entry holds a parsed document and a GVK index. The
// caller must hold the Cache mutex.
func indexLocked(entry *parsedGV, data []byte) error {
	if !entry.parsed {
		var parsed spec3.OpenAPI
		entry.specErr = json.Unmarshal(data, &parsed)
		if entry.specErr == nil {
			entry.spec = &parsed
		}
		entry.parsed = true
	}
	if entry.specErr != nil {
		return entry.specErr
	}
	if entry.indexed {
		return nil
	}
	entry.patchPaths, entry.schemas = buildGVKIndex(entry.spec)
	entry.indexed = true
	return nil
}

// buildGVKIndex walks a parsed document once and groups its PATCH-capable
// paths and its component schemas by the Group/Version/Kind they declare.
//
// A single GroupVersionKind can appear on more than one path, because
// subresources such as .../pods/{name}/status carry the same extension as the
// resource itself, and likewise on more than one component schema. Both are
// therefore kept as slices ordered by their key in the document, so that a
// caller sees every candidate in a stable order rather than whichever one Go
// map iteration happened to yield first.
func buildGVKIndex(doc *spec3.OpenAPI) (map[schema.GroupVersionKind][]*spec3.Path, map[schema.GroupVersionKind][]*spec.Schema) {
	paths := make(map[schema.GroupVersionKind][]*spec3.Path)
	schemas := make(map[schema.GroupVersionKind][]*spec.Schema)
	if doc == nil {
		return paths, schemas
	}

	if doc.Paths != nil {
		for _, name := range sortedKeys(doc.Paths.Paths) {
			path := doc.Paths.Paths[name]
			if path == nil || path.Patch == nil {
				continue
			}
			// PATCH operations carry a single GVK object.
			if gvk, ok := singleGVKExtension(path.Patch.Extensions); ok {
				paths[gvk] = append(paths[gvk], path)
			}
		}
	}

	if doc.Components != nil {
		for _, name := range sortedKeys(doc.Components.Schemas) {
			component := doc.Components.Schemas[name]
			if component == nil {
				continue
			}
			// Component schemas carry a list of GVKs.
			for _, gvk := range listGVKExtension(component.Extensions) {
				schemas[gvk] = append(schemas[gvk], component)
			}
		}
	}

	return paths, schemas
}

// sortedKeys returns the keys of m in increasing order, so that index
// construction does not depend on map iteration order.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// singleGVKExtension reads the GVK extension stored as a single object, the
// form used on operations.
func singleGVKExtension(extensions spec.Extensions) (schema.GroupVersionKind, bool) {
	var value map[string]string
	if err := extensions.GetObject(gvkExtensionKey, &value); err != nil || value == nil {
		return schema.GroupVersionKind{}, false
	}
	return schema.GroupVersionKind{
		Group:   value["group"],
		Version: value["version"],
		Kind:    value["kind"],
	}, true
}

// listGVKExtension reads the GVK extension stored as a list of objects, the
// form used on component schemas.
func listGVKExtension(extensions spec.Extensions) []schema.GroupVersionKind {
	var values []map[string]string
	if err := extensions.GetObject(gvkExtensionKey, &values); err != nil {
		return nil
	}
	gvks := make([]schema.GroupVersionKind, 0, len(values))
	for _, value := range values {
		gvks = append(gvks, schema.GroupVersionKind{
			Group:   value["group"],
			Version: value["version"],
			Kind:    value["kind"],
		})
	}
	return gvks
}
