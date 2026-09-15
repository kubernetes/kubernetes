/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	smdschema "sigs.k8s.io/structured-merge-diff/v7/schema"
	"sigs.k8s.io/structured-merge-diff/v7/typed"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/schemaconv"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// openAPISchemaTTL is how frequently we need to check
// whether the open API schema has changed or not.
const openAPISchemaTTL = time.Minute

// UnstructuredExtractor enables extracting the applied configuration state from object for fieldManager into an
// unstructured object type.
type UnstructuredExtractor interface {
	Extract(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
	ExtractStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
}

// gvParser holds the parseable types of a single group-version, indexed by GVK.
type gvParser struct {
	types map[schema.GroupVersionKind]*typed.ParseableType
}

func (p *gvParser) typeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {
	objectType, ok := p.types[gvk]
	if !ok {
		return nil, fmt.Errorf("no type found for %v", gvk)
	}
	return objectType, nil
}

// newGVParser builds a gvParser from the JSON-encoded OpenAPI v3 schema of a
// single group-version, the same way managedfields.NewTypeConverter interprets
// such schemas.
func newGVParser(data []byte) (*gvParser, error) {
	var doc spec3.OpenAPI
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("failed to parse openapi v3 schema: %w", err)
	}
	schemas := map[string]*spec.Schema{}
	if doc.Components != nil {
		schemas = doc.Components.Schemas
	}
	typeSchema, err := schemaconv.ToSchemaFromOpenAPI(schemas, false)
	if err != nil {
		return nil, fmt.Errorf("failed to convert openapi v3 schema models: %w", err)
	}
	parser := typed.Parser{Schema: smdschema.Schema{Types: typeSchema.Types}}
	types := map[schema.GroupVersionKind]*typed.ParseableType{}
	for name, model := range schemas {
		for _, gvk := range parseGroupVersionKind(model.Extensions) {
			if gvk.Kind == "" {
				continue
			}
			parsedType := parser.Type(name)
			types[gvk] = &parsedType
		}
	}
	return &gvParser{types: types}, nil
}

// parseGroupVersionKind extracts the "x-kubernetes-group-version-kind" entries
// of a JSON-decoded schema extension map.
func parseGroupVersionKind(extensions spec.Extensions) []schema.GroupVersionKind {
	var result []schema.GroupVersionKind
	gvkExtension, ok := extensions["x-kubernetes-group-version-kind"]
	if !ok {
		return nil
	}
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return nil
	}
	for _, gvk := range gvkList {
		gvkMap, ok := gvk.(map[string]interface{})
		if !ok {
			continue
		}
		group, ok := gvkMap["group"].(string)
		if !ok {
			continue
		}
		version, ok := gvkMap["version"].(string)
		if !ok {
			continue
		}
		kind, ok := gvkMap["kind"].(string)
		if !ok {
			continue
		}
		result = append(result, schema.GroupVersionKind{Group: group, Version: version, Kind: kind})
	}
	return result
}

// gvkParserCache caches one gvParser per group-version, built lazily from the
// OpenAPI v3 schema of that group-version the first time an object of it is
// extracted, to prevent from having to download and parse the models for
// group-versions that are never extracted from.
type gvkParserCache struct {
	// client fetches the per-group-version OpenAPI v3 schemas
	client openapi.Client
	// mu protects the fields below
	mu sync.Mutex
	// paths is the cached OpenAPI v3 discovery listing
	paths map[string]openapi.GroupVersion
	// lastChecked is the last time paths was refreshed
	lastChecked time.Time
	// parsers hold one gvParser per downloaded group-version
	parsers map[schema.GroupVersion]gvkParserCacheEntry
}

type gvkParserCacheEntry struct {
	// serverRelativeURL is the URL the parser was built from; it embeds a
	// content hash, so an unchanged URL means an unchanged schema
	serverRelativeURL string
	parser            *gvParser
}

// gvPathKey returns the key of gv in the OpenAPI v3 discovery listing,
// e.g. "api/v1" or "apis/apps/v1".
func gvPathKey(gv schema.GroupVersion) string {
	if gv.Group == "" {
		return "api/" + gv.Version
	}
	return "apis/" + gv.Group + "/" + gv.Version
}

// objectTypeForGVK retrieves the typed.ParseableType for a given gvk from the cache
func (c *gvkParserCache) objectTypeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// if the ttl on the discovery listing has expired,
	// regenerate it to observe schema updates
	if time.Since(c.lastChecked) > openAPISchemaTTL {
		paths, err := c.client.Paths()
		if err != nil {
			return nil, fmt.Errorf("failed to list openapi v3 group versions: %w", err)
		}
		c.paths = paths
		c.lastChecked = time.Now()
	}
	gv := gvk.GroupVersion()
	gvPath, ok := c.paths[gvPathKey(gv)]
	if !ok {
		return nil, fmt.Errorf("no openapi v3 schema found for %v", gv)
	}
	if entry, ok := c.parsers[gv]; ok && entry.serverRelativeURL == gvPath.ServerRelativeURL() {
		return entry.parser.typeForGVK(gvk)
	}
	data, err := gvPath.Schema("application/json")
	if err != nil {
		return nil, fmt.Errorf("failed to download openapi v3 schema: %w", err)
	}
	parser, err := newGVParser(data)
	if err != nil {
		return nil, err
	}
	c.parsers[gv] = gvkParserCacheEntry{serverRelativeURL: gvPath.ServerRelativeURL(), parser: parser}
	return parser.typeForGVK(gvk)
}

type extractor struct {
	cache *gvkParserCache
}

// NewUnstructuredExtractor creates the extractor with which you can extract the applied configuration
// for a given manager from an unstructured object.
func NewUnstructuredExtractor(dc discovery.DiscoveryInterface) (UnstructuredExtractor, error) {
	client := dc.OpenAPIV3()
	paths, err := client.Paths()
	if err != nil {
		return nil, fmt.Errorf("failed to list openapi v3 group versions: %w", err)
	}
	return &extractor{
		cache: &gvkParserCache{
			client:      client,
			paths:       paths,
			lastChecked: time.Now(),
			parsers:     map[schema.GroupVersion]gvkParserCacheEntry{},
		},
	}, nil
}

// Extract extracts the applied configuration owned by fieldManager from an unstructured object.
// Note that the apply configuration itself is also an unstructured object.
func (e *extractor) Extract(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "")
}

// ExtractStatus is the same as ExtractUnstructured except
// that it extracts the status subresource applied configuration.
// Experimental!
func (e *extractor) ExtractStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "status")
}

func (e *extractor) extractUnstructured(object *unstructured.Unstructured, fieldManager string, subresource string) (*unstructured.Unstructured, error) {
	gvk := object.GetObjectKind().GroupVersionKind()
	objectType, err := e.cache.objectTypeForGVK(gvk)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch the objectType: %w", err)
	}
	result := &unstructured.Unstructured{}
	err = managedfields.ExtractInto(object, *objectType, fieldManager, result, subresource) //nolint:forbidigo
	if err != nil {
		return nil, fmt.Errorf("failed calling ExtractInto for unstructured: %w", err)
	}
	result.SetName(object.GetName())
	result.SetNamespace(object.GetNamespace())
	result.SetKind(object.GetKind())
	result.SetAPIVersion(object.GetAPIVersion())
	return result, nil
}
