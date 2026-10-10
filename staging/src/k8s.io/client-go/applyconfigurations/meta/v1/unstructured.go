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

	"sigs.k8s.io/structured-merge-diff/v7/typed"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	"k8s.io/kube-openapi/pkg/spec3"
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

// gvkParserCache caches one GvkParser per group-version, built from the
// OpenAPI v3 schema of that group-version the first time an object of it is
// extracted, so that only the schemas of group-versions actually extracted
// from are downloaded and parsed.
type gvkParserCache struct {
	// client fetches the OpenAPI v3 discovery listing and per-group-version schemas
	client openapi.Client
	// mu protects the fields below
	mu sync.Mutex
	// paths is the OpenAPI v3 discovery listing, refreshed every openAPISchemaTTL
	paths map[string]openapi.GroupVersion
	// lastChecked is the last time paths was refreshed
	lastChecked time.Time
	// parsers is keyed by the schema URL from the listing, which embeds a hash
	// of the schema: a changed schema has a new URL, so it misses the cache,
	// and parsers whose URL has left the listing are dropped on refresh.
	parsers map[string]*managedfields.GvkParser
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
	// if the ttl on the discovery listing has expired, refresh it to observe
	// schema changes, and drop the parsers of schemas that are gone
	if time.Since(c.lastChecked) > openAPISchemaTTL {
		paths, err := c.client.Paths()
		if err != nil {
			return nil, fmt.Errorf("failed to list openapi v3 group versions: %w", err)
		}
		c.paths = paths
		c.lastChecked = time.Now()
		current := map[string]bool{}
		for _, gvPath := range paths {
			current[gvPath.ServerRelativeURL()] = true
		}
		for url := range c.parsers {
			if !current[url] {
				delete(c.parsers, url)
			}
		}
	}
	gvPath, ok := c.paths[gvPathKey(gvk.GroupVersion())]
	if !ok {
		return nil, fmt.Errorf("no openapi v3 schema found for %v", gvk.GroupVersion())
	}
	url := gvPath.ServerRelativeURL()
	parser, ok := c.parsers[url]
	if !ok {
		data, err := gvPath.Schema("application/json")
		if err != nil {
			return nil, fmt.Errorf("failed to download openapi v3 schema for %v: %w", gvk.GroupVersion(), err)
		}
		var doc spec3.OpenAPI
		if err := json.Unmarshal(data, &doc); err != nil {
			return nil, fmt.Errorf("failed to parse openapi v3 schema for %v: %w", gvk.GroupVersion(), err)
		}
		if doc.Components == nil {
			return nil, fmt.Errorf("openapi v3 schema for %v has no components", gvk.GroupVersion())
		}
		parser, err = managedfields.NewGVKParserFromOpenAPIV3(doc.Components.Schemas, false)
		if err != nil {
			return nil, fmt.Errorf("failed to build type information for %v: %w", gvk.GroupVersion(), err)
		}
		c.parsers[url] = parser
	}
	objectType := parser.Type(gvk)
	if objectType == nil {
		return nil, fmt.Errorf("no type found for %v", gvk)
	}
	return objectType, nil
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
			parsers:     map[string]*managedfields.GvkParser{},
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
