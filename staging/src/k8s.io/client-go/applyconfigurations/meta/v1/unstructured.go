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
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/client-go/discovery"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v6/typed"
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

// gvkParserCache caches the GVKParser in order to prevent from having to repeatedly
// parse the models from the open API schema when the schema itself changes infrequently.
type gvkParserCache struct {
	// discoveryClient is the client for retrieving the openAPI document and checking
	// whether the document has changed recently
	discoveryClient discovery.DiscoveryInterface
	// mu protects the gvkParser
	mu sync.Mutex
	// gvkParser retrieves the objectType for a given gvk
	gvkParser *managedfields.GvkParser
	// lastChecked is the last time we checked if the openAPI doc has changed.
	lastChecked time.Time
}

// regenerateGVKParser builds the parser from the raw OpenAPI schema.
func regenerateGVKParser(dc discovery.DiscoveryInterface) (*managedfields.GvkParser, error) {
	doc, err := dc.OpenAPISchema()
	if err != nil {
		return nil, err
	}

	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return nil, err
	}

	return managedfields.NewGVKParser(models, false)
}

// objectTypeForGVK retrieves the typed.ParseableType for a given gvk from the cache
func (c *gvkParserCache) objectTypeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// if the ttl on the openAPISchema has expired,
	// regenerate the gvk parser
	if time.Since(c.lastChecked) > openAPISchemaTTL {
		c.lastChecked = time.Now()
		parser, err := regenerateGVKParser(c.discoveryClient)
		if err != nil {
			return nil, err
		}
		c.gvkParser = parser
	}
	return c.gvkParser.Type(gvk), nil
}

type extractor struct {
	cache *gvkParserCache
}

// NewUnstructuredExtractor creates the extractor with which you can extract the applied configuration
// for a given manager from an unstructured object.
func NewUnstructuredExtractor(dc discovery.DiscoveryInterface) (UnstructuredExtractor, error) {
	parser, err := regenerateGVKParser(dc)
	if err != nil {
		return nil, fmt.Errorf("failed generating initial GVK Parser: %v", err)
	}
	return &extractor{
		cache: &gvkParserCache{
			gvkParser:       parser,
			discoveryClient: dc,
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
		return nil, fmt.Errorf("failed to fetch the objectType: %v", err)
	}
	result := &unstructured.Unstructured{}
	err = managedfields.ExtractInto(object, *objectType, fieldManager, result, subresource) //nolint:forbidigo
	if err != nil {
		return nil, fmt.Errorf("failed calling ExtractInto for unstructured: %v", err)
	}
	result.SetName(object.GetName())
	result.SetNamespace(object.GetNamespace())
	result.SetKind(object.GetKind())
	result.SetAPIVersion(object.GetAPIVersion())
	return result, nil
}
