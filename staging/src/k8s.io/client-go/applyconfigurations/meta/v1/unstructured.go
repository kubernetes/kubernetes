package v1

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/client-go/discovery"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// UnstructuredExtractor enables extracting the applied configuration state from object for fieldManager into an
// unstructured object type.
type UnstructuredExtractor interface {
	ExtractUnstructured(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
	ExtractUnstructuredStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
}

// gvkParserCache caches the GVKParser in order to prevent from having to repeatedly
// parse the models from the open API schema when the schema itself changes infrequently.
type gvkParserCache struct {
	// discoveryClient is the client for retrieving the openAPI document and checking
	// whether the document has changed recently
	discoveryClient discovery.DiscoveryInterface
	// ttl is how long the openAPI schema should be considered valid
	ttl time.Duration
	// mu protects the gvkParser
	mu sync.Mutex
	// gvkParser retrieves the objectType for a given gvk
	gvkParser *fieldmanager.GvkParser
	// lastChecked is the last time we checked if the openAPI doc has changed.
	lastChecked time.Time
}

// regenerateGVKParser builds the parser from the raw OpenAPI schema.
func (c *gvkParserCache) regenerateGVKParser() error {
	doc, err := c.discoveryClient.OpenAPISchema()
	if err != nil {
		return err
	}
	c.lastChecked = time.Now()
	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		return err
	}

	gvkParser, err := fieldmanager.NewGVKParser(models, false)
	if err != nil {
		return err
	}

	c.gvkParser = gvkParser
	return nil
}

// objectTypeForGVK retrieves the typed.ParseableType for a given gvk from the cache
func (c *gvkParserCache) objectTypeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.gvkParser != nil {
		// if the ttl on the parser cache has expired,
		// recheck the discovery client to see if the Open API schema has changed
		if time.Now().After(c.lastChecked.Add(c.ttl)) {
			c.lastChecked = time.Now()
			if c.discoveryClient.HasOpenAPISchemaChanged() {
				// the schema has changed, regenerate the parser
				if err := c.regenerateGVKParser(); err != nil {
					return nil, err
				}
			}
		}
	} else {
		if err := c.regenerateGVKParser(); err != nil {
			return nil, err
		}
	}
	return c.gvkParser.Type(gvk), nil
}

type extractor struct {
	cache *gvkParserCache
}

// NewUnstructuredExtractor creates the extractor with which you can extract the applied configuration
// for a given manager from an unstructured object.
func NewUnstructuredExtractor(dc discovery.DiscoveryInterface) UnstructuredExtractor {
	// TODO: expose ttl as an argument if we want to.
	defaultTTL := time.Minute
	return &extractor{
		cache: &gvkParserCache{
			discoveryClient: dc,
			ttl:             defaultTTL,
		},
	}
}

// ExtractUnstructured extracts the applied configuration owned by fiieldManager from an unstructured object.
// Note that the apply configuration itself is also an unstructured object.
func (e *extractor) ExtractUnstructured(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "")
}

// ExtractUnstructuredStatus is the same as ExtractUnstructured except
// that it extracts the status subresource applied configuration.
// Experimental!
func (e *extractor) ExtractUnstructuredStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "status")
}

func (e *extractor) extractUnstructured(object *unstructured.Unstructured, fieldManager string, subresource string) (*unstructured.Unstructured, error) {
	gvk := object.GetObjectKind().GroupVersionKind()
	objectType, err := e.cache.objectTypeForGVK(gvk)
	if err != nil {
		return nil, err
	}
	result := &unstructured.Unstructured{}
	err = managedfields.ExtractInto(object, *objectType, fieldManager, result, subresource)
	if err != nil {
		return nil, err
	}
	result.SetName(object.GetName())
	result.SetNamespace(object.GetNamespace())
	result.SetKind(object.GetKind())
	result.SetAPIVersion(object.GetAPIVersion())
	return result, nil
}
