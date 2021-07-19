package v1

import (
	"fmt"

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

//// objectTypeCache is a cache of typed.ParseableTypes
//type objectTypeCache interface {
//	objectTypeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error)
//}

// objectTypeCache is a objectTypeCache that does no caching
// (i.e. it downloads the OpenAPISchema every time)
// Useful during the proof-of-concept stage until we agree on a caching solution.
type objectTypeCache struct {
	// TODO: lock this?
	discoveryClient discovery.DiscoveryInterface
	gvkParser       *fieldmanager.GvkParser
}

// objectTypeForGVK retrieves the typed.ParseableType for a given gvk from the cache
func (c *objectTypeCache) objectTypeForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {

	if !c.discoveryClient.HasOpenAPISchemaChanged() && c.gvkParser != nil {
		// cache hit
		fmt.Println("cache hit")
		fmt.Printf("gvk = %+v\n", gvk)
		return c.gvkParser.Type(gvk), nil
	} else {
		// cache miss
		fmt.Println("cache miss")
		fmt.Printf("gvk = %+v\n", gvk)
		doc, err := c.discoveryClient.OpenAPISchema()
		if err != nil {
			return nil, err
		}
		models, err := proto.NewOpenAPIData(doc)
		if err != nil {
			return nil, err
		}

		gvkParser, err := fieldmanager.NewGVKParser(models, false)
		if err != nil {
			return nil, err
		}

		objType := gvkParser.Type(gvk)
		c.gvkParser = gvkParser

		return objType, nil
	}
}

type extractor struct {
	cache *objectTypeCache
}

// NewUnstructuredExtractor creates the extractor with which you can extract the applied configuration
// for a given manager from an unstructured object.
func NewUnstructuredExtractor(dc discovery.DiscoveryInterface) UnstructuredExtractor {
	return &extractor{
		cache: &objectTypeCache{
			discoveryClient: dc,
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
