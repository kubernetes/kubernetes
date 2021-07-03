package v1

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/client-go/discovery"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

type UnstructuredExtractor interface {
	ExtractUnstructured(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
	ExtractUnstructuredStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error)
}

type parserCache interface {
	parserForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error)
}

type nonCachingParserCache struct {
	discoveryClient *discovery.DiscoveryClient
}

func (c *nonCachingParserCache) parserForGVK(gvk schema.GroupVersionKind) (*typed.ParseableType, error) {
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

	return gvkParser.Type(gvk), nil
}

type extractor struct {
	cache parserCache
}

func NewUnstructuredExtractor(dc *discovery.DiscoveryClient) UnstructuredExtractor {
	return &extractor{
		cache: &nonCachingParserCache{dc},
	}
}

func (e *extractor) ExtractUnstructured(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "")
}

func (e *extractor) ExtractUnstructuredStatus(object *unstructured.Unstructured, fieldManager string) (*unstructured.Unstructured, error) {
	return e.extractUnstructured(object, fieldManager, "status")
}

func (e *extractor) extractUnstructured(object *unstructured.Unstructured, fieldManager string, subresource string) (*unstructured.Unstructured, error) {
	gvk := object.GetObjectKind().GroupVersionKind()
	parser, err := e.cache.parserForGVK(gvk)
	if err != nil {
		return nil, err
	}
	result := &unstructured.Unstructured{}
	err = managedfields.ExtractInto(object, *parser, fieldManager, result, subresource)
	if err != nil {
		return nil, err
	}
	result.SetName(object.GetName())
	result.SetNamespace(object.GetNamespace())
	result.SetKind(object.GetKind())
	result.SetAPIVersion(object.GetAPIVersion())
	return result, nil
}
