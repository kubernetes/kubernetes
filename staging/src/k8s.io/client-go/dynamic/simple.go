/*
Copyright 2018 The Kubernetes Authors.

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

package dynamic

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
)

type DynamicInterface interface {
	ClusterResource(resource schema.GroupVersionResource) DynamicResourceInterface
	NamespacedResource(resource schema.GroupVersionResource, namespace string) DynamicResourceInterface

	// Deprecated, this isn't how we want to do it
	ClusterSubresource(resource schema.GroupVersionResource, subresource string) DynamicResourceInterface
	// Deprecated, this isn't how we want to do it
	NamespacedSubresource(resource schema.GroupVersionResource, subresource, namespace string) DynamicResourceInterface
}

type DynamicResourceInterface interface {
	Create(obj *unstructured.Unstructured) (*unstructured.Unstructured, error)
	Update(obj *unstructured.Unstructured) (*unstructured.Unstructured, error)
	UpdateStatus(obj *unstructured.Unstructured) (*unstructured.Unstructured, error)
	Delete(name string, options *metav1.DeleteOptions) error
	DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(name string, options metav1.GetOptions) (*unstructured.Unstructured, error)
	List(opts metav1.ListOptions) (*unstructured.UnstructuredList, error)
	Watch(opts metav1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (*unstructured.Unstructured, error)
}

type dynamicClient struct {
	client *rest.RESTClient
}

var _ DynamicInterface = &dynamicClient{}

func NewForConfig(inConfig *rest.Config) (DynamicInterface, error) {
	config := rest.CopyConfig(inConfig)
	// for serializing the options
	config.GroupVersion = &schema.GroupVersion{}
	config.APIPath = "/if-you-see-this-search-for-the-break"
	config.AcceptContentTypes = "application/json"
	config.ContentType = "application/json"
	config.NegotiatedSerializer = basicNegotiatedSerializer{} // this gets used for discovery and error handling types
	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
	}

	restClient, err := rest.RESTClientFor(config)
	if err != nil {
		return nil, err
	}

	return &dynamicClient{client: restClient}, nil
}

type dynamicResourceClient struct {
	client      *dynamicClient
	namespace   string
	resource    schema.GroupVersionResource
	subresource string
}

func (c *dynamicClient) ClusterResource(resource schema.GroupVersionResource) DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource}
}
func (c *dynamicClient) NamespacedResource(resource schema.GroupVersionResource, namespace string) DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource, namespace: namespace}
}

func (c *dynamicClient) ClusterSubresource(resource schema.GroupVersionResource, subresource string) DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource, subresource: subresource}
}
func (c *dynamicClient) NamespacedSubresource(resource schema.GroupVersionResource, subresource, namespace string) DynamicResourceInterface {
	return &dynamicResourceClient{client: c, resource: resource, namespace: namespace, subresource: subresource}
}

func (c *dynamicResourceClient) Create(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	if len(c.subresource) > 0 {
		return nil, fmt.Errorf("create not supported for subresources")
	}

	outBytes, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}

	result := c.client.client.Post().AbsPath(c.makeURLSegments("")...).Body(outBytes).Do()
	if err := result.Error(); err != nil {
		return nil, err
	}

	retBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	uncastObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, retBytes)
	if err != nil {
		return nil, err
	}
	return uncastObj.(*unstructured.Unstructured), nil
}

func (c *dynamicResourceClient) Update(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	outBytes, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}

	result := c.client.client.Put().AbsPath(c.makeURLSegments(accessor.GetName())...).Body(outBytes).Do()
	if err := result.Error(); err != nil {
		return nil, err
	}

	retBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	uncastObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, retBytes)
	if err != nil {
		return nil, err
	}
	return uncastObj.(*unstructured.Unstructured), nil
}

func (c *dynamicResourceClient) UpdateStatus(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}

	result := c.client.client.Put().AbsPath(append(c.makeURLSegments(accessor.GetName()), "status")...).Body(obj).Do()
	uncastObj, err := result.Get()
	if err != nil {
		return nil, err
	}
	return uncastObj.(*unstructured.Unstructured), nil
}

func (c *dynamicResourceClient) Delete(name string, opts *metav1.DeleteOptions) error {
	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	deleteOptionsByte, err := runtime.Encode(deleteOptionsCodec.LegacyCodec(schema.GroupVersion{Version: "v1"}), opts)
	if err != nil {
		return err
	}

	result := c.client.client.Delete().AbsPath(c.makeURLSegments(name)...).Body(deleteOptionsByte).Do()
	return result.Error()
}

func (c *dynamicResourceClient) DeleteCollection(opts *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	if len(c.subresource) > 0 {
		return fmt.Errorf("deletecollection not supported for subresources")
	}

	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	deleteOptionsByte, err := runtime.Encode(deleteOptionsCodec.LegacyCodec(schema.GroupVersion{Version: "v1"}), opts)
	if err != nil {
		return err
	}

	result := c.client.client.Delete().AbsPath(c.makeURLSegments("")...).Body(deleteOptionsByte).SpecificallyVersionedParams(&listOptions, dynamicParameterCodec, versionV1).Do()
	return result.Error()
}

func (c *dynamicResourceClient) Get(name string, opts metav1.GetOptions) (*unstructured.Unstructured, error) {
	result := c.client.client.Get().AbsPath(c.makeURLSegments(name)...).SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	retBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	uncastObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, retBytes)
	if err != nil {
		return nil, err
	}
	return uncastObj.(*unstructured.Unstructured), nil
}

func (c *dynamicResourceClient) List(opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	if len(c.subresource) > 0 {
		return nil, fmt.Errorf("list not supported for subresources")
	}

	result := c.client.client.Get().AbsPath(c.makeURLSegments("")...).SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	retBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	uncastObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, retBytes)
	if err != nil {
		return nil, err
	}
	if list, ok := uncastObj.(*unstructured.UnstructuredList); ok {
		return list, nil
	}

	list, err := uncastObj.(*unstructured.Unstructured).ToList()
	if err != nil {
		return nil, err
	}
	return list, nil
}

func (c *dynamicResourceClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	if len(c.subresource) > 0 {
		return nil, fmt.Errorf("watch not supported for subresources")
	}

	internalGV := schema.GroupVersions{
		{Group: c.resource.Group, Version: runtime.APIVersionInternal},
		// always include the legacy group as a decoding target to handle non-error `Status` return types
		{Group: "", Version: runtime.APIVersionInternal},
	}
	s := &rest.Serializers{
		Encoder: watchNegotiatedSerializerInstance.EncoderForVersion(watchJsonSerializerInfo.Serializer, c.resource.GroupVersion()),
		Decoder: watchNegotiatedSerializerInstance.DecoderToVersion(watchJsonSerializerInfo.Serializer, internalGV),

		RenegotiatedDecoder: func(contentType string, params map[string]string) (runtime.Decoder, error) {
			return watchNegotiatedSerializerInstance.DecoderToVersion(watchJsonSerializerInfo.Serializer, internalGV), nil
		},
		StreamingSerializer: watchJsonSerializerInfo.StreamSerializer.Serializer,
		Framer:              watchJsonSerializerInfo.StreamSerializer.Framer,
	}

	wrappedDecoderFn := func(body io.ReadCloser) streaming.Decoder {
		framer := s.Framer.NewFrameReader(body)
		return streaming.NewDecoder(framer, s.StreamingSerializer)
	}

	opts.Watch = true
	return c.client.client.Get().AbsPath(c.makeURLSegments("")...).
		SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).
		WatchWithSpecificDecoders(wrappedDecoderFn, unstructured.UnstructuredJSONScheme)
}

func (c *dynamicResourceClient) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (*unstructured.Unstructured, error) {
	result := c.client.client.Patch(pt).AbsPath(append(c.makeURLSegments(name), subresources...)...).Body(data).Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	retBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	uncastObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, retBytes)
	if err != nil {
		return nil, err
	}
	return uncastObj.(*unstructured.Unstructured), nil
}

func (c *dynamicResourceClient) makeURLSegments(name string) []string {
	url := []string{}
	if len(c.resource.Group) == 0 {
		url = append(url, "api")
	} else {
		url = append(url, "apis", c.resource.Group)
	}
	url = append(url, c.resource.Version)

	if len(c.namespace) > 0 {
		url = append(url, "namespaces", c.namespace)
	}
	url = append(url, c.resource.Resource)

	if len(name) > 0 {
		url = append(url, name)

		// subresources only work on things with names
		if len(c.subresource) > 0 {
			url = append(url, c.subresource)
		}
	} else {
		if len(c.subresource) > 0 {
			panic("somehow snuck a subresource and an empty name.  programmer error")
		}
	}

	return url
}
