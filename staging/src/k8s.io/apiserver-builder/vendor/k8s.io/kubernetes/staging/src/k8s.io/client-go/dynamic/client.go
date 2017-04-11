/*
Copyright 2016 The Kubernetes Authors.

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

// Package dynamic provides a client interface to arbitrary Kubernetes
// APIs that exposes common high level operations and exposes common
// metadata.
package dynamic

import (
	"encoding/json"
	"errors"
	"io"
	"net/url"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/conversion/queryparams"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
)

// Client is a Kubernetes client that allows you to access metadata
// and manipulate metadata of a Kubernetes API group.
type Client struct {
	cl             *restclient.RESTClient
	parameterCodec runtime.ParameterCodec
}

// NewClient returns a new client based on the passed in config. The
// codec is ignored, as the dynamic client uses it's own codec.
func NewClient(conf *restclient.Config) (*Client, error) {
	// avoid changing the original config
	confCopy := *conf
	conf = &confCopy

	contentConfig := ContentConfig()
	contentConfig.GroupVersion = conf.GroupVersion
	if conf.NegotiatedSerializer != nil {
		contentConfig.NegotiatedSerializer = conf.NegotiatedSerializer
	}
	conf.ContentConfig = contentConfig

	if conf.APIPath == "" {
		conf.APIPath = "/api"
	}

	if len(conf.UserAgent) == 0 {
		conf.UserAgent = restclient.DefaultKubernetesUserAgent()
	}

	cl, err := restclient.RESTClientFor(conf)
	if err != nil {
		return nil, err
	}

	return &Client{cl: cl}, nil
}

// GetRateLimiter returns rate limier.
func (c *Client) GetRateLimiter() flowcontrol.RateLimiter {
	return c.cl.GetRateLimiter()
}

// Resource returns an API interface to the specified resource for this client's
// group and version. If resource is not a namespaced resource, then namespace
// is ignored. The ResourceClient inherits the parameter codec of c.
func (c *Client) Resource(resource *metav1.APIResource, namespace string) *ResourceClient {
	return &ResourceClient{
		cl:             c.cl,
		resource:       resource,
		ns:             namespace,
		parameterCodec: c.parameterCodec,
	}
}

// ParameterCodec returns a client with the provided parameter codec.
func (c *Client) ParameterCodec(parameterCodec runtime.ParameterCodec) *Client {
	return &Client{
		cl:             c.cl,
		parameterCodec: parameterCodec,
	}
}

// ResourceClient is an API interface to a specific resource under a
// dynamic client.
type ResourceClient struct {
	cl             *restclient.RESTClient
	resource       *metav1.APIResource
	ns             string
	parameterCodec runtime.ParameterCodec
}

// List returns a list of objects for this resource.
func (rc *ResourceClient) List(opts metav1.ListOptions) (runtime.Object, error) {
	parameterEncoder := rc.parameterCodec
	if parameterEncoder == nil {
		parameterEncoder = defaultParameterEncoder
	}
	return rc.cl.Get().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		VersionedParams(&opts, parameterEncoder).
		Do().
		Get()
}

// Get gets the resource with the specified name.
func (rc *ResourceClient) Get(name string) (*unstructured.Unstructured, error) {
	result := new(unstructured.Unstructured)
	err := rc.cl.Get().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		Name(name).
		Do().
		Into(result)
	return result, err
}

// Delete deletes the resource with the specified name.
func (rc *ResourceClient) Delete(name string, opts *metav1.DeleteOptions) error {
	return rc.cl.Delete().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		Name(name).
		Body(opts).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (rc *ResourceClient) DeleteCollection(deleteOptions *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	parameterEncoder := rc.parameterCodec
	if parameterEncoder == nil {
		parameterEncoder = defaultParameterEncoder
	}
	return rc.cl.Delete().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		VersionedParams(&listOptions, parameterEncoder).
		Body(deleteOptions).
		Do().
		Error()
}

// Create creates the provided resource.
func (rc *ResourceClient) Create(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	result := new(unstructured.Unstructured)
	err := rc.cl.Post().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		Body(obj).
		Do().
		Into(result)
	return result, err
}

// Update updates the provided resource.
func (rc *ResourceClient) Update(obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	result := new(unstructured.Unstructured)
	if len(obj.GetName()) == 0 {
		return result, errors.New("object missing name")
	}
	err := rc.cl.Put().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		Name(obj.GetName()).
		Body(obj).
		Do().
		Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the resource.
func (rc *ResourceClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	parameterEncoder := rc.parameterCodec
	if parameterEncoder == nil {
		parameterEncoder = defaultParameterEncoder
	}
	opts.Watch = true
	return rc.cl.Get().
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		VersionedParams(&opts, parameterEncoder).
		Watch()
}

func (rc *ResourceClient) Patch(name string, pt types.PatchType, data []byte) (*unstructured.Unstructured, error) {
	result := new(unstructured.Unstructured)
	err := rc.cl.Patch(pt).
		NamespaceIfScoped(rc.ns, rc.resource.Namespaced).
		Resource(rc.resource.Name).
		Name(name).
		Body(data).
		Do().
		Into(result)
	return result, err
}

// dynamicCodec is a codec that wraps the standard unstructured codec
// with special handling for Status objects.
type dynamicCodec struct{}

func (dynamicCodec) Decode(data []byte, gvk *schema.GroupVersionKind, obj runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := unstructured.UnstructuredJSONScheme.Decode(data, gvk, obj)
	if err != nil {
		return nil, nil, err
	}

	if _, ok := obj.(*metav1.Status); !ok && strings.ToLower(gvk.Kind) == "status" {
		obj = &metav1.Status{}
		err := json.Unmarshal(data, obj)
		if err != nil {
			return nil, nil, err
		}
	}

	return obj, gvk, nil
}

func (dynamicCodec) Encode(obj runtime.Object, w io.Writer) error {
	return unstructured.UnstructuredJSONScheme.Encode(obj, w)
}

// ContentConfig returns a restclient.ContentConfig for dynamic types.
func ContentConfig() restclient.ContentConfig {
	var jsonInfo runtime.SerializerInfo
	// TODO: api.Codecs here should become "pkg/apis/server/scheme" which is the minimal core you need
	// to talk to a kubernetes server
	for _, info := range api.Codecs.SupportedMediaTypes() {
		if info.MediaType == runtime.ContentTypeJSON {
			jsonInfo = info
			break
		}
	}

	jsonInfo.Serializer = dynamicCodec{}
	jsonInfo.PrettySerializer = nil
	return restclient.ContentConfig{
		AcceptContentTypes:   runtime.ContentTypeJSON,
		ContentType:          runtime.ContentTypeJSON,
		NegotiatedSerializer: serializer.NegotiatedSerializerWrapper(jsonInfo),
	}
}

// paramaterCodec is a codec converts an API object to query
// parameters without trying to convert to the target version.
type parameterCodec struct{}

func (parameterCodec) EncodeParameters(obj runtime.Object, to schema.GroupVersion) (url.Values, error) {
	return queryparams.Convert(obj)
}

func (parameterCodec) DecodeParameters(parameters url.Values, from schema.GroupVersion, into runtime.Object) error {
	return errors.New("DecodeParameters not implemented on dynamic parameterCodec")
}

var defaultParameterEncoder runtime.ParameterCodec = parameterCodec{}

type versionedParameterEncoderWithV1Fallback struct{}

func (versionedParameterEncoderWithV1Fallback) EncodeParameters(obj runtime.Object, to schema.GroupVersion) (url.Values, error) {
	ret, err := api.ParameterCodec.EncodeParameters(obj, to)
	if err != nil && runtime.IsNotRegisteredError(err) {
		// fallback to v1
		return api.ParameterCodec.EncodeParameters(obj, v1.SchemeGroupVersion)
	}
	return ret, err
}

func (versionedParameterEncoderWithV1Fallback) DecodeParameters(parameters url.Values, from schema.GroupVersion, into runtime.Object) error {
	return errors.New("DecodeParameters not implemented on versionedParameterEncoderWithV1Fallback")
}

// VersionedParameterEncoderWithV1Fallback is useful for encoding query
// parameters for thirdparty resources. It tries to convert object to the
// specified version before converting it to query parameters, and falls back to
// converting to v1 if the object is not registered in the specified version.
// For the record, currently API server always treats query parameters sent to a
// thirdparty resource endpoint as v1.
var VersionedParameterEncoderWithV1Fallback runtime.ParameterCodec = versionedParameterEncoderWithV1Fallback{}
