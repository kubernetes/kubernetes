/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/conversion/queryparams"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/watch"
)

// Client is a Kubernetes client that allows you to access metadata
// and manipulate metadata of a Kubernetes API group.
type Client struct {
	cl *restclient.RESTClient
}

// NewClient returns a new client based on the passed in config. The
// codec is ignored, as the dynamic client uses it's own codec.
func NewClient(conf *restclient.Config) (*Client, error) {
	// avoid changing the original config
	confCopy := *conf
	conf = &confCopy

	codec := dynamicCodec{}

	// TODO: it's questionable that this should be using anything other than unstructured schema and JSON
	streamingInfo, _ := api.Codecs.StreamingSerializerForMediaType("application/json;stream=watch", nil)
	conf.NegotiatedSerializer = serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{Serializer: codec}, streamingInfo)

	if conf.APIPath == "" {
		conf.APIPath = "/api"
	}

	if len(conf.UserAgent) == 0 {
		conf.UserAgent = restclient.DefaultKubernetesUserAgent()
	}

	if conf.QPS == 0.0 {
		conf.QPS = 5.0
	}
	if conf.Burst == 0 {
		conf.Burst = 10
	}

	cl, err := restclient.RESTClientFor(conf)
	if err != nil {
		return nil, err
	}

	return &Client{cl: cl}, nil
}

// Resource returns an API interface to the specified resource for
// this client's group and version. If resource is not a namespaced
// resource, then namespace is ignored.
func (c *Client) Resource(resource *unversioned.APIResource, namespace string) *ResourceClient {
	return &ResourceClient{
		cl:       c.cl,
		resource: resource,
		ns:       namespace,
	}
}

// ResourceClient is an API interface to a specific resource under a
// dynamic client.
type ResourceClient struct {
	cl       *restclient.RESTClient
	resource *unversioned.APIResource
	ns       string
}

// namespace applies a namespace to the request if the configured
// resource is a namespaced resource. Otherwise, it just returns the
// passed in request.
func (rc *ResourceClient) namespace(req *restclient.Request) *restclient.Request {
	if rc.resource.Namespaced {
		return req.Namespace(rc.ns)
	}
	return req
}

// List returns a list of objects for this resource.
func (rc *ResourceClient) List(opts v1.ListOptions) (*runtime.UnstructuredList, error) {
	result := new(runtime.UnstructuredList)
	err := rc.namespace(rc.cl.Get()).
		Resource(rc.resource.Name).
		VersionedParams(&opts, parameterEncoder).
		Do().
		Into(result)
	return result, err
}

// Get gets the resource with the specified name.
func (rc *ResourceClient) Get(name string) (*runtime.Unstructured, error) {
	result := new(runtime.Unstructured)
	err := rc.namespace(rc.cl.Get()).
		Resource(rc.resource.Name).
		Name(name).
		Do().
		Into(result)
	return result, err
}

// Delete deletes the resource with the specified name.
func (rc *ResourceClient) Delete(name string, opts *v1.DeleteOptions) error {
	return rc.namespace(rc.cl.Delete()).
		Resource(rc.resource.Name).
		Name(name).
		Body(opts).
		Do().
		Error()
}

// DeleteCollection deletes a collection of objects.
func (rc *ResourceClient) DeleteCollection(deleteOptions *v1.DeleteOptions, listOptions v1.ListOptions) error {
	return rc.namespace(rc.cl.Delete()).
		Resource(rc.resource.Name).
		VersionedParams(&listOptions, parameterEncoder).
		Body(deleteOptions).
		Do().
		Error()
}

// Create creates the provided resource.
func (rc *ResourceClient) Create(obj *runtime.Unstructured) (*runtime.Unstructured, error) {
	result := new(runtime.Unstructured)
	err := rc.namespace(rc.cl.Post()).
		Resource(rc.resource.Name).
		Body(obj).
		Do().
		Into(result)
	return result, err
}

// Update updates the provided resource.
func (rc *ResourceClient) Update(obj *runtime.Unstructured) (*runtime.Unstructured, error) {
	result := new(runtime.Unstructured)
	if len(obj.GetName()) == 0 {
		return result, errors.New("object missing name")
	}
	err := rc.namespace(rc.cl.Put()).
		Resource(rc.resource.Name).
		Name(obj.GetName()).
		Body(obj).
		Do().
		Into(result)
	return result, err
}

// Watch returns a watch.Interface that watches the resource.
func (rc *ResourceClient) Watch(opts v1.ListOptions) (watch.Interface, error) {
	return rc.namespace(rc.cl.Get().Prefix("watch")).
		Resource(rc.resource.Name).
		VersionedParams(&opts, parameterEncoder).
		Watch()
}

// dynamicCodec is a codec that wraps the standard unstructured codec
// with special handling for Status objects.
type dynamicCodec struct{}

func (dynamicCodec) Decode(data []byte, gvk *unversioned.GroupVersionKind, obj runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	obj, gvk, err := runtime.UnstructuredJSONScheme.Decode(data, gvk, obj)
	if err != nil {
		return nil, nil, err
	}

	if _, ok := obj.(*unversioned.Status); !ok && strings.ToLower(gvk.Kind) == "status" {
		obj = &unversioned.Status{}
		err := json.Unmarshal(data, obj)
		if err != nil {
			return nil, nil, err
		}
	}

	return obj, gvk, nil
}

func (dynamicCodec) EncodeToStream(obj runtime.Object, w io.Writer, overrides ...unversioned.GroupVersion) error {
	return runtime.UnstructuredJSONScheme.EncodeToStream(obj, w, overrides...)
}

// paramaterCodec is a codec converts an API object to query
// parameters without trying to convert to the target version.
type parameterCodec struct{}

func (parameterCodec) EncodeParameters(obj runtime.Object, to unversioned.GroupVersion) (url.Values, error) {
	return queryparams.Convert(obj)
}

func (parameterCodec) DecodeParameters(parameters url.Values, from unversioned.GroupVersion, into runtime.Object) error {
	return errors.New("DecodeParameters not implemented on dynamic parameterCodec")
}

var parameterEncoder runtime.ParameterCodec = parameterCodec{}
