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

package metadata

import (
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/klog"

	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/rest"
)

var deleteScheme = runtime.NewScheme()
var parameterScheme = runtime.NewScheme()
var deleteOptionsCodec = serializer.NewCodecFactory(deleteScheme)
var dynamicParameterCodec = runtime.NewParameterCodec(parameterScheme)

var versionV1 = schema.GroupVersion{Version: "v1"}

func init() {
	metav1.AddToGroupVersion(parameterScheme, versionV1)
	metav1.AddToGroupVersion(deleteScheme, versionV1)
}

// Client allows callers to retrieve the object metadata for any
// Kubernetes-compatible API endpoint. The client uses the
// meta.k8s.io/v1 PartialObjectMetadata resource to more efficiently
// retrieve just the necessary metadata, but on older servers
// (Kubernetes 1.14 and before) will retrieve the object and then
// convert the metadata.
type Client struct {
	client *rest.RESTClient
}

var _ Interface = &Client{}

// ConfigFor returns a copy of the provided config with the
// appropriate metadata client defaults set.
func ConfigFor(inConfig *rest.Config) *rest.Config {
	config := rest.CopyConfig(inConfig)
	config.AcceptContentTypes = "application/vnd.kubernetes.protobuf,application/json"
	config.ContentType = "application/vnd.kubernetes.protobuf"
	config.NegotiatedSerializer = metainternalversionscheme.Codecs.WithoutConversion()
	if config.UserAgent == "" {
		config.UserAgent = rest.DefaultKubernetesUserAgent()
	}
	return config
}

// NewForConfigOrDie creates a new metadata client for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) Interface {
	ret, err := NewForConfig(c)
	if err != nil {
		panic(err)
	}
	return ret
}

// NewForConfig creates a new metadata client that can retrieve object
// metadata details about any Kubernetes object (core, aggregated, or custom
// resource based) in the form of PartialObjectMetadata objects, or returns
// an error.
func NewForConfig(inConfig *rest.Config) (Interface, error) {
	config := ConfigFor(inConfig)
	// for serializing the options
	config.GroupVersion = &schema.GroupVersion{}
	config.APIPath = "/this-value-should-never-be-sent"

	restClient, err := rest.RESTClientFor(config)
	if err != nil {
		return nil, err
	}

	return &Client{client: restClient}, nil
}

type client struct {
	client    *Client
	namespace string
	resource  schema.GroupVersionResource
}

// Resource returns an interface that can access cluster or namespace
// scoped instances of resource.
func (c *Client) Resource(resource schema.GroupVersionResource) Getter {
	return &client{client: c, resource: resource}
}

// Namespace returns an interface that can access namespace-scoped instances of the
// provided resource.
func (c *client) Namespace(ns string) ResourceInterface {
	ret := *c
	ret.namespace = ns
	return &ret
}

// Delete removes the provided resource from the server.
func (c *client) Delete(name string, opts *metav1.DeleteOptions, subresources ...string) error {
	if len(name) == 0 {
		return fmt.Errorf("name is required")
	}
	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	deleteOptionsByte, err := runtime.Encode(deleteOptionsCodec.LegacyCodec(schema.GroupVersion{Version: "v1"}), opts)
	if err != nil {
		return err
	}

	result := c.client.client.
		Delete().
		AbsPath(append(c.makeURLSegments(name), subresources...)...).
		Body(deleteOptionsByte).
		Do()
	return result.Error()
}

// DeleteCollection triggers deletion of all resources in the specified scope (namespace or cluster).
func (c *client) DeleteCollection(opts *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	if opts == nil {
		opts = &metav1.DeleteOptions{}
	}
	deleteOptionsByte, err := runtime.Encode(deleteOptionsCodec.LegacyCodec(schema.GroupVersion{Version: "v1"}), opts)
	if err != nil {
		return err
	}

	result := c.client.client.
		Delete().
		AbsPath(c.makeURLSegments("")...).
		Body(deleteOptionsByte).
		SpecificallyVersionedParams(&listOptions, dynamicParameterCodec, versionV1).
		Do()
	return result.Error()
}

// Get returns the resource with name from the specified scope (namespace or cluster).
func (c *client) Get(name string, opts metav1.GetOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
	if len(name) == 0 {
		return nil, fmt.Errorf("name is required")
	}
	result := c.client.client.Get().AbsPath(append(c.makeURLSegments(name), subresources...)...).
		SetHeader("Accept", "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json").
		SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).
		Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	obj, err := result.Get()
	if runtime.IsNotRegisteredError(err) {
		klog.V(5).Infof("Unable to retrieve PartialObjectMetadata: %#v", err)
		rawBytes, err := result.Raw()
		if err != nil {
			return nil, err
		}
		var partial metav1.PartialObjectMetadata
		if err := json.Unmarshal(rawBytes, &partial); err != nil {
			return nil, fmt.Errorf("unable to decode returned object as PartialObjectMetadata: %v", err)
		}
		if !isLikelyObjectMetadata(&partial) {
			return nil, fmt.Errorf("object does not appear to match the ObjectMeta schema: %#v", partial)
		}
		partial.TypeMeta = metav1.TypeMeta{}
		return &partial, nil
	}
	if err != nil {
		return nil, err
	}
	partial, ok := obj.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected object, expected PartialObjectMetadata but got %T", obj)
	}
	return partial, nil
}

// List returns all resources within the specified scope (namespace or cluster).
func (c *client) List(opts metav1.ListOptions) (*metav1.PartialObjectMetadataList, error) {
	result := c.client.client.Get().AbsPath(c.makeURLSegments("")...).
		SetHeader("Accept", "application/vnd.kubernetes.protobuf;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1,application/json").
		SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).
		Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	obj, err := result.Get()
	if runtime.IsNotRegisteredError(err) {
		klog.V(5).Infof("Unable to retrieve PartialObjectMetadataList: %#v", err)
		rawBytes, err := result.Raw()
		if err != nil {
			return nil, err
		}
		var partial metav1.PartialObjectMetadataList
		if err := json.Unmarshal(rawBytes, &partial); err != nil {
			return nil, fmt.Errorf("unable to decode returned object as PartialObjectMetadataList: %v", err)
		}
		partial.TypeMeta = metav1.TypeMeta{}
		return &partial, nil
	}
	if err != nil {
		return nil, err
	}
	partial, ok := obj.(*metav1.PartialObjectMetadataList)
	if !ok {
		return nil, fmt.Errorf("unexpected object, expected PartialObjectMetadata but got %T", obj)
	}
	return partial, nil
}

// Watch finds all changes to the resources in the specified scope (namespace or cluster).
func (c *client) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.client.Get().
		AbsPath(c.makeURLSegments("")...).
		SetHeader("Accept", "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json").
		SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).
		Timeout(timeout).
		Watch()
}

// Patch modifies the named resource in the specified scope (namespace or cluster).
func (c *client) Patch(name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (*metav1.PartialObjectMetadata, error) {
	if len(name) == 0 {
		return nil, fmt.Errorf("name is required")
	}
	result := c.client.client.
		Patch(pt).
		AbsPath(append(c.makeURLSegments(name), subresources...)...).
		Body(data).
		SetHeader("Accept", "application/vnd.kubernetes.protobuf;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1,application/json").
		SpecificallyVersionedParams(&opts, dynamicParameterCodec, versionV1).
		Do()
	if err := result.Error(); err != nil {
		return nil, err
	}
	obj, err := result.Get()
	if runtime.IsNotRegisteredError(err) {
		rawBytes, err := result.Raw()
		if err != nil {
			return nil, err
		}
		var partial metav1.PartialObjectMetadata
		if err := json.Unmarshal(rawBytes, &partial); err != nil {
			return nil, fmt.Errorf("unable to decode returned object as PartialObjectMetadata: %v", err)
		}
		if !isLikelyObjectMetadata(&partial) {
			return nil, fmt.Errorf("object does not appear to match the ObjectMeta schema")
		}
		partial.TypeMeta = metav1.TypeMeta{}
		return &partial, nil
	}
	if err != nil {
		return nil, err
	}
	partial, ok := obj.(*metav1.PartialObjectMetadata)
	if !ok {
		return nil, fmt.Errorf("unexpected object, expected PartialObjectMetadata but got %T", obj)
	}
	return partial, nil
}

func (c *client) makeURLSegments(name string) []string {
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
	}

	return url
}

func isLikelyObjectMetadata(meta *metav1.PartialObjectMetadata) bool {
	return len(meta.UID) > 0 || !meta.CreationTimestamp.IsZero() || len(meta.Name) > 0 || len(meta.GenerateName) > 0
}
