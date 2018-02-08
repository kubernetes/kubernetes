/*
Copyright 2017 The Kubernetes Authors.

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

package scale

import (
	"fmt"

	autoscaling "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
)

var scaleConverter = NewScaleConverter()
var codecs = serializer.NewCodecFactory(scaleConverter.Scheme())

// restInterfaceProvider turns a restclient.Config into a restclient.Interface.
// It's overridable for the purposes of testing.
type restInterfaceProvider func(*restclient.Config) (restclient.Interface, error)

// scaleClient is an implementation of ScalesGetter
// which makes use of a RESTMapper and a generic REST
// client to support an discoverable resource.
// It behaves somewhat similarly to the dynamic ClientPool,
// but is more specifically scoped to Scale.
type scaleClient struct {
	mapper meta.RESTMapper

	apiPathResolverFunc dynamic.APIPathResolverFunc
	scaleKindResolver   ScaleKindResolver
	clientBase          restclient.Interface
}

// NewForConfig creates a new ScalesGetter which resolves kinds
// to resources using the given RESTMapper, and API paths using
// the given dynamic.APIPathResolverFunc.
func NewForConfig(cfg *restclient.Config, mapper meta.RESTMapper, resolver dynamic.APIPathResolverFunc, scaleKindResolver ScaleKindResolver) (ScalesGetter, error) {
	// so that the RESTClientFor doesn't complain
	cfg.GroupVersion = &schema.GroupVersion{}

	cfg.NegotiatedSerializer = serializer.DirectCodecFactory{
		CodecFactory: codecs,
	}
	if len(cfg.UserAgent) == 0 {
		cfg.UserAgent = restclient.DefaultKubernetesUserAgent()
	}

	client, err := restclient.RESTClientFor(cfg)
	if err != nil {
		return nil, err
	}

	return New(client, mapper, resolver, scaleKindResolver), nil
}

// New creates a new ScalesGetter using the given client to make requests.
// The GroupVersion on the client is ignored.
func New(baseClient restclient.Interface, mapper meta.RESTMapper, resolver dynamic.APIPathResolverFunc, scaleKindResolver ScaleKindResolver) ScalesGetter {
	return &scaleClient{
		mapper: mapper,

		apiPathResolverFunc: resolver,
		scaleKindResolver:   scaleKindResolver,
		clientBase:          baseClient,
	}
}

// pathAndVersionFor returns the appropriate base path and the associated full GroupVersionResource
// for the given GroupResource
func (c *scaleClient) pathAndVersionFor(resource schema.GroupResource) (string, schema.GroupVersionResource, error) {
	gvr, err := c.mapper.ResourceFor(resource.WithVersion(""))
	if err != nil {
		return "", gvr, fmt.Errorf("unable to get full preferred group-version-resource for %s: %v", resource.String(), err)
	}

	groupVer := gvr.GroupVersion()

	// we need to set the API path based on GroupVersion (defaulting to the legacy path if none is set)
	// TODO: we "cheat" here since the API path really only depends on group ATM, but this should
	// *probably* take GroupVersionResource and not GroupVersionKind.
	apiPath := c.apiPathResolverFunc(groupVer.WithKind(""))
	if apiPath == "" {
		apiPath = "/api"
	}

	path := restclient.DefaultVersionedAPIPath(apiPath, groupVer)

	return path, gvr, nil
}

// namespacedScaleClient is an ScaleInterface for fetching
// Scales in a given namespace.
type namespacedScaleClient struct {
	client    *scaleClient
	namespace string
}

func (c *scaleClient) Scales(namespace string) ScaleInterface {
	return &namespacedScaleClient{
		client:    c,
		namespace: namespace,
	}
}

func (c *namespacedScaleClient) Get(resource schema.GroupResource, name string) (*autoscaling.Scale, error) {
	// Currently, a /scale endpoint can return different scale types.
	// Until we have support for the alternative API representations proposal,
	// we need to deal with accepting different API versions.
	// In practice, this is autoscaling/v1.Scale and extensions/v1beta1.Scale

	path, gvr, err := c.client.pathAndVersionFor(resource)
	if err != nil {
		return nil, fmt.Errorf("unable to get client for %s: %v", resource.String(), err)
	}

	result := c.client.clientBase.Get().
		AbsPath(path).
		Namespace(c.namespace).
		Resource(gvr.Resource).
		Name(name).
		SubResource("scale").
		Do()
	if err := result.Error(); err != nil {
		return nil, fmt.Errorf("could not fetch the scale for %s %s: %v", resource.String(), name, err)
	}

	scaleBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	decoder := scaleConverter.codecs.UniversalDecoder(scaleConverter.ScaleVersions()...)
	rawScaleObj, err := runtime.Decode(decoder, scaleBytes)
	if err != nil {
		return nil, err
	}

	// convert whatever this is to autoscaling/v1.Scale
	scaleObj, err := scaleConverter.ConvertToVersion(rawScaleObj, autoscaling.SchemeGroupVersion)
	if err != nil {
		return nil, fmt.Errorf("received an object from a /scale endpoint which was not convertible to autoscaling Scale: %v", err)
	}

	return scaleObj.(*autoscaling.Scale), nil
}

func (c *namespacedScaleClient) Update(resource schema.GroupResource, scale *autoscaling.Scale) (*autoscaling.Scale, error) {
	path, gvr, err := c.client.pathAndVersionFor(resource)
	if err != nil {
		return nil, fmt.Errorf("unable to get client for %s: %v", resource.String(), err)
	}

	// Currently, a /scale endpoint can receive and return different scale types.
	// Until we have support for the alternative API representations proposal,
	// we need to deal with sending and accepting different API versions.

	// figure out what scale we actually need here
	desiredGVK, err := c.client.scaleKindResolver.ScaleForResource(gvr)
	if err != nil {
		return nil, fmt.Errorf("could not find proper group-version for scale subresource of %s: %v", gvr.String(), err)
	}

	// convert this to whatever this endpoint wants
	scaleUpdate, err := scaleConverter.ConvertToVersion(scale, desiredGVK.GroupVersion())
	if err != nil {
		return nil, fmt.Errorf("could not convert scale update to external Scale: %v", err)
	}
	encoder := scaleConverter.codecs.LegacyCodec(desiredGVK.GroupVersion())
	scaleUpdateBytes, err := runtime.Encode(encoder, scaleUpdate)
	if err != nil {
		return nil, fmt.Errorf("could not encode scale update to external Scale: %v", err)
	}

	result := c.client.clientBase.Put().
		AbsPath(path).
		Namespace(c.namespace).
		Resource(gvr.Resource).
		Name(scale.Name).
		SubResource("scale").
		Body(scaleUpdateBytes).
		Do()
	if err := result.Error(); err != nil {
		return nil, fmt.Errorf("could not update the scale for %s %s: %v", resource.String(), scale.Name, err)
	}

	scaleBytes, err := result.Raw()
	if err != nil {
		return nil, err
	}
	decoder := scaleConverter.codecs.UniversalDecoder(scaleConverter.ScaleVersions()...)
	rawScaleObj, err := runtime.Decode(decoder, scaleBytes)
	if err != nil {
		return nil, err
	}

	// convert whatever this is back to autoscaling/v1.Scale
	scaleObj, err := scaleConverter.ConvertToVersion(rawScaleObj, autoscaling.SchemeGroupVersion)
	if err != nil {
		return nil, fmt.Errorf("received an object from a /scale endpoint which was not convertible to autoscaling Scale: %v", err)
	}

	return scaleObj.(*autoscaling.Scale), err
}
