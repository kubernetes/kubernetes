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
	"sync"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/api/meta"
	scaleapi "k8s.io/client-go/pkg/apis/autoscaling/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/pkg/api"
)

// scaleClient is similar to dynamic.ClientPool, except that it knows a bit
// more about what it's doing, so it can be more specific.
type scaleClient struct {
	config *restclient.Config
	mapper meta.RESTMapper
	clients map[schema.GroupVersion]*restclient.RESTClient
	apiPathResolverFunc dynamic.APIPathResolverFunc
	sync.Mutex
}

func NewScaleClient(cfg *restclient.Config) ScalesGetter {
	restMapper := api.Registry.RESTMapper()
	return &scaleClient{
		config: cfg,
		mapper: restMapper,
		clients: make(map[schema.GroupVersion]*restclient.RESTClient),
		apiPathResolverFunc: dynamic.LegacyAPIPathResolverFunc,
	}
}

// restClient returns a RESTClient, as well as the resource name, for the given GroupKind.
func (c *scaleClient) restClient(kind schema.GroupKind) (*restclient.RESTClient, string, error) {
	c.Lock()
	defer c.Unlock()

	mapping, err := c.mapper.RESTMapping(kind)
	if err != nil {
		return nil, "", fmt.Errorf("unable to get REST mapping for %s: %v", kind.String(), err)
	}

	groupVersion := mapping.GroupVersionKind.GroupVersion()

	// check if we have an existing client
	if existingClient, found := c.clients[groupVersion]; found {
		return existingClient, mapping.Resource, nil
	}

	// avoid changing the original config
	confCopy := *c.config
	conf := &confCopy

	// we need to set the api path based on group version, if no group, default to the legacy path
	conf.APIPath = c.apiPathResolverFunc(mapping.GroupVersionKind)

	conf.GroupVersion = &groupVersion

	contentConfig := dynamic.ContentConfig()
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
		return nil, "", err
	}

	c.clients[groupVersion] = cl

	return cl, mapping.Resource, nil
}

type namespacedScaleClient struct {
	client *scaleClient
	namespace string
}

func (c *scaleClient) Scales(namespace string) ScaleInterface {
	return &namespacedScaleClient{
		client: c,
		namespace: namespace,
	}
}

func (c *namespacedScaleClient) Get(kind schema.GroupKind, name string) (*scaleapi.Scale, error) {
	// TODO: do we need the version here too?  Should we auto-determine the version somehow?
	client, resource, err := c.client.restClient(kind)
	if err != nil {
		return nil, fmt.Errorf("unable to get client for %s: %v", kind.String(), err)
	}

	result := new(scaleapi.Scale)
	err = client.Get().
		Namespace(c.namespace).
		Resource(resource).
		Name(name).
		SubResource("scale").
		Do().
		Into(result)
	return result, err
}

func (c *namespacedScaleClient) Update(kind schema.GroupKind, scale *scaleapi.Scale) (*scaleapi.Scale, error) {
	// TODO: do we need the version here too?  Should we auto-determine the version somehow?
	client, resource, err := c.client.restClient(kind)
	if err != nil {
		return nil, fmt.Errorf("unable to get client for %s: %v", kind.String(), err)
	}

	result := new(scaleapi.Scale)
	err = client.Put().
		Namespace(c.namespace).
		Resource(resource).
		Name(scale.Name).
		SubResource("scale").
		Body(scale).
		Do().
		Into(result)
	return result, err
}
