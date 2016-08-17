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

package dynamic

import (
	"sync"

	"k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
	"k8s.io/client-go/1.4/pkg/runtime"
	"k8s.io/client-go/1.4/pkg/runtime/serializer"
	"k8s.io/client-go/1.4/rest"
)

// ClientPool manages a pool of dynamic clients.
type ClientPool interface {
	// ClientForGroupVersion returns a client configured for the specified groupVersion.
	ClientForGroupVersion(groupVersion unversioned.GroupVersion) (*Client, error)
}

// APIPathResolverFunc knows how to convert a groupVersion to its API path.
type APIPathResolverFunc func(groupVersion unversioned.GroupVersion) string

// LegacyAPIPathResolverFunc can resolve paths properly with the legacy API.
func LegacyAPIPathResolverFunc(groupVersion unversioned.GroupVersion) string {
	if len(groupVersion.Group) == 0 {
		return "/api"
	}
	return "/apis"
}

// clientPoolImpl implements Factory
type clientPoolImpl struct {
	lock                sync.RWMutex
	config              *rest.Config
	clients             map[unversioned.GroupVersion]*Client
	apiPathResolverFunc APIPathResolverFunc
}

// NewClientPool returns a ClientPool from the specified config
func NewClientPool(config *rest.Config, apiPathResolverFunc APIPathResolverFunc) ClientPool {
	confCopy := *config
	return &clientPoolImpl{
		config:              &confCopy,
		clients:             map[unversioned.GroupVersion]*Client{},
		apiPathResolverFunc: apiPathResolverFunc,
	}
}

// ClientForGroupVersion returns a client for the specified groupVersion, creates one if none exists
func (c *clientPoolImpl) ClientForGroupVersion(groupVersion unversioned.GroupVersion) (*Client, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// do we have a client already configured?
	if existingClient, found := c.clients[groupVersion]; found {
		return existingClient, nil
	}

	// avoid changing the original config
	confCopy := *c.config
	conf := &confCopy

	// we need to set the api path based on group version, if no group, default to legacy path
	conf.APIPath = c.apiPathResolverFunc(groupVersion)

	// we need to make a client
	conf.GroupVersion = &groupVersion

	if conf.NegotiatedSerializer == nil {
		streamingInfo, _ := api.Codecs.StreamingSerializerForMediaType("application/json;stream=watch", nil)
		conf.NegotiatedSerializer = serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{Serializer: dynamicCodec{}}, streamingInfo)
	}

	dynamicClient, err := NewClient(conf)
	if err != nil {
		return nil, err
	}
	c.clients[groupVersion] = dynamicClient
	return dynamicClient, nil
}
