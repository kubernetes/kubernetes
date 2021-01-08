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

package resource

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
)

// clientPool allows RESTClient's to be reused per schema.GroupVersion.
type clientPool struct {
	mu         sync.Mutex
	clientFunc func(version schema.GroupVersion) (RESTClient, error)
	cache      map[schema.GroupVersion]RESTClient
}

func newClientPool(clientFunc func(version schema.GroupVersion) (RESTClient, error)) *clientPool {
	return &clientPool{
		clientFunc: clientFunc,
		cache:      make(map[schema.GroupVersion]RESTClient),
	}
}

func (c *clientPool) get(version schema.GroupVersion) (RESTClient, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	client, ok := c.cache[version]
	if !ok {
		var err error
		client, err = c.clientFunc(version)
		if err != nil {
			return nil, err
		}
		c.cache[version] = client
	}

	return client, nil
}

// TODO require negotiatedSerializer.  leaving it optional lets us plumb current behavior and deal with the difference after major plumbing is complete
func (clientConfigFn ClientConfigFunc) clientForGroupVersion(gv schema.GroupVersion, negotiatedSerializer runtime.NegotiatedSerializer) (RESTClient, error) {
	cfg, err := clientConfigFn()
	if err != nil {
		return nil, err
	}
	if negotiatedSerializer != nil {
		cfg.ContentConfig.NegotiatedSerializer = negotiatedSerializer
	}
	cfg.GroupVersion = &gv
	if len(gv.Group) == 0 {
		cfg.APIPath = "/api"
	} else {
		cfg.APIPath = "/apis"
	}

	return rest.RESTClientFor(cfg)
}

func (clientConfigFn ClientConfigFunc) unstructuredClientForGroupVersion(gv schema.GroupVersion) (RESTClient, error) {
	cfg, err := clientConfigFn()
	if err != nil {
		return nil, err
	}
	cfg.ContentConfig = UnstructuredPlusDefaultContentConfig()
	cfg.GroupVersion = &gv
	if len(gv.Group) == 0 {
		cfg.APIPath = "/api"
	} else {
		cfg.APIPath = "/apis"
	}

	return rest.RESTClientFor(cfg)
}
