/*
Copyright 2022 The KCP Authors.

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

package client

import (
	"net/http"
	"sync"

	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/client-go/rest"
)

// Constructor is a wrapper around a constructor method for the client of type R.
type Constructor[R any] struct {
	NewForConfigAndClient func(*rest.Config, *http.Client) (R, error)
}

// Cache is a client factory that caches previous results.
type Cache[R any] interface {
	ClusterOrDie(clusterPath logicalcluster.Path) R
	Cluster(clusterPath logicalcluster.Path) (R, error)
}

// NewCache creates a new client factory cache using the given constructor.
func NewCache[R any](cfg *rest.Config, client *http.Client, constructor *Constructor[R]) Cache[R] {
	return &clientCache[R]{
		cfg:         cfg,
		client:      client,
		constructor: constructor,

		RWMutex:              &sync.RWMutex{},
		clientsByClusterPath: map[logicalcluster.Path]R{},
	}
}

type clientCache[R any] struct {
	cfg         *rest.Config
	client      *http.Client
	constructor *Constructor[R]

	*sync.RWMutex
	clientsByClusterPath map[logicalcluster.Path]R
}

// ClusterOrDie returns a new client scoped to the given logical cluster, or panics if there
// is any error.
func (c *clientCache[R]) ClusterOrDie(clusterPath logicalcluster.Path) R {
	client, err := c.Cluster(clusterPath)
	if err != nil {
		// we ensure that the config is valid in the constructor, and we assume that any changes
		// we make to it during scoping will not make it invalid, in order to hide the error from
		// downstream callers (as it should forever be nil); this is slightly risky
		panic(err)
	}
	return client
}

// Cluster returns a new client scoped to the given logical cluster.
func (c *clientCache[R]) Cluster(clusterPath logicalcluster.Path) (R, error) {
	var cachedClient R
	var exists bool
	c.RLock()
	cachedClient, exists = c.clientsByClusterPath[clusterPath]
	c.RUnlock()
	if exists {
		return cachedClient, nil
	}

	cfg := SetCluster(rest.CopyConfig(c.cfg), clusterPath)
	instance, err := c.constructor.NewForConfigAndClient(cfg, c.client)
	if err != nil {
		var result R
		return result, err
	}

	c.Lock()
	defer c.Unlock()
	cachedClient, exists = c.clientsByClusterPath[clusterPath]
	if exists {
		return cachedClient, nil
	}

	c.clientsByClusterPath[clusterPath] = instance

	return instance, nil
}
