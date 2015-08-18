/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"k8s.io/kubernetes/pkg/api/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

func NewClientCache(loader clientcmd.ClientConfig) *ClientCache {
	return &ClientCache{
		clients: make(map[string]*client.Client),
		configs: make(map[string]*client.Config),
		loader:  loader,
	}
}

// ClientCache caches previously loaded clients for reuse, and ensures MatchServerVersion
// is invoked only once
type ClientCache struct {
	loader        clientcmd.ClientConfig
	clients       map[string]*client.Client
	configs       map[string]*client.Config
	defaultConfig *client.Config
	defaultClient *client.Client
	matchVersion  bool
}

// ClientConfigForVersion returns the correct config for a server
func (c *ClientCache) ClientConfigForVersion(version string) (*client.Config, error) {
	if c.defaultConfig == nil {
		config, err := c.loader.ClientConfig()
		if err != nil {
			return nil, err
		}
		c.defaultConfig = config
		if c.matchVersion {
			if err := client.MatchesServerVersion(c.defaultClient, config); err != nil {
				return nil, err
			}
		}
	}
	if config, ok := c.configs[version]; ok {
		return config, nil
	}
	// TODO: have a better config copy method
	config := *c.defaultConfig
	negotiatedVersion, err := client.NegotiateVersion(c.defaultClient, &config, version, registered.RegisteredVersions)
	if err != nil {
		return nil, err
	}
	config.Version = negotiatedVersion
	client.SetKubernetesDefaults(&config)
	c.configs[version] = &config

	return &config, nil
}

// ClientForVersion initializes or reuses a client for the specified version, or returns an
// error if that is not possible
func (c *ClientCache) ClientForVersion(version string) (*client.Client, error) {
	if client, ok := c.clients[version]; ok {
		return client, nil
	}
	config, err := c.ClientConfigForVersion(version)
	if err != nil {
		return nil, err
	}
	client, err := client.New(config)
	if err != nil {
		return nil, err
	}

	c.clients[config.Version] = client
	return client, nil
}
