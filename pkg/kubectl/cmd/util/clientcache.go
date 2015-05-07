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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
)

// clientCache caches previously loaded clients for reuse, and ensures MatchServerVersion
// is invoked only once
type clientCache struct {
	loader        clientcmd.ClientConfig
	clients       map[string]*client.Client
	defaultConfig *client.Config
	matchVersion  bool
}

// ClientConfigForVersion returns the correct config for a server
func (c *clientCache) ClientConfigForVersion(version string) (*client.Config, error) {
	if c.defaultConfig == nil {
		config, err := c.loader.ClientConfig()
		if err != nil {
			return nil, err
		}
		c.defaultConfig = config
		if c.matchVersion {
			if err := client.MatchesServerVersion(config); err != nil {
				return nil, err
			}
		}
	}
	// TODO: have a better config copy method
	config := *c.defaultConfig
	if len(version) != 0 {
		config.Version = version
	}
	client.SetKubernetesDefaults(&config)

	return &config, nil
}

// ClientForVersion initializes or reuses a client for the specified version, or returns an
// error if that is not possible
func (c *clientCache) ClientForVersion(version string) (*client.Client, error) {
	config, err := c.ClientConfigForVersion(version)
	if err != nil {
		return nil, err
	}

	if client, ok := c.clients[config.Version]; ok {
		return client, nil
	}

	client, err := client.New(config)
	if err != nil {
		return nil, err
	}

	c.clients[config.Version] = client
	return client, nil
}
