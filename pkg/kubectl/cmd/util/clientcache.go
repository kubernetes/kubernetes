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
	apiutil "k8s.io/kubernetes/pkg/api/util"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

func NewClientCache(loader clientcmd.ClientConfig) *ClientCache {
	return &ClientCache{
		clients:       make(map[string]*client.RESTClient),
		configs:       make(map[string]*client.Config),
		defaultConfig: make(map[string]*client.Config),
		loader:        loader,
	}
}

// ClientCache caches previously loaded clients for reuse, and ensures MatchServerVersion
// is invoked only once
type ClientCache struct {
	loader clientcmd.ClientConfig
	// clients is a map between group/version and clients. Note that version is
	// allowed to be empty.
	clients map[string]*client.RESTClient
	// configs is a map between group/version and configs. Note that version is
	// allowed to be empty.
	configs map[string]*client.Config
	// defaultConfig is a map between group and their default config.
	defaultConfig map[string]*client.Config
	defaultClient *client.Client
	matchVersion  bool
}

// ClientConfigForVersion returns the correct config for a server. Version is
// allowed to be empty.
func (c *ClientCache) ClientConfigForVersion(group, version string) (*client.Config, error) {
	if _, found := c.defaultConfig[group]; !found {
		config, err := c.loader.ClientConfig(group)
		if err != nil {
			return nil, err
		}
		c.defaultConfig[group] = config
		if c.matchVersion {
			if err := client.MatchesServerVersion(c.defaultClient, config); err != nil {
				return nil, err
			}
		}
	}
	if config, ok := c.configs[apiutil.GetGroupVersion(group, version)]; ok {
		return config, nil
	}
	// TODO: have a better config copy method
	config := *c.defaultConfig[group]
	negotiatedGroupVersion, err := client.NegotiateVersion(&config, group, version, registered.RegisteredVersions)
	if err != nil {
		return nil, err
	}
	config.GroupVersion = negotiatedGroupVersion
	if err := client.SetKubernetesDefaultsForGroup(group, &config); err != nil {
		return nil, err
	}
	c.configs[config.GroupVersion] = &config

	return &config, nil
}

// RESTClientForVersion initializes or reuses a RESTClient for the specified version, or returns an
// error if that is not possible
func (c *ClientCache) RESTClientForVersion(groupVersion string) (*client.RESTClient, error) {
	if client, ok := c.clients[groupVersion]; ok {
		return client, nil
	}
	config, err := c.ClientConfigForVersion(apiutil.GetGroup(groupVersion), apiutil.GetVersion(groupVersion))
	if err != nil {
		return nil, err
	}
	client, err := client.RESTClientFor(config)
	if err != nil {
		return nil, err
	}
	// TODO: Version should be renamed to GroupVersion
	c.clients[config.GroupVersion] = client
	return client, nil
}

// ClientForVersion returns a Client with RESTClients for all groups.
// Currently this function only accepts one groupVersion. It will
// try to initialize or reuse the RESTClient of the group in the groupVersion
// for the specified version. For RESTClients of other groups, it will initialize
// or reuse the one of the negotiated version. An error is returned if it is not
// possible to return such a Client.
// TODO: We will deprecate this function. We will add a bunch of functions that
// return typed RESTClient, like ExperimentalClient. Users of ClientForVersion
// should either convert to use a versioned client.Client, e.g., pkg/client/v1/,
// or use the functions that return typed RESTClient.
func (c *ClientCache) ClientForVersion(groupVersion string) (*client.Client, error) {
	// TODO: The following could be a for loop when we have a MetaClient
	// initialize the RESTClient
	var legacyVersion string
	if apiutil.GetGroup(groupVersion) == "" {
		legacyVersion = apiutil.GetVersion(groupVersion)
	} else {
		// empty version means no preference.
		legacyVersion = ""
	}
	legacyClient, ok := c.clients[apiutil.GetGroupVersion("", legacyVersion)]
	if !ok {
		config, err := c.ClientConfigForVersion("", legacyVersion)
		if err != nil {
			return nil, err
		}
		legacyClient, err = client.RESTClientFor(config)
		if err != nil {
			return nil, err
		}
		// TODO: Version should be renamed to GroupVersion
		c.clients[config.GroupVersion] = legacyClient
	}

	// initialize the ExperimentalClient
	var experimentalVersion string
	if apiutil.GetGroup(groupVersion) == "experimental" {
		experimentalVersion = apiutil.GetVersion(groupVersion)
	} else {
		// empty version means no preference.
		experimentalVersion = ""
	}
	experimentalClient, ok := c.clients[apiutil.GetGroupVersion("experimental", experimentalVersion)]
	if !ok {
		config, err := c.ClientConfigForVersion("experimental", experimentalVersion)
		if err != nil && len(experimentalVersion) == 0 {
			// This is a hack. This is to work around when "experimental" is not supported by the server,
			// and the caller doesn't specifically want a ExperiementalClient. But this also hides other
			// types of error.
			return &client.Client{RESTClient: legacyClient, ExperimentalClient: nil}, nil
		}
		if err != nil {
			return nil, err
		}
		experimentalClient, err = client.RESTClientFor(config)
		if err != nil {
			return nil, err
		}
		c.clients[config.GroupVersion] = experimentalClient
	}
	return &client.Client{RESTClient: legacyClient, ExperimentalClient: &client.ExperimentalClient{experimentalClient}}, nil
}
