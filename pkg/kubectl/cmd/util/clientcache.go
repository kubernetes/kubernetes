/*
Copyright 2014 The Kubernetes Authors.

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
	fed_clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

func NewClientCache(loader clientcmd.ClientConfig) *ClientCache {
	return &ClientCache{
		clients:       make(map[unversioned.GroupVersion]*client.Client),
		configs:       make(map[unversioned.GroupVersion]*restclient.Config),
		fedClientSets: make(map[unversioned.GroupVersion]fed_clientset.Interface),
		loader:        loader,
	}
}

// ClientCache caches previously loaded clients for reuse, and ensures MatchServerVersion
// is invoked only once
type ClientCache struct {
	loader        clientcmd.ClientConfig
	clients       map[unversioned.GroupVersion]*client.Client
	fedClientSets map[unversioned.GroupVersion]fed_clientset.Interface
	configs       map[unversioned.GroupVersion]*restclient.Config
	defaultConfig *restclient.Config
	defaultClient *client.Client
	matchVersion  bool
}

// ClientConfigForVersion returns the correct config for a server
func (c *ClientCache) ClientConfigForVersion(version *unversioned.GroupVersion) (*restclient.Config, error) {
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
	if version != nil {
		if config, ok := c.configs[*version]; ok {
			return config, nil
		}
	}

	// TODO: have a better config copy method
	config := *c.defaultConfig

	// TODO these fall out when we finish the refactor
	var preferredGV *unversioned.GroupVersion
	if version != nil {
		versionCopy := *version
		preferredGV = &versionCopy
	}

	client.SetKubernetesDefaults(&config)
	negotiatedVersion, err := client.NegotiateVersion(c.defaultClient, &config, preferredGV, registered.EnabledVersions())
	if err != nil {
		return nil, err
	}
	config.GroupVersion = negotiatedVersion

	if version != nil {
		c.configs[*version] = &config
	}

	// `version` does not necessarily equal `config.Version`.  However, we know that we call this method again with
	// `config.Version`, we should get the config we've just built.
	configCopy := config
	c.configs[*config.GroupVersion] = &configCopy

	return &config, nil
}

// ClientForVersion initializes or reuses a client for the specified version, or returns an
// error if that is not possible
func (c *ClientCache) ClientForVersion(version *unversioned.GroupVersion) (*client.Client, error) {
	if version != nil {
		if client, ok := c.clients[*version]; ok {
			return client, nil
		}
	}
	config, err := c.ClientConfigForVersion(version)
	if err != nil {
		return nil, err
	}

	kubeclient, err := client.New(config)
	if err != nil {
		return nil, err
	}
	c.clients[*config.GroupVersion] = kubeclient

	// `version` does not necessarily equal `config.Version`.  However, we know that if we call this method again with
	// `version`, we should get a client based on the same config we just found.  There's no guarantee that a client
	// is copiable, so create a new client and save it in the cache.
	if version != nil {
		configCopy := *config
		kubeclient, err := client.New(&configCopy)
		if err != nil {
			return nil, err
		}
		c.clients[*version] = kubeclient
	}

	return kubeclient, nil
}

func (c *ClientCache) FederationClientSetForVersion(version *unversioned.GroupVersion) (fed_clientset.Interface, error) {
	if version != nil {
		if clientSet, found := c.fedClientSets[*version]; found {
			return clientSet, nil
		}
	}
	config, err := c.ClientConfigForVersion(version)
	if err != nil {
		return nil, err
	}

	// TODO: support multi versions of client with clientset
	clientSet, err := fed_clientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	c.fedClientSets[*config.GroupVersion] = clientSet

	if version != nil {
		configCopy := *config
		clientSet, err := fed_clientset.NewForConfig(&configCopy)
		if err != nil {
			return nil, err
		}
		c.fedClientSets[*version] = clientSet
	}

	return clientSet, nil
}

func (c *ClientCache) FederationClientForVersion(version *unversioned.GroupVersion) (*restclient.RESTClient, error) {
	fedClientSet, err := c.FederationClientSetForVersion(version)
	if err != nil {
		return nil, err
	}
	return fedClientSet.(*fed_clientset.Clientset).FederationClient.RESTClient, nil
}
