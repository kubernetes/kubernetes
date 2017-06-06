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
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	oldclient "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/version"
)

func NewClientCache(loader clientcmd.ClientConfig, discoveryClientFactory DiscoveryClientFactory) *ClientCache {
	return &ClientCache{
		clientsets:             make(map[schema.GroupVersion]internalclientset.Interface),
		configs:                make(map[schema.GroupVersion]*restclient.Config),
		fedClientSets:          make(map[schema.GroupVersion]fedclientset.Interface),
		loader:                 loader,
		discoveryClientFactory: discoveryClientFactory,
	}
}

// ClientCache caches previously loaded clients for reuse, and ensures MatchServerVersion
// is invoked only once
type ClientCache struct {
	loader        clientcmd.ClientConfig
	clientsets    map[schema.GroupVersion]internalclientset.Interface
	fedClientSets map[schema.GroupVersion]fedclientset.Interface
	configs       map[schema.GroupVersion]*restclient.Config

	// noVersionConfig provides a cached config for the case of no required version specified
	noVersionConfig *restclient.Config

	matchVersion bool

	lock          sync.Mutex
	defaultConfig *restclient.Config
	// discoveryClientFactory comes as a factory method so that we can defer resolution until after
	// argument evaluation
	discoveryClientFactory DiscoveryClientFactory
	discoveryClient        discovery.DiscoveryInterface
}

// also looks up the discovery client.  We can't do this during init because the flags won't have been set
// because this is constructed pre-command execution before the command tree is
// even set up. Requires the lock to already be acquired
func (c *ClientCache) getDefaultConfig() (restclient.Config, discovery.DiscoveryInterface, error) {
	if c.defaultConfig != nil && c.discoveryClient != nil {
		return *c.defaultConfig, c.discoveryClient, nil
	}

	config, err := c.loader.ClientConfig()
	if err != nil {
		return restclient.Config{}, nil, err
	}
	discoveryClient, err := c.discoveryClientFactory.DiscoveryClient()
	if err != nil {
		return restclient.Config{}, nil, err
	}
	if c.matchVersion {
		if err := discovery.MatchesServerVersion(version.Get(), discoveryClient); err != nil {
			return restclient.Config{}, nil, err
		}
	}

	c.defaultConfig = config
	c.discoveryClient = discoveryClient
	return *c.defaultConfig, c.discoveryClient, nil
}

// ClientConfigForVersion returns the correct config for a server
func (c *ClientCache) ClientConfigForVersion(requiredVersion *schema.GroupVersion) (*restclient.Config, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.clientConfigForVersion(requiredVersion)
}

// clientConfigForVersion returns the correct config for a server
func (c *ClientCache) clientConfigForVersion(requiredVersion *schema.GroupVersion) (*restclient.Config, error) {
	// only lookup in the cache if the requiredVersion is set
	if requiredVersion != nil {
		if config, ok := c.configs[*requiredVersion]; ok {
			return copyConfig(config), nil
		}
	} else if c.noVersionConfig != nil {
		return copyConfig(c.noVersionConfig), nil
	}

	// this returns a shallow copy to work with
	config, discoveryClient, err := c.getDefaultConfig()
	if err != nil {
		return nil, err
	}

	if requiredVersion != nil {
		if err := discovery.ServerSupportsVersion(discoveryClient, *requiredVersion); err != nil {
			return nil, err
		}
		config.GroupVersion = requiredVersion
	} else {
		// TODO remove this hack.  This is allowing the GetOptions to be serialized.
		config.GroupVersion = &schema.GroupVersion{Group: "", Version: "v1"}
	}

	// TODO this isn't what we want.  Each clientset should be setting defaults as it sees fit.
	oldclient.SetKubernetesDefaults(&config)

	if requiredVersion != nil {
		c.configs[*requiredVersion] = copyConfig(&config)
	} else {
		c.noVersionConfig = copyConfig(&config)
	}

	// `version` does not necessarily equal `config.Version`.  However, we know that we call this method again with
	// `config.Version`, we should get the config we've just built.
	c.configs[*config.GroupVersion] = copyConfig(&config)

	return copyConfig(&config), nil
}

func copyConfig(in *restclient.Config) *restclient.Config {
	configCopy := *in
	copyGroupVersion := *configCopy.GroupVersion
	configCopy.GroupVersion = &copyGroupVersion
	return &configCopy
}

// ClientSetForVersion initializes or reuses a clientset for the specified version, or returns an
// error if that is not possible
func (c *ClientCache) ClientSetForVersion(requiredVersion *schema.GroupVersion) (internalclientset.Interface, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if requiredVersion != nil {
		if clientset, ok := c.clientsets[*requiredVersion]; ok {
			return clientset, nil
		}
	}
	config, err := c.clientConfigForVersion(requiredVersion)
	if err != nil {
		return nil, err
	}

	clientset, err := internalclientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	c.clientsets[*config.GroupVersion] = clientset

	// `version` does not necessarily equal `config.Version`.  However, we know that if we call this method again with
	// `version`, we should get a client based on the same config we just found.  There's no guarantee that a client
	// is copiable, so create a new client and save it in the cache.
	if requiredVersion != nil {
		configCopy := *config
		clientset, err := internalclientset.NewForConfig(&configCopy)
		if err != nil {
			return nil, err
		}
		c.clientsets[*requiredVersion] = clientset
	}

	return clientset, nil
}

func (c *ClientCache) FederationClientSetForVersion(version *schema.GroupVersion) (fedclientset.Interface, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	return c.federationClientSetForVersion(version)
}

func (c *ClientCache) federationClientSetForVersion(version *schema.GroupVersion) (fedclientset.Interface, error) {
	if version != nil {
		if clientSet, found := c.fedClientSets[*version]; found {
			return clientSet, nil
		}
	}
	config, err := c.clientConfigForVersion(version)
	if err != nil {
		return nil, err
	}

	// TODO: support multi versions of client with clientset
	clientSet, err := fedclientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	c.fedClientSets[*config.GroupVersion] = clientSet

	if version != nil {
		configCopy := *config
		clientSet, err := fedclientset.NewForConfig(&configCopy)
		if err != nil {
			return nil, err
		}
		c.fedClientSets[*version] = clientSet
	}

	return clientSet, nil
}

func (c *ClientCache) FederationClientForVersion(version *schema.GroupVersion) (*restclient.RESTClient, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	fedClientSet, err := c.federationClientSetForVersion(version)
	if err != nil {
		return nil, err
	}
	return fedClientSet.Federation().RESTClient().(*restclient.RESTClient), nil
}
