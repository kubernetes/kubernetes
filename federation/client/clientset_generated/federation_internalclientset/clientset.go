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

package federation_internalclientset

import (
	"github.com/golang/glog"
	internalversioncore "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	internalversionextensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion"
	internalversionfederation "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() internalversioncore.CoreInterface

	Extensions() internalversionextensions.ExtensionsInterface

	Federation() internalversionfederation.FederationInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*internalversioncore.CoreClient
	*internalversionextensions.ExtensionsClient
	*internalversionfederation.FederationClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() internalversioncore.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() internalversionfederation.FederationInterface {
	if c == nil {
		return nil
	}
	return c.FederationClient
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *restclient.Config) (*Clientset, error) {
	configShallowCopy := *c
	if configShallowCopy.RateLimiter == nil && configShallowCopy.QPS > 0 {
		configShallowCopy.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(configShallowCopy.QPS, configShallowCopy.Burst)
	}
	var clientset Clientset
	var err error
	clientset.CoreClient, err = internalversioncore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsClient, err = internalversionextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.FederationClient, err = internalversionfederation.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}

	clientset.DiscoveryClient, err = discovery.NewDiscoveryClientForConfig(&configShallowCopy)
	if err != nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
		return nil, err
	}
	return &clientset, nil
}

// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *Clientset {
	var clientset Clientset
	clientset.CoreClient = internalversioncore.NewForConfigOrDie(c)
	clientset.ExtensionsClient = internalversionextensions.NewForConfigOrDie(c)
	clientset.FederationClient = internalversionfederation.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreClient = internalversioncore.New(c)
	clientset.ExtensionsClient = internalversionextensions.New(c)
	clientset.FederationClient = internalversionfederation.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
