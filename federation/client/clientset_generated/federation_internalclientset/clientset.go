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
	CoreInternalversion() internalversioncore.CoreInternalversionInterface
	Core() internalversioncore.CoreInternalversionInterface
	ExtensionsInternalversion() internalversionextensions.ExtensionsInternalversionInterface
	Extensions() internalversionextensions.ExtensionsInternalversionInterface
	FederationInternalversion() internalversionfederation.FederationInternalversionInterface
	Federation() internalversionfederation.FederationInternalversionInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*internalversioncore.CoreInternalversionClient
	*internalversionextensions.ExtensionsInternalversionClient
	*internalversionfederation.FederationInternalversionClient
}

// CoreInternalversion retrieves the CoreInternalversionClient
func (c *Clientset) CoreInternalversion() internalversioncore.CoreInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalversionClient
}

// Core retrieves the CoreInternalversionClient
func (c *Clientset) Core() internalversioncore.CoreInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalversionClient
}

// ExtensionsInternalversion retrieves the ExtensionsInternalversionClient
func (c *Clientset) ExtensionsInternalversion() internalversionextensions.ExtensionsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalversionClient
}

// Extensions retrieves the ExtensionsInternalversionClient
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalversionClient
}

// FederationInternalversion retrieves the FederationInternalversionClient
func (c *Clientset) FederationInternalversion() internalversionfederation.FederationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.FederationInternalversionClient
}

// Federation retrieves the FederationInternalversionClient
func (c *Clientset) Federation() internalversionfederation.FederationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.FederationInternalversionClient
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
	clientset.CoreInternalversionClient, err = internalversioncore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsInternalversionClient, err = internalversionextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.FederationInternalversionClient, err = internalversionfederation.NewForConfig(&configShallowCopy)
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
	clientset.CoreInternalversionClient = internalversioncore.NewForConfigOrDie(c)
	clientset.ExtensionsInternalversionClient = internalversionextensions.NewForConfigOrDie(c)
	clientset.FederationInternalversionClient = internalversionfederation.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreInternalversionClient = internalversioncore.New(c)
	clientset.ExtensionsInternalversionClient = internalversionextensions.New(c)
	clientset.FederationInternalversionClient = internalversionfederation.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
