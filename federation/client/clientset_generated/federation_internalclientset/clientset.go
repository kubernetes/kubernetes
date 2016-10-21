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
	CoreInternalVersion() internalversioncore.CoreInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Core() internalversioncore.CoreInternalVersionInterface
	ExtensionsInternalVersion() internalversionextensions.ExtensionsInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() internalversionextensions.ExtensionsInternalVersionInterface
	FederationInternalVersion() internalversionfederation.FederationInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Federation() internalversionfederation.FederationInternalVersionInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*internalversioncore.CoreInternalVersionClient
	*internalversionextensions.ExtensionsInternalVersionClient
	*internalversionfederation.FederationInternalVersionClient
}

// CoreInternalVersion retrieves the CoreInternalVersionClient
func (c *Clientset) CoreInternalVersion() internalversioncore.CoreInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalVersionClient
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() internalversioncore.CoreInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalVersionClient
}

// ExtensionsInternalVersion retrieves the ExtensionsInternalVersionClient
func (c *Clientset) ExtensionsInternalVersion() internalversionextensions.ExtensionsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalVersionClient
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalVersionClient
}

// FederationInternalVersion retrieves the FederationInternalVersionClient
func (c *Clientset) FederationInternalVersion() internalversionfederation.FederationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.FederationInternalVersionClient
}

// Deprecated: Federation retrieves the default version of FederationClient.
// Please explicitly pick a version.
func (c *Clientset) Federation() internalversionfederation.FederationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.FederationInternalVersionClient
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
	clientset.CoreInternalVersionClient, err = internalversioncore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsInternalVersionClient, err = internalversionextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.FederationInternalVersionClient, err = internalversionfederation.NewForConfig(&configShallowCopy)
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
	clientset.CoreInternalVersionClient = internalversioncore.NewForConfigOrDie(c)
	clientset.ExtensionsInternalVersionClient = internalversionextensions.NewForConfigOrDie(c)
	clientset.FederationInternalVersionClient = internalversionfederation.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreInternalVersionClient = internalversioncore.New(c)
	clientset.ExtensionsInternalVersionClient = internalversionextensions.New(c)
	clientset.FederationInternalVersionClient = internalversionfederation.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
