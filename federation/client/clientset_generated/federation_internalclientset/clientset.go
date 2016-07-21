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
	unversionedcore "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/unversioned"
	unversionedfederation "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/unversioned"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Federation() unversionedfederation.FederationInterface
	Core() unversionedcore.CoreInterface
	Extensions() unversionedextensions.ExtensionsInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*unversionedfederation.FederationClient
	*unversionedcore.CoreClient
	*unversionedextensions.ExtensionsClient
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() unversionedfederation.FederationInterface {
	if c == nil {
		return nil
	}
	return c.FederationClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() unversionedcore.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() unversionedextensions.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
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
	clientset.FederationClient, err = unversionedfederation.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CoreClient, err = unversionedcore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsClient, err = unversionedextensions.NewForConfig(&configShallowCopy)
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
	clientset.FederationClient = unversionedfederation.NewForConfigOrDie(c)
	clientset.CoreClient = unversionedcore.NewForConfigOrDie(c)
	clientset.ExtensionsClient = unversionedextensions.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *restclient.RESTClient) *Clientset {
	var clientset Clientset
	clientset.FederationClient = unversionedfederation.New(c)
	clientset.CoreClient = unversionedcore.New(c)
	clientset.ExtensionsClient = unversionedextensions.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
