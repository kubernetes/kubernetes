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

package federation_release_1_3

import (
	"github.com/golang/glog"
	v1core "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_3/typed/core/v1"
	v1beta1federation "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_3/typed/federation/v1beta1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Federation() v1beta1federation.FederationInterface
	Core() v1core.CoreInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1beta1federation.FederationClient
	*v1core.CoreClient
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() v1beta1federation.FederationInterface {
	if c == nil {
		return nil
	}
	return c.FederationClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() v1core.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
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
	clientset.FederationClient, err = v1beta1federation.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CoreClient, err = v1core.NewForConfig(&configShallowCopy)
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
	clientset.FederationClient = v1beta1federation.NewForConfigOrDie(c)
	clientset.CoreClient = v1core.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *restclient.RESTClient) *Clientset {
	var clientset Clientset
	clientset.FederationClient = v1beta1federation.New(c)
	clientset.CoreClient = v1core.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
