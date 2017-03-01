/*
Copyright 2017 The Kubernetes Authors.

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
	glog "github.com/golang/glog"
	discovery "k8s.io/client-go/discovery"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
	autoscalinginternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion"
	batchinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion"
	coreinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	extensionsinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion"
	federationinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() coreinternalversion.CoreInterface
	Autoscaling() autoscalinginternalversion.AutoscalingInterface
	Batch() batchinternalversion.BatchInterface
	Extensions() extensionsinternalversion.ExtensionsInterface
	Federation() federationinternalversion.FederationInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*coreinternalversion.CoreClient
	*autoscalinginternalversion.AutoscalingClient
	*batchinternalversion.BatchClient
	*extensionsinternalversion.ExtensionsClient
	*federationinternalversion.FederationClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() coreinternalversion.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() autoscalinginternalversion.AutoscalingInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingClient
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() batchinternalversion.BatchInterface {
	if c == nil {
		return nil
	}
	return c.BatchClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() extensionsinternalversion.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() federationinternalversion.FederationInterface {
	if c == nil {
		return nil
	}
	return c.FederationClient
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	if c == nil {
		return nil
	}
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *rest.Config) (*Clientset, error) {
	configShallowCopy := *c
	if configShallowCopy.RateLimiter == nil && configShallowCopy.QPS > 0 {
		configShallowCopy.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(configShallowCopy.QPS, configShallowCopy.Burst)
	}
	var cs Clientset
	var err error
	cs.CoreClient, err = coreinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingClient, err = autoscalinginternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchClient, err = batchinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsClient, err = extensionsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.FederationClient, err = federationinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}

	cs.DiscoveryClient, err = discovery.NewDiscoveryClientForConfig(&configShallowCopy)
	if err != nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
		return nil, err
	}
	return &cs, nil
}

// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *rest.Config) *Clientset {
	var cs Clientset
	cs.CoreClient = coreinternalversion.NewForConfigOrDie(c)
	cs.AutoscalingClient = autoscalinginternalversion.NewForConfigOrDie(c)
	cs.BatchClient = batchinternalversion.NewForConfigOrDie(c)
	cs.ExtensionsClient = extensionsinternalversion.NewForConfigOrDie(c)
	cs.FederationClient = federationinternalversion.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreClient = coreinternalversion.New(c)
	cs.AutoscalingClient = autoscalinginternalversion.New(c)
	cs.BatchClient = batchinternalversion.New(c)
	cs.ExtensionsClient = extensionsinternalversion.New(c)
	cs.FederationClient = federationinternalversion.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
