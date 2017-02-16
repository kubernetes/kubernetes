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
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	discovery "k8s.io/client-go/discovery"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/scheme"
	clientautoscalinginternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion"
	clientbatchinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion"
	clientcoreinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	clientextensionsinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion"
	clientfederationinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() clientcoreinternalversion.CoreInterface

	Autoscaling() clientautoscalinginternalversion.AutoscalingInterface

	Batch() clientbatchinternalversion.BatchInterface

	Extensions() clientextensionsinternalversion.ExtensionsInterface

	Federation() clientfederationinternalversion.FederationInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*clientcoreinternalversion.CoreClient
	*clientautoscalinginternalversion.AutoscalingClient
	*clientbatchinternalversion.BatchClient
	*clientextensionsinternalversion.ExtensionsClient
	*clientfederationinternalversion.FederationClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() clientcoreinternalversion.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() clientautoscalinginternalversion.AutoscalingInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingClient
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() clientbatchinternalversion.BatchInterface {
	if c == nil {
		return nil
	}
	return c.BatchClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() clientextensionsinternalversion.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() clientfederationinternalversion.FederationInterface {
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
	if configShallowCopy.ParameterCodec == nil {
		configShallowCopy.ParameterCodec = scheme.ParameterCodec
	}
	if configShallowCopy.NegotiatedSerializer == nil {
		configShallowCopy.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: scheme.Codecs}
	}
	var cs Clientset
	var err error
	cs.CoreClient, err = clientcoreinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingClient, err = clientautoscalinginternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchClient, err = clientbatchinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsClient, err = clientextensionsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.FederationClient, err = clientfederationinternalversion.NewForConfig(&configShallowCopy)
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
	cs.CoreClient = clientcoreinternalversion.NewForConfigOrDie(c)
	cs.AutoscalingClient = clientautoscalinginternalversion.NewForConfigOrDie(c)
	cs.BatchClient = clientbatchinternalversion.NewForConfigOrDie(c)
	cs.ExtensionsClient = clientextensionsinternalversion.NewForConfigOrDie(c)
	cs.FederationClient = clientfederationinternalversion.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreClient = clientcoreinternalversion.New(c)
	cs.AutoscalingClient = clientautoscalinginternalversion.New(c)
	cs.BatchClient = clientbatchinternalversion.New(c)
	cs.ExtensionsClient = clientextensionsinternalversion.New(c)
	cs.FederationClient = clientfederationinternalversion.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
