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

package federation_clientset

import (
	"github.com/golang/glog"
	v1batch "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/batch/v1"
	v1core "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1"
	v1beta1extensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1"
	v1beta1federation "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreV1() v1core.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() v1core.CoreV1Interface
	BatchV1() v1batch.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() v1batch.BatchV1Interface
	ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() v1beta1extensions.ExtensionsV1beta1Interface
	FederationV1beta1() v1beta1federation.FederationV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Federation() v1beta1federation.FederationV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1core.CoreV1Client
	*v1batch.BatchV1Client
	*v1beta1extensions.ExtensionsV1beta1Client
	*v1beta1federation.FederationV1beta1Client
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() v1core.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() v1core.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// FederationV1beta1 retrieves the FederationV1beta1Client
func (c *Clientset) FederationV1beta1() v1beta1federation.FederationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.FederationV1beta1Client
}

// Deprecated: Federation retrieves the default version of FederationClient.
// Please explicitly pick a version.
func (c *Clientset) Federation() v1beta1federation.FederationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.FederationV1beta1Client
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	if c == nil {
		return nil
	}
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *restclient.Config) (*Clientset, error) {
	configShallowCopy := *c
	if configShallowCopy.RateLimiter == nil && configShallowCopy.QPS > 0 {
		configShallowCopy.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(configShallowCopy.QPS, configShallowCopy.Burst)
	}
	var cs Clientset
	var err error
	cs.CoreV1Client, err = v1core.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV1Client, err = v1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsV1beta1Client, err = v1beta1extensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.FederationV1beta1Client, err = v1beta1federation.NewForConfig(&configShallowCopy)
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
func NewForConfigOrDie(c *restclient.Config) *Clientset {
	var cs Clientset
	cs.CoreV1Client = v1core.NewForConfigOrDie(c)
	cs.BatchV1Client = v1batch.NewForConfigOrDie(c)
	cs.ExtensionsV1beta1Client = v1beta1extensions.NewForConfigOrDie(c)
	cs.FederationV1beta1Client = v1beta1federation.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var cs Clientset
	cs.CoreV1Client = v1core.New(c)
	cs.BatchV1Client = v1batch.New(c)
	cs.ExtensionsV1beta1Client = v1beta1extensions.New(c)
	cs.FederationV1beta1Client = v1beta1federation.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
