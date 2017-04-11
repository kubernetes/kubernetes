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

package internalclientset

import (
	glog "github.com/golang/glog"
	innsmouthinternalversion "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/internalclientset/typed/innsmouth/internalversion"
	miskatonicinternalversion "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/internalclientset/typed/miskatonic/internalversion"
	discovery "k8s.io/client-go/discovery"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Innsmouth() innsmouthinternalversion.InnsmouthInterface
	Miskatonic() miskatonicinternalversion.MiskatonicInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*innsmouthinternalversion.InnsmouthClient
	*miskatonicinternalversion.MiskatonicClient
}

// Innsmouth retrieves the InnsmouthClient
func (c *Clientset) Innsmouth() innsmouthinternalversion.InnsmouthInterface {
	if c == nil {
		return nil
	}
	return c.InnsmouthClient
}

// Miskatonic retrieves the MiskatonicClient
func (c *Clientset) Miskatonic() miskatonicinternalversion.MiskatonicInterface {
	if c == nil {
		return nil
	}
	return c.MiskatonicClient
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
	cs.InnsmouthClient, err = innsmouthinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.MiskatonicClient, err = miskatonicinternalversion.NewForConfig(&configShallowCopy)
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
	cs.InnsmouthClient = innsmouthinternalversion.NewForConfigOrDie(c)
	cs.MiskatonicClient = miskatonicinternalversion.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.InnsmouthClient = innsmouthinternalversion.New(c)
	cs.MiskatonicClient = miskatonicinternalversion.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
