/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"github.com/golang/glog"
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned"
	unversionedbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() unversionedcore.CoreInterface
	Extensions() unversionedextensions.ExtensionsInterface
	Autoscaling() unversionedautoscaling.AutoscalingInterface
	Batch() unversionedbatch.BatchInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*unversionedcore.CoreClient
	*unversionedextensions.ExtensionsClient
	*unversionedautoscaling.AutoscalingClient
	*unversionedbatch.BatchClient
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

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() unversionedautoscaling.AutoscalingInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingClient
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() unversionedbatch.BatchInterface {
	if c == nil {
		return nil
	}
	return c.BatchClient
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
	clientset.CoreClient, err = unversionedcore.NewForConfig(&configShallowCopy)
	if err != nil {
		return &clientset, err
	}
	clientset.ExtensionsClient, err = unversionedextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return &clientset, err
	}
	clientset.AutoscalingClient, err = unversionedautoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return &clientset, err
	}
	clientset.BatchClient, err = unversionedbatch.NewForConfig(&configShallowCopy)
	if err != nil {
		return &clientset, err
	}

	clientset.DiscoveryClient, err = discovery.NewDiscoveryClientForConfig(&configShallowCopy)
	if err != nil {
		glog.Errorf("failed to create the DiscoveryClient: %v", err)
	}
	return &clientset, err
}

// NewForConfigOrDie creates a new Clientset for the given config and
// panics if there is an error in the config.
func NewForConfigOrDie(c *restclient.Config) *Clientset {
	var clientset Clientset
	clientset.CoreClient = unversionedcore.NewForConfigOrDie(c)
	clientset.ExtensionsClient = unversionedextensions.NewForConfigOrDie(c)
	clientset.AutoscalingClient = unversionedautoscaling.NewForConfigOrDie(c)
	clientset.BatchClient = unversionedbatch.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *restclient.RESTClient) *Clientset {
	var clientset Clientset
	clientset.CoreClient = unversionedcore.New(c)
	clientset.ExtensionsClient = unversionedextensions.New(c)
	clientset.AutoscalingClient = unversionedautoscaling.New(c)
	clientset.BatchClient = unversionedbatch.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
