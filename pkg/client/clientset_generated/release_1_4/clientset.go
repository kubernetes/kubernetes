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

package release_1_4

import (
	"github.com/golang/glog"
	v1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/authorization/v1beta1"
	v1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/autoscaling/v1"
	v1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/batch/v1"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/core/v1"
	v1beta1extensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/extensions/v1beta1"
	v1alpha1policy "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/typed/policy/v1alpha1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() v1core.CoreInterface
	Authorization() v1beta1authorization.AuthorizationInterface
	Autoscaling() v1autoscaling.AutoscalingInterface
	Batch() v1batch.BatchInterface
	Extensions() v1beta1extensions.ExtensionsInterface
	Policy() v1alpha1policy.PolicyInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1core.CoreClient
	*v1beta1authorization.AuthorizationClient
	*v1autoscaling.AutoscalingClient
	*v1batch.BatchClient
	*v1beta1extensions.ExtensionsClient
	*v1alpha1policy.PolicyClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() v1core.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Authorization retrieves the AuthorizationClient
func (c *Clientset) Authorization() v1beta1authorization.AuthorizationInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationClient
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() v1autoscaling.AutoscalingInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingClient
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() v1batch.BatchInterface {
	if c == nil {
		return nil
	}
	return c.BatchClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Policy retrieves the PolicyClient
func (c *Clientset) Policy() v1alpha1policy.PolicyInterface {
	if c == nil {
		return nil
	}
	return c.PolicyClient
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
	clientset.CoreClient, err = v1core.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthorizationClient, err = v1beta1authorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AutoscalingClient, err = v1autoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchClient, err = v1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsClient, err = v1beta1extensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.PolicyClient, err = v1alpha1policy.NewForConfig(&configShallowCopy)
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
	clientset.CoreClient = v1core.NewForConfigOrDie(c)
	clientset.AuthorizationClient = v1beta1authorization.NewForConfigOrDie(c)
	clientset.AutoscalingClient = v1autoscaling.NewForConfigOrDie(c)
	clientset.BatchClient = v1batch.NewForConfigOrDie(c)
	clientset.ExtensionsClient = v1beta1extensions.NewForConfigOrDie(c)
	clientset.PolicyClient = v1alpha1policy.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *restclient.RESTClient) *Clientset {
	var clientset Clientset
	clientset.CoreClient = v1core.New(c)
	clientset.AuthorizationClient = v1beta1authorization.New(c)
	clientset.AutoscalingClient = v1autoscaling.New(c)
	clientset.BatchClient = v1batch.New(c)
	clientset.ExtensionsClient = v1beta1extensions.New(c)
	clientset.PolicyClient = v1alpha1policy.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
