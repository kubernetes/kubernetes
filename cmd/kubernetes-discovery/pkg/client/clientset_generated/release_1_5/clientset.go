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

package release_1_5

import (
	"github.com/golang/glog"
	v1alpha1apiregistration "k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated/release_1_5/typed/apiregistration/v1alpha1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	ApiregistrationV1alpha1() v1alpha1apiregistration.ApiregistrationV1alpha1Interface
	// Deprecated: please explicitly pick a version if possible.
	Apiregistration() v1alpha1apiregistration.ApiregistrationV1alpha1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1alpha1apiregistration.ApiregistrationV1alpha1Client
}

// ApiregistrationV1alpha1 retrieves the ApiregistrationV1alpha1Client
func (c *Clientset) ApiregistrationV1alpha1() v1alpha1apiregistration.ApiregistrationV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.ApiregistrationV1alpha1Client
}

// Deprecated: Apiregistration retrieves the default version of ApiregistrationClient.
// Please explicitly pick a version.
func (c *Clientset) Apiregistration() v1alpha1apiregistration.ApiregistrationV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.ApiregistrationV1alpha1Client
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
	clientset.ApiregistrationV1alpha1Client, err = v1alpha1apiregistration.NewForConfig(&configShallowCopy)
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
	clientset.ApiregistrationV1alpha1Client = v1alpha1apiregistration.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.ApiregistrationV1alpha1Client = v1alpha1apiregistration.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
