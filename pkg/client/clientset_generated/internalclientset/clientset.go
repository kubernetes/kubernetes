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

package internalclientset

import (
	"github.com/golang/glog"
	unversionedauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned"
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned"
	unversionedbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	unversionedrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/unversioned"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() unversionedcore.CoreInterface
	Extensions() unversionedextensions.ExtensionsInterface
	Autoscaling() unversionedautoscaling.AutoscalingInterface
	Authentication() unversionedauthentication.AuthenticationInterface
	Batch() unversionedbatch.BatchInterface
	Rbac() unversionedrbac.RbacInterface
	Certificates() unversionedcertificates.CertificatesInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*unversionedcore.CoreClient
	*unversionedextensions.ExtensionsClient
	*unversionedautoscaling.AutoscalingClient
	*unversionedauthentication.AuthenticationClient
	*unversionedbatch.BatchClient
	*unversionedrbac.RbacClient
	*unversionedcertificates.CertificatesClient
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

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() unversionedauthentication.AuthenticationInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationClient
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() unversionedbatch.BatchInterface {
	if c == nil {
		return nil
	}
	return c.BatchClient
}

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() unversionedrbac.RbacInterface {
	if c == nil {
		return nil
	}
	return c.RbacClient
}

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() unversionedcertificates.CertificatesInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesClient
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
		return nil, err
	}
	clientset.ExtensionsClient, err = unversionedextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AutoscalingClient, err = unversionedautoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthenticationClient, err = unversionedauthentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchClient, err = unversionedbatch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.RbacClient, err = unversionedrbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CertificatesClient, err = unversionedcertificates.NewForConfig(&configShallowCopy)
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
	clientset.CoreClient = unversionedcore.NewForConfigOrDie(c)
	clientset.ExtensionsClient = unversionedextensions.NewForConfigOrDie(c)
	clientset.AutoscalingClient = unversionedautoscaling.NewForConfigOrDie(c)
	clientset.AuthenticationClient = unversionedauthentication.NewForConfigOrDie(c)
	clientset.BatchClient = unversionedbatch.NewForConfigOrDie(c)
	clientset.RbacClient = unversionedrbac.NewForConfigOrDie(c)
	clientset.CertificatesClient = unversionedcertificates.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *restclient.RESTClient) *Clientset {
	var clientset Clientset
	clientset.CoreClient = unversionedcore.New(c)
	clientset.ExtensionsClient = unversionedextensions.New(c)
	clientset.AutoscalingClient = unversionedautoscaling.New(c)
	clientset.AuthenticationClient = unversionedauthentication.New(c)
	clientset.BatchClient = unversionedbatch.New(c)
	clientset.RbacClient = unversionedrbac.New(c)
	clientset.CertificatesClient = unversionedcertificates.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
