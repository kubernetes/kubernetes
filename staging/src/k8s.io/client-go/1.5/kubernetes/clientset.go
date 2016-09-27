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

package kubernetes

import (
	"github.com/golang/glog"
	discovery "k8s.io/client-go/1.5/discovery"
	v1alpha1apps "k8s.io/client-go/1.5/kubernetes/typed/apps/v1alpha1"
	v1beta1authentication "k8s.io/client-go/1.5/kubernetes/typed/authentication/v1beta1"
	v1beta1authorization "k8s.io/client-go/1.5/kubernetes/typed/authorization/v1beta1"
	v1autoscaling "k8s.io/client-go/1.5/kubernetes/typed/autoscaling/v1"
	v1batch "k8s.io/client-go/1.5/kubernetes/typed/batch/v1"
	v1alpha1certificates "k8s.io/client-go/1.5/kubernetes/typed/certificates/v1alpha1"
	v1core "k8s.io/client-go/1.5/kubernetes/typed/core/v1"
	v1beta1extensions "k8s.io/client-go/1.5/kubernetes/typed/extensions/v1beta1"
	v1alpha1policy "k8s.io/client-go/1.5/kubernetes/typed/policy/v1alpha1"
	v1alpha1rbac "k8s.io/client-go/1.5/kubernetes/typed/rbac/v1alpha1"
	v1beta1storage "k8s.io/client-go/1.5/kubernetes/typed/storage/v1beta1"
	"k8s.io/client-go/1.5/pkg/util/flowcontrol"
	_ "k8s.io/client-go/1.5/plugin/pkg/client/auth"
	rest "k8s.io/client-go/1.5/rest"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() v1core.CoreInterface
	Apps() v1alpha1apps.AppsInterface
	Authentication() v1beta1authentication.AuthenticationInterface
	Authorization() v1beta1authorization.AuthorizationInterface
	Autoscaling() v1autoscaling.AutoscalingInterface
	Batch() v1batch.BatchInterface
	Certificates() v1alpha1certificates.CertificatesInterface
	Extensions() v1beta1extensions.ExtensionsInterface
	Policy() v1alpha1policy.PolicyInterface
	Rbac() v1alpha1rbac.RbacInterface
	Storage() v1beta1storage.StorageInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1core.CoreClient
	*v1alpha1apps.AppsClient
	*v1beta1authentication.AuthenticationClient
	*v1beta1authorization.AuthorizationClient
	*v1autoscaling.AutoscalingClient
	*v1batch.BatchClient
	*v1alpha1certificates.CertificatesClient
	*v1beta1extensions.ExtensionsClient
	*v1alpha1policy.PolicyClient
	*v1alpha1rbac.RbacClient
	*v1beta1storage.StorageClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() v1core.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Apps retrieves the AppsClient
func (c *Clientset) Apps() v1alpha1apps.AppsInterface {
	if c == nil {
		return nil
	}
	return c.AppsClient
}

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() v1beta1authentication.AuthenticationInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationClient
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

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() v1alpha1certificates.CertificatesInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesClient
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

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() v1alpha1rbac.RbacInterface {
	if c == nil {
		return nil
	}
	return c.RbacClient
}

// Storage retrieves the StorageClient
func (c *Clientset) Storage() v1beta1storage.StorageInterface {
	if c == nil {
		return nil
	}
	return c.StorageClient
}

// Discovery retrieves the DiscoveryClient
func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return c.DiscoveryClient
}

// NewForConfig creates a new Clientset for the given config.
func NewForConfig(c *rest.Config) (*Clientset, error) {
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
	clientset.AppsClient, err = v1alpha1apps.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthenticationClient, err = v1beta1authentication.NewForConfig(&configShallowCopy)
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
	clientset.CertificatesClient, err = v1alpha1certificates.NewForConfig(&configShallowCopy)
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
	clientset.RbacClient, err = v1alpha1rbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.StorageClient, err = v1beta1storage.NewForConfig(&configShallowCopy)
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
func NewForConfigOrDie(c *rest.Config) *Clientset {
	var clientset Clientset
	clientset.CoreClient = v1core.NewForConfigOrDie(c)
	clientset.AppsClient = v1alpha1apps.NewForConfigOrDie(c)
	clientset.AuthenticationClient = v1beta1authentication.NewForConfigOrDie(c)
	clientset.AuthorizationClient = v1beta1authorization.NewForConfigOrDie(c)
	clientset.AutoscalingClient = v1autoscaling.NewForConfigOrDie(c)
	clientset.BatchClient = v1batch.NewForConfigOrDie(c)
	clientset.CertificatesClient = v1alpha1certificates.NewForConfigOrDie(c)
	clientset.ExtensionsClient = v1beta1extensions.NewForConfigOrDie(c)
	clientset.PolicyClient = v1alpha1policy.NewForConfigOrDie(c)
	clientset.RbacClient = v1alpha1rbac.NewForConfigOrDie(c)
	clientset.StorageClient = v1beta1storage.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c *rest.RESTClient) *Clientset {
	var clientset Clientset
	clientset.CoreClient = v1core.New(c)
	clientset.AppsClient = v1alpha1apps.New(c)
	clientset.AuthenticationClient = v1beta1authentication.New(c)
	clientset.AuthorizationClient = v1beta1authorization.New(c)
	clientset.AutoscalingClient = v1autoscaling.New(c)
	clientset.BatchClient = v1batch.New(c)
	clientset.CertificatesClient = v1alpha1certificates.New(c)
	clientset.ExtensionsClient = v1beta1extensions.New(c)
	clientset.PolicyClient = v1alpha1policy.New(c)
	clientset.RbacClient = v1alpha1rbac.New(c)
	clientset.StorageClient = v1beta1storage.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
