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
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	discovery "k8s.io/client-go/discovery"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/scheme"
	clientappsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	clientauthenticationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	clientauthorizationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	clientautoscalinginternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	clientbatchinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	clientcertificatesinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	clientcoreinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	clientextensionsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	clientpolicyinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	clientrbacinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	clientstorageinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() clientcoreinternalversion.CoreInterface

	Apps() clientappsinternalversion.AppsInterface

	Authentication() clientauthenticationinternalversion.AuthenticationInterface

	Authorization() clientauthorizationinternalversion.AuthorizationInterface

	Autoscaling() clientautoscalinginternalversion.AutoscalingInterface

	Batch() clientbatchinternalversion.BatchInterface

	Certificates() clientcertificatesinternalversion.CertificatesInterface

	Extensions() clientextensionsinternalversion.ExtensionsInterface

	Policy() clientpolicyinternalversion.PolicyInterface

	Rbac() clientrbacinternalversion.RbacInterface

	Storage() clientstorageinternalversion.StorageInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*clientcoreinternalversion.CoreClient
	*clientappsinternalversion.AppsClient
	*clientauthenticationinternalversion.AuthenticationClient
	*clientauthorizationinternalversion.AuthorizationClient
	*clientautoscalinginternalversion.AutoscalingClient
	*clientbatchinternalversion.BatchClient
	*clientcertificatesinternalversion.CertificatesClient
	*clientextensionsinternalversion.ExtensionsClient
	*clientpolicyinternalversion.PolicyClient
	*clientrbacinternalversion.RbacClient
	*clientstorageinternalversion.StorageClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() clientcoreinternalversion.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Apps retrieves the AppsClient
func (c *Clientset) Apps() clientappsinternalversion.AppsInterface {
	if c == nil {
		return nil
	}
	return c.AppsClient
}

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() clientauthenticationinternalversion.AuthenticationInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationClient
}

// Authorization retrieves the AuthorizationClient
func (c *Clientset) Authorization() clientauthorizationinternalversion.AuthorizationInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationClient
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

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() clientcertificatesinternalversion.CertificatesInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() clientextensionsinternalversion.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Policy retrieves the PolicyClient
func (c *Clientset) Policy() clientpolicyinternalversion.PolicyInterface {
	if c == nil {
		return nil
	}
	return c.PolicyClient
}

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() clientrbacinternalversion.RbacInterface {
	if c == nil {
		return nil
	}
	return c.RbacClient
}

// Storage retrieves the StorageClient
func (c *Clientset) Storage() clientstorageinternalversion.StorageInterface {
	if c == nil {
		return nil
	}
	return c.StorageClient
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
	cs.AppsClient, err = clientappsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationClient, err = clientauthenticationinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationClient, err = clientauthorizationinternalversion.NewForConfig(&configShallowCopy)
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
	cs.CertificatesClient, err = clientcertificatesinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsClient, err = clientextensionsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.PolicyClient, err = clientpolicyinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacClient, err = clientrbacinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageClient, err = clientstorageinternalversion.NewForConfig(&configShallowCopy)
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
	cs.AppsClient = clientappsinternalversion.NewForConfigOrDie(c)
	cs.AuthenticationClient = clientauthenticationinternalversion.NewForConfigOrDie(c)
	cs.AuthorizationClient = clientauthorizationinternalversion.NewForConfigOrDie(c)
	cs.AutoscalingClient = clientautoscalinginternalversion.NewForConfigOrDie(c)
	cs.BatchClient = clientbatchinternalversion.NewForConfigOrDie(c)
	cs.CertificatesClient = clientcertificatesinternalversion.NewForConfigOrDie(c)
	cs.ExtensionsClient = clientextensionsinternalversion.NewForConfigOrDie(c)
	cs.PolicyClient = clientpolicyinternalversion.NewForConfigOrDie(c)
	cs.RbacClient = clientrbacinternalversion.NewForConfigOrDie(c)
	cs.StorageClient = clientstorageinternalversion.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreClient = clientcoreinternalversion.New(c)
	cs.AppsClient = clientappsinternalversion.New(c)
	cs.AuthenticationClient = clientauthenticationinternalversion.New(c)
	cs.AuthorizationClient = clientauthorizationinternalversion.New(c)
	cs.AutoscalingClient = clientautoscalinginternalversion.New(c)
	cs.BatchClient = clientbatchinternalversion.New(c)
	cs.CertificatesClient = clientcertificatesinternalversion.New(c)
	cs.ExtensionsClient = clientextensionsinternalversion.New(c)
	cs.PolicyClient = clientpolicyinternalversion.New(c)
	cs.RbacClient = clientrbacinternalversion.New(c)
	cs.StorageClient = clientstorageinternalversion.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
