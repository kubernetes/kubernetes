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
	discovery "k8s.io/client-go/discovery"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
	appsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	authenticationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	authorizationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	autoscalinginternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	batchinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	certificatesinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	coreinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	extensionsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	policyinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	rbacinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	settingsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/settings/internalversion"
	storageinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	Core() coreinternalversion.CoreInterface
	Apps() appsinternalversion.AppsInterface
	Authentication() authenticationinternalversion.AuthenticationInterface
	Authorization() authorizationinternalversion.AuthorizationInterface
	Autoscaling() autoscalinginternalversion.AutoscalingInterface
	Batch() batchinternalversion.BatchInterface
	Certificates() certificatesinternalversion.CertificatesInterface
	Extensions() extensionsinternalversion.ExtensionsInterface
	Policy() policyinternalversion.PolicyInterface
	Rbac() rbacinternalversion.RbacInterface
	Settings() settingsinternalversion.SettingsInterface
	Storage() storageinternalversion.StorageInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*coreinternalversion.CoreClient
	*appsinternalversion.AppsClient
	*authenticationinternalversion.AuthenticationClient
	*authorizationinternalversion.AuthorizationClient
	*autoscalinginternalversion.AutoscalingClient
	*batchinternalversion.BatchClient
	*certificatesinternalversion.CertificatesClient
	*extensionsinternalversion.ExtensionsClient
	*policyinternalversion.PolicyClient
	*rbacinternalversion.RbacClient
	*settingsinternalversion.SettingsClient
	*storageinternalversion.StorageClient
}

// Core retrieves the CoreClient
func (c *Clientset) Core() coreinternalversion.CoreInterface {
	if c == nil {
		return nil
	}
	return c.CoreClient
}

// Apps retrieves the AppsClient
func (c *Clientset) Apps() appsinternalversion.AppsInterface {
	if c == nil {
		return nil
	}
	return c.AppsClient
}

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() authenticationinternalversion.AuthenticationInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationClient
}

// Authorization retrieves the AuthorizationClient
func (c *Clientset) Authorization() authorizationinternalversion.AuthorizationInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationClient
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

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() certificatesinternalversion.CertificatesInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesClient
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() extensionsinternalversion.ExtensionsInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsClient
}

// Policy retrieves the PolicyClient
func (c *Clientset) Policy() policyinternalversion.PolicyInterface {
	if c == nil {
		return nil
	}
	return c.PolicyClient
}

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() rbacinternalversion.RbacInterface {
	if c == nil {
		return nil
	}
	return c.RbacClient
}

// Settings retrieves the SettingsClient
func (c *Clientset) Settings() settingsinternalversion.SettingsInterface {
	if c == nil {
		return nil
	}
	return c.SettingsClient
}

// Storage retrieves the StorageClient
func (c *Clientset) Storage() storageinternalversion.StorageInterface {
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
	var cs Clientset
	var err error
	cs.CoreClient, err = coreinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AppsClient, err = appsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationClient, err = authenticationinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationClient, err = authorizationinternalversion.NewForConfig(&configShallowCopy)
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
	cs.CertificatesClient, err = certificatesinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsClient, err = extensionsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.PolicyClient, err = policyinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacClient, err = rbacinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.SettingsClient, err = settingsinternalversion.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageClient, err = storageinternalversion.NewForConfig(&configShallowCopy)
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
	cs.AppsClient = appsinternalversion.NewForConfigOrDie(c)
	cs.AuthenticationClient = authenticationinternalversion.NewForConfigOrDie(c)
	cs.AuthorizationClient = authorizationinternalversion.NewForConfigOrDie(c)
	cs.AutoscalingClient = autoscalinginternalversion.NewForConfigOrDie(c)
	cs.BatchClient = batchinternalversion.NewForConfigOrDie(c)
	cs.CertificatesClient = certificatesinternalversion.NewForConfigOrDie(c)
	cs.ExtensionsClient = extensionsinternalversion.NewForConfigOrDie(c)
	cs.PolicyClient = policyinternalversion.NewForConfigOrDie(c)
	cs.RbacClient = rbacinternalversion.NewForConfigOrDie(c)
	cs.SettingsClient = settingsinternalversion.NewForConfigOrDie(c)
	cs.StorageClient = storageinternalversion.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreClient = coreinternalversion.New(c)
	cs.AppsClient = appsinternalversion.New(c)
	cs.AuthenticationClient = authenticationinternalversion.New(c)
	cs.AuthorizationClient = authorizationinternalversion.New(c)
	cs.AutoscalingClient = autoscalinginternalversion.New(c)
	cs.BatchClient = batchinternalversion.New(c)
	cs.CertificatesClient = certificatesinternalversion.New(c)
	cs.ExtensionsClient = extensionsinternalversion.New(c)
	cs.PolicyClient = policyinternalversion.New(c)
	cs.RbacClient = rbacinternalversion.New(c)
	cs.SettingsClient = settingsinternalversion.New(c)
	cs.StorageClient = storageinternalversion.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
