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
	internalversionapps "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	internalversionauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	internalversionauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	internalversionautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	internalversionbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	internalversioncertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	internalversioncore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	internalversionextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	internalversionpolicy "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	internalversionrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	internalversionstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreInternalVersion() internalversioncore.CoreInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Core() internalversioncore.CoreInternalVersionInterface
	AppsInternalVersion() internalversionapps.AppsInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Apps() internalversionapps.AppsInternalVersionInterface
	AuthenticationInternalVersion() internalversionauthentication.AuthenticationInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Authentication() internalversionauthentication.AuthenticationInternalVersionInterface
	AuthorizationInternalVersion() internalversionauthorization.AuthorizationInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Authorization() internalversionauthorization.AuthorizationInternalVersionInterface
	AutoscalingInternalVersion() internalversionautoscaling.AutoscalingInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() internalversionautoscaling.AutoscalingInternalVersionInterface
	BatchInternalVersion() internalversionbatch.BatchInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Batch() internalversionbatch.BatchInternalVersionInterface
	CertificatesInternalVersion() internalversioncertificates.CertificatesInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Certificates() internalversioncertificates.CertificatesInternalVersionInterface
	ExtensionsInternalVersion() internalversionextensions.ExtensionsInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() internalversionextensions.ExtensionsInternalVersionInterface
	PolicyInternalVersion() internalversionpolicy.PolicyInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Policy() internalversionpolicy.PolicyInternalVersionInterface
	RbacInternalVersion() internalversionrbac.RbacInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Rbac() internalversionrbac.RbacInternalVersionInterface
	StorageInternalVersion() internalversionstorage.StorageInternalVersionInterface
	// Deprecated: please explicitly pick a version if possible.
	Storage() internalversionstorage.StorageInternalVersionInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*internalversioncore.CoreInternalVersionClient
	*internalversionapps.AppsInternalVersionClient
	*internalversionauthentication.AuthenticationInternalVersionClient
	*internalversionauthorization.AuthorizationInternalVersionClient
	*internalversionautoscaling.AutoscalingInternalVersionClient
	*internalversionbatch.BatchInternalVersionClient
	*internalversioncertificates.CertificatesInternalVersionClient
	*internalversionextensions.ExtensionsInternalVersionClient
	*internalversionpolicy.PolicyInternalVersionClient
	*internalversionrbac.RbacInternalVersionClient
	*internalversionstorage.StorageInternalVersionClient
}

// CoreInternalVersion retrieves the CoreInternalVersionClient
func (c *Clientset) CoreInternalVersion() internalversioncore.CoreInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalVersionClient
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() internalversioncore.CoreInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalVersionClient
}

// AppsInternalVersion retrieves the AppsInternalVersionClient
func (c *Clientset) AppsInternalVersion() internalversionapps.AppsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AppsInternalVersionClient
}

// Deprecated: Apps retrieves the default version of AppsClient.
// Please explicitly pick a version.
func (c *Clientset) Apps() internalversionapps.AppsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AppsInternalVersionClient
}

// AuthenticationInternalVersion retrieves the AuthenticationInternalVersionClient
func (c *Clientset) AuthenticationInternalVersion() internalversionauthentication.AuthenticationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationInternalVersionClient
}

// Deprecated: Authentication retrieves the default version of AuthenticationClient.
// Please explicitly pick a version.
func (c *Clientset) Authentication() internalversionauthentication.AuthenticationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationInternalVersionClient
}

// AuthorizationInternalVersion retrieves the AuthorizationInternalVersionClient
func (c *Clientset) AuthorizationInternalVersion() internalversionauthorization.AuthorizationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationInternalVersionClient
}

// Deprecated: Authorization retrieves the default version of AuthorizationClient.
// Please explicitly pick a version.
func (c *Clientset) Authorization() internalversionauthorization.AuthorizationInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationInternalVersionClient
}

// AutoscalingInternalVersion retrieves the AutoscalingInternalVersionClient
func (c *Clientset) AutoscalingInternalVersion() internalversionautoscaling.AutoscalingInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingInternalVersionClient
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() internalversionautoscaling.AutoscalingInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingInternalVersionClient
}

// BatchInternalVersion retrieves the BatchInternalVersionClient
func (c *Clientset) BatchInternalVersion() internalversionbatch.BatchInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.BatchInternalVersionClient
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() internalversionbatch.BatchInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.BatchInternalVersionClient
}

// CertificatesInternalVersion retrieves the CertificatesInternalVersionClient
func (c *Clientset) CertificatesInternalVersion() internalversioncertificates.CertificatesInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesInternalVersionClient
}

// Deprecated: Certificates retrieves the default version of CertificatesClient.
// Please explicitly pick a version.
func (c *Clientset) Certificates() internalversioncertificates.CertificatesInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesInternalVersionClient
}

// ExtensionsInternalVersion retrieves the ExtensionsInternalVersionClient
func (c *Clientset) ExtensionsInternalVersion() internalversionextensions.ExtensionsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalVersionClient
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalVersionClient
}

// PolicyInternalVersion retrieves the PolicyInternalVersionClient
func (c *Clientset) PolicyInternalVersion() internalversionpolicy.PolicyInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.PolicyInternalVersionClient
}

// Deprecated: Policy retrieves the default version of PolicyClient.
// Please explicitly pick a version.
func (c *Clientset) Policy() internalversionpolicy.PolicyInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.PolicyInternalVersionClient
}

// RbacInternalVersion retrieves the RbacInternalVersionClient
func (c *Clientset) RbacInternalVersion() internalversionrbac.RbacInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.RbacInternalVersionClient
}

// Deprecated: Rbac retrieves the default version of RbacClient.
// Please explicitly pick a version.
func (c *Clientset) Rbac() internalversionrbac.RbacInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.RbacInternalVersionClient
}

// StorageInternalVersion retrieves the StorageInternalVersionClient
func (c *Clientset) StorageInternalVersion() internalversionstorage.StorageInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.StorageInternalVersionClient
}

// Deprecated: Storage retrieves the default version of StorageClient.
// Please explicitly pick a version.
func (c *Clientset) Storage() internalversionstorage.StorageInternalVersionInterface {
	if c == nil {
		return nil
	}
	return c.StorageInternalVersionClient
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
	clientset.CoreInternalVersionClient, err = internalversioncore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AppsInternalVersionClient, err = internalversionapps.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthenticationInternalVersionClient, err = internalversionauthentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthorizationInternalVersionClient, err = internalversionauthorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AutoscalingInternalVersionClient, err = internalversionautoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchInternalVersionClient, err = internalversionbatch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CertificatesInternalVersionClient, err = internalversioncertificates.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsInternalVersionClient, err = internalversionextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.PolicyInternalVersionClient, err = internalversionpolicy.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.RbacInternalVersionClient, err = internalversionrbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.StorageInternalVersionClient, err = internalversionstorage.NewForConfig(&configShallowCopy)
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
	clientset.CoreInternalVersionClient = internalversioncore.NewForConfigOrDie(c)
	clientset.AppsInternalVersionClient = internalversionapps.NewForConfigOrDie(c)
	clientset.AuthenticationInternalVersionClient = internalversionauthentication.NewForConfigOrDie(c)
	clientset.AuthorizationInternalVersionClient = internalversionauthorization.NewForConfigOrDie(c)
	clientset.AutoscalingInternalVersionClient = internalversionautoscaling.NewForConfigOrDie(c)
	clientset.BatchInternalVersionClient = internalversionbatch.NewForConfigOrDie(c)
	clientset.CertificatesInternalVersionClient = internalversioncertificates.NewForConfigOrDie(c)
	clientset.ExtensionsInternalVersionClient = internalversionextensions.NewForConfigOrDie(c)
	clientset.PolicyInternalVersionClient = internalversionpolicy.NewForConfigOrDie(c)
	clientset.RbacInternalVersionClient = internalversionrbac.NewForConfigOrDie(c)
	clientset.StorageInternalVersionClient = internalversionstorage.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreInternalVersionClient = internalversioncore.New(c)
	clientset.AppsInternalVersionClient = internalversionapps.New(c)
	clientset.AuthenticationInternalVersionClient = internalversionauthentication.New(c)
	clientset.AuthorizationInternalVersionClient = internalversionauthorization.New(c)
	clientset.AutoscalingInternalVersionClient = internalversionautoscaling.New(c)
	clientset.BatchInternalVersionClient = internalversionbatch.New(c)
	clientset.CertificatesInternalVersionClient = internalversioncertificates.New(c)
	clientset.ExtensionsInternalVersionClient = internalversionextensions.New(c)
	clientset.PolicyInternalVersionClient = internalversionpolicy.New(c)
	clientset.RbacInternalVersionClient = internalversionrbac.New(c)
	clientset.StorageInternalVersionClient = internalversionstorage.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
