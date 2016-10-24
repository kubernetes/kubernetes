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
	CoreInternalversion() internalversioncore.CoreInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Core() internalversioncore.CoreInternalversionInterface
	AppsInternalversion() internalversionapps.AppsInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Apps() internalversionapps.AppsInternalversionInterface
	AuthenticationInternalversion() internalversionauthentication.AuthenticationInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Authentication() internalversionauthentication.AuthenticationInternalversionInterface
	AuthorizationInternalversion() internalversionauthorization.AuthorizationInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Authorization() internalversionauthorization.AuthorizationInternalversionInterface
	AutoscalingInternalversion() internalversionautoscaling.AutoscalingInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() internalversionautoscaling.AutoscalingInternalversionInterface
	BatchInternalversion() internalversionbatch.BatchInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Batch() internalversionbatch.BatchInternalversionInterface
	CertificatesInternalversion() internalversioncertificates.CertificatesInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Certificates() internalversioncertificates.CertificatesInternalversionInterface
	ExtensionsInternalversion() internalversionextensions.ExtensionsInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() internalversionextensions.ExtensionsInternalversionInterface
	PolicyInternalversion() internalversionpolicy.PolicyInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Policy() internalversionpolicy.PolicyInternalversionInterface
	RbacInternalversion() internalversionrbac.RbacInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Rbac() internalversionrbac.RbacInternalversionInterface
	StorageInternalversion() internalversionstorage.StorageInternalversionInterface
	// Deprecated: please explicitly pick a version if possible.
	Storage() internalversionstorage.StorageInternalversionInterface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*internalversioncore.CoreInternalversionClient
	*internalversionapps.AppsInternalversionClient
	*internalversionauthentication.AuthenticationInternalversionClient
	*internalversionauthorization.AuthorizationInternalversionClient
	*internalversionautoscaling.AutoscalingInternalversionClient
	*internalversionbatch.BatchInternalversionClient
	*internalversioncertificates.CertificatesInternalversionClient
	*internalversionextensions.ExtensionsInternalversionClient
	*internalversionpolicy.PolicyInternalversionClient
	*internalversionrbac.RbacInternalversionClient
	*internalversionstorage.StorageInternalversionClient
}

// CoreInternalversion retrieves the CoreInternalversionClient
func (c *Clientset) CoreInternalversion() internalversioncore.CoreInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalversionClient
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() internalversioncore.CoreInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CoreInternalversionClient
}

// AppsInternalversion retrieves the AppsInternalversionClient
func (c *Clientset) AppsInternalversion() internalversionapps.AppsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AppsInternalversionClient
}

// Deprecated: Apps retrieves the default version of AppsClient.
// Please explicitly pick a version.
func (c *Clientset) Apps() internalversionapps.AppsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AppsInternalversionClient
}

// AuthenticationInternalversion retrieves the AuthenticationInternalversionClient
func (c *Clientset) AuthenticationInternalversion() internalversionauthentication.AuthenticationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationInternalversionClient
}

// Deprecated: Authentication retrieves the default version of AuthenticationClient.
// Please explicitly pick a version.
func (c *Clientset) Authentication() internalversionauthentication.AuthenticationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AuthenticationInternalversionClient
}

// AuthorizationInternalversion retrieves the AuthorizationInternalversionClient
func (c *Clientset) AuthorizationInternalversion() internalversionauthorization.AuthorizationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationInternalversionClient
}

// Deprecated: Authorization retrieves the default version of AuthorizationClient.
// Please explicitly pick a version.
func (c *Clientset) Authorization() internalversionauthorization.AuthorizationInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AuthorizationInternalversionClient
}

// AutoscalingInternalversion retrieves the AutoscalingInternalversionClient
func (c *Clientset) AutoscalingInternalversion() internalversionautoscaling.AutoscalingInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingInternalversionClient
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() internalversionautoscaling.AutoscalingInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.AutoscalingInternalversionClient
}

// BatchInternalversion retrieves the BatchInternalversionClient
func (c *Clientset) BatchInternalversion() internalversionbatch.BatchInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.BatchInternalversionClient
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() internalversionbatch.BatchInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.BatchInternalversionClient
}

// CertificatesInternalversion retrieves the CertificatesInternalversionClient
func (c *Clientset) CertificatesInternalversion() internalversioncertificates.CertificatesInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesInternalversionClient
}

// Deprecated: Certificates retrieves the default version of CertificatesClient.
// Please explicitly pick a version.
func (c *Clientset) Certificates() internalversioncertificates.CertificatesInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.CertificatesInternalversionClient
}

// ExtensionsInternalversion retrieves the ExtensionsInternalversionClient
func (c *Clientset) ExtensionsInternalversion() internalversionextensions.ExtensionsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalversionClient
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.ExtensionsInternalversionClient
}

// PolicyInternalversion retrieves the PolicyInternalversionClient
func (c *Clientset) PolicyInternalversion() internalversionpolicy.PolicyInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.PolicyInternalversionClient
}

// Deprecated: Policy retrieves the default version of PolicyClient.
// Please explicitly pick a version.
func (c *Clientset) Policy() internalversionpolicy.PolicyInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.PolicyInternalversionClient
}

// RbacInternalversion retrieves the RbacInternalversionClient
func (c *Clientset) RbacInternalversion() internalversionrbac.RbacInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.RbacInternalversionClient
}

// Deprecated: Rbac retrieves the default version of RbacClient.
// Please explicitly pick a version.
func (c *Clientset) Rbac() internalversionrbac.RbacInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.RbacInternalversionClient
}

// StorageInternalversion retrieves the StorageInternalversionClient
func (c *Clientset) StorageInternalversion() internalversionstorage.StorageInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.StorageInternalversionClient
}

// Deprecated: Storage retrieves the default version of StorageClient.
// Please explicitly pick a version.
func (c *Clientset) Storage() internalversionstorage.StorageInternalversionInterface {
	if c == nil {
		return nil
	}
	return c.StorageInternalversionClient
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
	clientset.CoreInternalversionClient, err = internalversioncore.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AppsInternalversionClient, err = internalversionapps.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthenticationInternalversionClient, err = internalversionauthentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthorizationInternalversionClient, err = internalversionauthorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AutoscalingInternalversionClient, err = internalversionautoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchInternalversionClient, err = internalversionbatch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CertificatesInternalversionClient, err = internalversioncertificates.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsInternalversionClient, err = internalversionextensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.PolicyInternalversionClient, err = internalversionpolicy.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.RbacInternalversionClient, err = internalversionrbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.StorageInternalversionClient, err = internalversionstorage.NewForConfig(&configShallowCopy)
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
	clientset.CoreInternalversionClient = internalversioncore.NewForConfigOrDie(c)
	clientset.AppsInternalversionClient = internalversionapps.NewForConfigOrDie(c)
	clientset.AuthenticationInternalversionClient = internalversionauthentication.NewForConfigOrDie(c)
	clientset.AuthorizationInternalversionClient = internalversionauthorization.NewForConfigOrDie(c)
	clientset.AutoscalingInternalversionClient = internalversionautoscaling.NewForConfigOrDie(c)
	clientset.BatchInternalversionClient = internalversionbatch.NewForConfigOrDie(c)
	clientset.CertificatesInternalversionClient = internalversioncertificates.NewForConfigOrDie(c)
	clientset.ExtensionsInternalversionClient = internalversionextensions.NewForConfigOrDie(c)
	clientset.PolicyInternalversionClient = internalversionpolicy.NewForConfigOrDie(c)
	clientset.RbacInternalversionClient = internalversionrbac.NewForConfigOrDie(c)
	clientset.StorageInternalversionClient = internalversionstorage.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreInternalversionClient = internalversioncore.New(c)
	clientset.AppsInternalversionClient = internalversionapps.New(c)
	clientset.AuthenticationInternalversionClient = internalversionauthentication.New(c)
	clientset.AuthorizationInternalversionClient = internalversionauthorization.New(c)
	clientset.AutoscalingInternalversionClient = internalversionautoscaling.New(c)
	clientset.BatchInternalversionClient = internalversionbatch.New(c)
	clientset.CertificatesInternalversionClient = internalversioncertificates.New(c)
	clientset.ExtensionsInternalversionClient = internalversionextensions.New(c)
	clientset.PolicyInternalversionClient = internalversionpolicy.New(c)
	clientset.RbacInternalversionClient = internalversionrbac.New(c)
	clientset.StorageInternalversionClient = internalversionstorage.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
