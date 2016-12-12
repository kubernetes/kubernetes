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

package clientset

import (
	"github.com/golang/glog"
	discovery "k8s.io/client-go/discovery"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
	v1beta1apps "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1"
	v1authentication "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1"
	v1beta1authentication "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1beta1"
	v1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1"
	v1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1beta1"
	v1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v1"
	v2alpha1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1"
	v1batch "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v1"
	v2alpha1batch "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v2alpha1"
	v1beta1certificates "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	v1beta1extensions "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	v1beta1policy "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/policy/v1beta1"
	v1alpha1rbac "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1"
	v1beta1rbac "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1beta1"
	v1beta1storage "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/storage/v1beta1"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreV1() v1core.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() v1core.CoreV1Interface
	AppsV1beta1() v1beta1apps.AppsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Apps() v1beta1apps.AppsV1beta1Interface
	AuthenticationV1() v1authentication.AuthenticationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authentication() v1authentication.AuthenticationV1Interface
	AuthenticationV1beta1() v1beta1authentication.AuthenticationV1beta1Interface

	AuthorizationV1() v1authorization.AuthorizationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authorization() v1authorization.AuthorizationV1Interface
	AuthorizationV1beta1() v1beta1authorization.AuthorizationV1beta1Interface

	AutoscalingV1() v1autoscaling.AutoscalingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() v1autoscaling.AutoscalingV1Interface
	AutoscalingV2alpha1() v2alpha1autoscaling.AutoscalingV2alpha1Interface

	BatchV1() v1batch.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() v1batch.BatchV1Interface
	BatchV2alpha1() v2alpha1batch.BatchV2alpha1Interface

	CertificatesV1beta1() v1beta1certificates.CertificatesV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Certificates() v1beta1certificates.CertificatesV1beta1Interface
	ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() v1beta1extensions.ExtensionsV1beta1Interface
	PolicyV1beta1() v1beta1policy.PolicyV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Policy() v1beta1policy.PolicyV1beta1Interface
	RbacV1beta1() v1beta1rbac.RbacV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Rbac() v1beta1rbac.RbacV1beta1Interface
	RbacV1alpha1() v1alpha1rbac.RbacV1alpha1Interface

	StorageV1beta1() v1beta1storage.StorageV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Storage() v1beta1storage.StorageV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1core.CoreV1Client
	*v1beta1apps.AppsV1beta1Client
	*v1authentication.AuthenticationV1Client
	*v1beta1authentication.AuthenticationV1beta1Client
	*v1authorization.AuthorizationV1Client
	*v1beta1authorization.AuthorizationV1beta1Client
	*v1autoscaling.AutoscalingV1Client
	*v2alpha1autoscaling.AutoscalingV2alpha1Client
	*v1batch.BatchV1Client
	*v2alpha1batch.BatchV2alpha1Client
	*v1beta1certificates.CertificatesV1beta1Client
	*v1beta1extensions.ExtensionsV1beta1Client
	*v1beta1policy.PolicyV1beta1Client
	*v1beta1rbac.RbacV1beta1Client
	*v1alpha1rbac.RbacV1alpha1Client
	*v1beta1storage.StorageV1beta1Client
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() v1core.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() v1core.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() v1beta1apps.AppsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta1Client
}

// Deprecated: Apps retrieves the default version of AppsClient.
// Please explicitly pick a version.
func (c *Clientset) Apps() v1beta1apps.AppsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta1Client
}

// AuthenticationV1 retrieves the AuthenticationV1Client
func (c *Clientset) AuthenticationV1() v1authentication.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// Deprecated: Authentication retrieves the default version of AuthenticationClient.
// Please explicitly pick a version.
func (c *Clientset) Authentication() v1authentication.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() v1beta1authentication.AuthenticationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1beta1Client
}

// AuthorizationV1 retrieves the AuthorizationV1Client
func (c *Clientset) AuthorizationV1() v1authorization.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// Deprecated: Authorization retrieves the default version of AuthorizationClient.
// Please explicitly pick a version.
func (c *Clientset) Authorization() v1authorization.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() v1beta1authorization.AuthorizationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1beta1Client
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() v1autoscaling.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() v1autoscaling.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// AutoscalingV2alpha1 retrieves the AutoscalingV2alpha1Client
func (c *Clientset) AutoscalingV2alpha1() v2alpha1autoscaling.AutoscalingV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV2alpha1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() v2alpha1batch.BatchV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV2alpha1Client
}

// CertificatesV1beta1 retrieves the CertificatesV1beta1Client
func (c *Clientset) CertificatesV1beta1() v1beta1certificates.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// Deprecated: Certificates retrieves the default version of CertificatesClient.
// Please explicitly pick a version.
func (c *Clientset) Certificates() v1beta1certificates.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() v1beta1policy.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// Deprecated: Policy retrieves the default version of PolicyClient.
// Please explicitly pick a version.
func (c *Clientset) Policy() v1beta1policy.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// RbacV1beta1 retrieves the RbacV1beta1Client
func (c *Clientset) RbacV1beta1() v1beta1rbac.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// Deprecated: Rbac retrieves the default version of RbacClient.
// Please explicitly pick a version.
func (c *Clientset) Rbac() v1beta1rbac.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() v1alpha1rbac.RbacV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1alpha1Client
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() v1beta1storage.StorageV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1beta1Client
}

// Deprecated: Storage retrieves the default version of StorageClient.
// Please explicitly pick a version.
func (c *Clientset) Storage() v1beta1storage.StorageV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1beta1Client
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
	cs.CoreV1Client, err = v1core.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AppsV1beta1Client, err = v1beta1apps.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1Client, err = v1authentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1beta1Client, err = v1beta1authentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1Client, err = v1authorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1beta1Client, err = v1beta1authorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV1Client, err = v1autoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV2alpha1Client, err = v2alpha1autoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV1Client, err = v1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV2alpha1Client, err = v2alpha1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.CertificatesV1beta1Client, err = v1beta1certificates.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsV1beta1Client, err = v1beta1extensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.PolicyV1beta1Client, err = v1beta1policy.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1beta1Client, err = v1beta1rbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1alpha1Client, err = v1alpha1rbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageV1beta1Client, err = v1beta1storage.NewForConfig(&configShallowCopy)
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
	cs.CoreV1Client = v1core.NewForConfigOrDie(c)
	cs.AppsV1beta1Client = v1beta1apps.NewForConfigOrDie(c)
	cs.AuthenticationV1Client = v1authentication.NewForConfigOrDie(c)
	cs.AuthenticationV1beta1Client = v1beta1authentication.NewForConfigOrDie(c)
	cs.AuthorizationV1Client = v1authorization.NewForConfigOrDie(c)
	cs.AuthorizationV1beta1Client = v1beta1authorization.NewForConfigOrDie(c)
	cs.AutoscalingV1Client = v1autoscaling.NewForConfigOrDie(c)
	cs.AutoscalingV2alpha1Client = v2alpha1autoscaling.NewForConfigOrDie(c)
	cs.BatchV1Client = v1batch.NewForConfigOrDie(c)
	cs.BatchV2alpha1Client = v2alpha1batch.NewForConfigOrDie(c)
	cs.CertificatesV1beta1Client = v1beta1certificates.NewForConfigOrDie(c)
	cs.ExtensionsV1beta1Client = v1beta1extensions.NewForConfigOrDie(c)
	cs.PolicyV1beta1Client = v1beta1policy.NewForConfigOrDie(c)
	cs.RbacV1beta1Client = v1beta1rbac.NewForConfigOrDie(c)
	cs.RbacV1alpha1Client = v1alpha1rbac.NewForConfigOrDie(c)
	cs.StorageV1beta1Client = v1beta1storage.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreV1Client = v1core.New(c)
	cs.AppsV1beta1Client = v1beta1apps.New(c)
	cs.AuthenticationV1Client = v1authentication.New(c)
	cs.AuthenticationV1beta1Client = v1beta1authentication.New(c)
	cs.AuthorizationV1Client = v1authorization.New(c)
	cs.AuthorizationV1beta1Client = v1beta1authorization.New(c)
	cs.AutoscalingV1Client = v1autoscaling.New(c)
	cs.AutoscalingV2alpha1Client = v2alpha1autoscaling.New(c)
	cs.BatchV1Client = v1batch.New(c)
	cs.BatchV2alpha1Client = v2alpha1batch.New(c)
	cs.CertificatesV1beta1Client = v1beta1certificates.New(c)
	cs.ExtensionsV1beta1Client = v1beta1extensions.New(c)
	cs.PolicyV1beta1Client = v1beta1policy.New(c)
	cs.RbacV1beta1Client = v1beta1rbac.New(c)
	cs.RbacV1alpha1Client = v1alpha1rbac.New(c)
	cs.StorageV1beta1Client = v1beta1storage.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
