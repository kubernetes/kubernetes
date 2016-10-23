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
	v1alpha1apps "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1alpha1"
	v1beta1authentication "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1"
	v1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authorization/v1beta1"
	v1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1"
	v1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v1"
	v2alpha1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v2alpha1"
	v1alpha1certificates "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	v1beta1extensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1"
	v1alpha1policy "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1alpha1"
	v1alpha1rbac "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/rbac/v1alpha1"
	v1beta1storage "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/storage/v1beta1"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
	discovery "k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreV1() v1core.CoreV1Interface
	Core() v1core.CoreV1Interface
	AppsV1alpha1() v1alpha1apps.AppsV1alpha1Interface
	Apps() v1alpha1apps.AppsV1alpha1Interface
	AuthenticationV1beta1() v1beta1authentication.AuthenticationV1beta1Interface
	Authentication() v1beta1authentication.AuthenticationV1beta1Interface
	AuthorizationV1beta1() v1beta1authorization.AuthorizationV1beta1Interface
	Authorization() v1beta1authorization.AuthorizationV1beta1Interface
	AutoscalingV1() v1autoscaling.AutoscalingV1Interface
	Autoscaling() v1autoscaling.AutoscalingV1Interface
	BatchV2alpha1() v2alpha1batch.BatchV2alpha1Interface

	BatchV1() v1batch.BatchV1Interface
	Batch() v1batch.BatchV1Interface
	CertificatesV1alpha1() v1alpha1certificates.CertificatesV1alpha1Interface
	Certificates() v1alpha1certificates.CertificatesV1alpha1Interface
	ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface
	Extensions() v1beta1extensions.ExtensionsV1beta1Interface
	PolicyV1alpha1() v1alpha1policy.PolicyV1alpha1Interface
	Policy() v1alpha1policy.PolicyV1alpha1Interface
	RbacV1alpha1() v1alpha1rbac.RbacV1alpha1Interface
	Rbac() v1alpha1rbac.RbacV1alpha1Interface
	StorageV1beta1() v1beta1storage.StorageV1beta1Interface
	Storage() v1beta1storage.StorageV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*v1core.CoreV1Client
	*v1alpha1apps.AppsV1alpha1Client
	*v1beta1authentication.AuthenticationV1beta1Client
	*v1beta1authorization.AuthorizationV1beta1Client
	*v1autoscaling.AutoscalingV1Client
	*v2alpha1batch.BatchV2alpha1Client
	*v1batch.BatchV1Client
	*v1alpha1certificates.CertificatesV1alpha1Client
	*v1beta1extensions.ExtensionsV1beta1Client
	*v1alpha1policy.PolicyV1alpha1Client
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

// Core retrieves the CoreV1Client
func (c *Clientset) Core() v1core.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// AppsV1alpha1 retrieves the AppsV1alpha1Client
func (c *Clientset) AppsV1alpha1() v1alpha1apps.AppsV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1alpha1Client
}

// Apps retrieves the AppsV1alpha1Client
func (c *Clientset) Apps() v1alpha1apps.AppsV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1alpha1Client
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() v1beta1authentication.AuthenticationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1beta1Client
}

// Authentication retrieves the AuthenticationV1beta1Client
func (c *Clientset) Authentication() v1beta1authentication.AuthenticationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1beta1Client
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() v1beta1authorization.AuthorizationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1beta1Client
}

// Authorization retrieves the AuthorizationV1beta1Client
func (c *Clientset) Authorization() v1beta1authorization.AuthorizationV1beta1Interface {
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

// Autoscaling retrieves the AutoscalingV1Client
func (c *Clientset) Autoscaling() v1autoscaling.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() v2alpha1batch.BatchV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV2alpha1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Batch retrieves the BatchV1Client
func (c *Clientset) Batch() v1batch.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// CertificatesV1alpha1 retrieves the CertificatesV1alpha1Client
func (c *Clientset) CertificatesV1alpha1() v1alpha1certificates.CertificatesV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1alpha1Client
}

// Certificates retrieves the CertificatesV1alpha1Client
func (c *Clientset) Certificates() v1alpha1certificates.CertificatesV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1alpha1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// PolicyV1alpha1 retrieves the PolicyV1alpha1Client
func (c *Clientset) PolicyV1alpha1() v1alpha1policy.PolicyV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1alpha1Client
}

// Policy retrieves the PolicyV1alpha1Client
func (c *Clientset) Policy() v1alpha1policy.PolicyV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1alpha1Client
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() v1alpha1rbac.RbacV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1alpha1Client
}

// Rbac retrieves the RbacV1alpha1Client
func (c *Clientset) Rbac() v1alpha1rbac.RbacV1alpha1Interface {
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

// Storage retrieves the StorageV1beta1Client
func (c *Clientset) Storage() v1beta1storage.StorageV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1beta1Client
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
	clientset.CoreV1Client, err = v1core.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AppsV1alpha1Client, err = v1alpha1apps.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthenticationV1beta1Client, err = v1beta1authentication.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AuthorizationV1beta1Client, err = v1beta1authorization.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.AutoscalingV1Client, err = v1autoscaling.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchV2alpha1Client, err = v2alpha1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.BatchV1Client, err = v1batch.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.CertificatesV1alpha1Client, err = v1alpha1certificates.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.ExtensionsV1beta1Client, err = v1beta1extensions.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.PolicyV1alpha1Client, err = v1alpha1policy.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.RbacV1alpha1Client, err = v1alpha1rbac.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	clientset.StorageV1beta1Client, err = v1beta1storage.NewForConfig(&configShallowCopy)
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
	clientset.CoreV1Client = v1core.NewForConfigOrDie(c)
	clientset.AppsV1alpha1Client = v1alpha1apps.NewForConfigOrDie(c)
	clientset.AuthenticationV1beta1Client = v1beta1authentication.NewForConfigOrDie(c)
	clientset.AuthorizationV1beta1Client = v1beta1authorization.NewForConfigOrDie(c)
	clientset.AutoscalingV1Client = v1autoscaling.NewForConfigOrDie(c)
	clientset.BatchV2alpha1Client = v2alpha1batch.NewForConfigOrDie(c)
	clientset.BatchV1Client = v1batch.NewForConfigOrDie(c)
	clientset.CertificatesV1alpha1Client = v1alpha1certificates.NewForConfigOrDie(c)
	clientset.ExtensionsV1beta1Client = v1beta1extensions.NewForConfigOrDie(c)
	clientset.PolicyV1alpha1Client = v1alpha1policy.NewForConfigOrDie(c)
	clientset.RbacV1alpha1Client = v1alpha1rbac.NewForConfigOrDie(c)
	clientset.StorageV1beta1Client = v1beta1storage.NewForConfigOrDie(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &clientset
}

// New creates a new Clientset for the given RESTClient.
func New(c restclient.Interface) *Clientset {
	var clientset Clientset
	clientset.CoreV1Client = v1core.New(c)
	clientset.AppsV1alpha1Client = v1alpha1apps.New(c)
	clientset.AuthenticationV1beta1Client = v1beta1authentication.New(c)
	clientset.AuthorizationV1beta1Client = v1beta1authorization.New(c)
	clientset.AutoscalingV1Client = v1autoscaling.New(c)
	clientset.BatchV2alpha1Client = v2alpha1batch.New(c)
	clientset.BatchV1Client = v1batch.New(c)
	clientset.CertificatesV1alpha1Client = v1alpha1certificates.New(c)
	clientset.ExtensionsV1beta1Client = v1beta1extensions.New(c)
	clientset.PolicyV1alpha1Client = v1alpha1policy.New(c)
	clientset.RbacV1alpha1Client = v1alpha1rbac.New(c)
	clientset.StorageV1beta1Client = v1beta1storage.New(c)

	clientset.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &clientset
}
