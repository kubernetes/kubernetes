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
	"k8s.io/apimachinery/pkg/runtime/serializer"
	discovery "k8s.io/client-go/discovery"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	rest "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/scheme"
	clientappsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1"
	clientauthenticationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1"
	clientauthenticationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1beta1"
	clientauthorizationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1"
	clientauthorizationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1beta1"
	clientautoscalingv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v1"
	clientautoscalingv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1"
	clientbatchv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v1"
	clientbatchv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v2alpha1"
	clientcertificatesv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
	clientcorev1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	clientextensionsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	clientpolicyv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/policy/v1beta1"
	clientrbacv1alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1"
	clientrbacv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1beta1"
	clientstoragev1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/storage/v1beta1"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	CoreV1() clientcorev1.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() clientcorev1.CoreV1Interface
	AppsV1beta1() clientappsv1beta1.AppsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Apps() clientappsv1beta1.AppsV1beta1Interface
	AuthenticationV1() clientauthenticationv1.AuthenticationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authentication() clientauthenticationv1.AuthenticationV1Interface
	AuthenticationV1beta1() clientauthenticationv1beta1.AuthenticationV1beta1Interface

	AuthorizationV1() clientauthorizationv1.AuthorizationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authorization() clientauthorizationv1.AuthorizationV1Interface
	AuthorizationV1beta1() clientauthorizationv1beta1.AuthorizationV1beta1Interface

	AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() clientautoscalingv1.AutoscalingV1Interface
	AutoscalingV2alpha1() clientautoscalingv2alpha1.AutoscalingV2alpha1Interface

	BatchV1() clientbatchv1.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() clientbatchv1.BatchV1Interface
	BatchV2alpha1() clientbatchv2alpha1.BatchV2alpha1Interface

	CertificatesV1beta1() clientcertificatesv1beta1.CertificatesV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Certificates() clientcertificatesv1beta1.CertificatesV1beta1Interface
	ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface
	PolicyV1beta1() clientpolicyv1beta1.PolicyV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Policy() clientpolicyv1beta1.PolicyV1beta1Interface
	RbacV1beta1() clientrbacv1beta1.RbacV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Rbac() clientrbacv1beta1.RbacV1beta1Interface
	RbacV1alpha1() clientrbacv1alpha1.RbacV1alpha1Interface

	StorageV1beta1() clientstoragev1beta1.StorageV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Storage() clientstoragev1beta1.StorageV1beta1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*clientcorev1.CoreV1Client
	*clientappsv1beta1.AppsV1beta1Client
	*clientauthenticationv1.AuthenticationV1Client
	*clientauthenticationv1beta1.AuthenticationV1beta1Client
	*clientauthorizationv1.AuthorizationV1Client
	*clientauthorizationv1beta1.AuthorizationV1beta1Client
	*clientautoscalingv1.AutoscalingV1Client
	*clientautoscalingv2alpha1.AutoscalingV2alpha1Client
	*clientbatchv1.BatchV1Client
	*clientbatchv2alpha1.BatchV2alpha1Client
	*clientcertificatesv1beta1.CertificatesV1beta1Client
	*clientextensionsv1beta1.ExtensionsV1beta1Client
	*clientpolicyv1beta1.PolicyV1beta1Client
	*clientrbacv1beta1.RbacV1beta1Client
	*clientrbacv1alpha1.RbacV1alpha1Client
	*clientstoragev1beta1.StorageV1beta1Client
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() clientcorev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() clientcorev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() clientappsv1beta1.AppsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta1Client
}

// Deprecated: Apps retrieves the default version of AppsClient.
// Please explicitly pick a version.
func (c *Clientset) Apps() clientappsv1beta1.AppsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta1Client
}

// AuthenticationV1 retrieves the AuthenticationV1Client
func (c *Clientset) AuthenticationV1() clientauthenticationv1.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// Deprecated: Authentication retrieves the default version of AuthenticationClient.
// Please explicitly pick a version.
func (c *Clientset) Authentication() clientauthenticationv1.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() clientauthenticationv1beta1.AuthenticationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1beta1Client
}

// AuthorizationV1 retrieves the AuthorizationV1Client
func (c *Clientset) AuthorizationV1() clientauthorizationv1.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// Deprecated: Authorization retrieves the default version of AuthorizationClient.
// Please explicitly pick a version.
func (c *Clientset) Authorization() clientauthorizationv1.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() clientauthorizationv1beta1.AuthorizationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1beta1Client
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() clientautoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// AutoscalingV2alpha1 retrieves the AutoscalingV2alpha1Client
func (c *Clientset) AutoscalingV2alpha1() clientautoscalingv2alpha1.AutoscalingV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV2alpha1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() clientbatchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() clientbatchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() clientbatchv2alpha1.BatchV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV2alpha1Client
}

// CertificatesV1beta1 retrieves the CertificatesV1beta1Client
func (c *Clientset) CertificatesV1beta1() clientcertificatesv1beta1.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// Deprecated: Certificates retrieves the default version of CertificatesClient.
// Please explicitly pick a version.
func (c *Clientset) Certificates() clientcertificatesv1beta1.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() clientpolicyv1beta1.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// Deprecated: Policy retrieves the default version of PolicyClient.
// Please explicitly pick a version.
func (c *Clientset) Policy() clientpolicyv1beta1.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// RbacV1beta1 retrieves the RbacV1beta1Client
func (c *Clientset) RbacV1beta1() clientrbacv1beta1.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// Deprecated: Rbac retrieves the default version of RbacClient.
// Please explicitly pick a version.
func (c *Clientset) Rbac() clientrbacv1beta1.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() clientrbacv1alpha1.RbacV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1alpha1Client
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() clientstoragev1beta1.StorageV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1beta1Client
}

// Deprecated: Storage retrieves the default version of StorageClient.
// Please explicitly pick a version.
func (c *Clientset) Storage() clientstoragev1beta1.StorageV1beta1Interface {
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
	if configShallowCopy.ParameterCodec == nil {
		configShallowCopy.ParameterCodec = scheme.ParameterCodec
	}
	if configShallowCopy.NegotiatedSerializer == nil {
		configShallowCopy.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: scheme.Codecs}
	}
	var cs Clientset
	var err error
	cs.CoreV1Client, err = clientcorev1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AppsV1beta1Client, err = clientappsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1Client, err = clientauthenticationv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1beta1Client, err = clientauthenticationv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1Client, err = clientauthorizationv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1beta1Client, err = clientauthorizationv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV1Client, err = clientautoscalingv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV2alpha1Client, err = clientautoscalingv2alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV1Client, err = clientbatchv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV2alpha1Client, err = clientbatchv2alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.CertificatesV1beta1Client, err = clientcertificatesv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsV1beta1Client, err = clientextensionsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.PolicyV1beta1Client, err = clientpolicyv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1beta1Client, err = clientrbacv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1alpha1Client, err = clientrbacv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageV1beta1Client, err = clientstoragev1beta1.NewForConfig(&configShallowCopy)
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
	cs.CoreV1Client = clientcorev1.NewForConfigOrDie(c)
	cs.AppsV1beta1Client = clientappsv1beta1.NewForConfigOrDie(c)
	cs.AuthenticationV1Client = clientauthenticationv1.NewForConfigOrDie(c)
	cs.AuthenticationV1beta1Client = clientauthenticationv1beta1.NewForConfigOrDie(c)
	cs.AuthorizationV1Client = clientauthorizationv1.NewForConfigOrDie(c)
	cs.AuthorizationV1beta1Client = clientauthorizationv1beta1.NewForConfigOrDie(c)
	cs.AutoscalingV1Client = clientautoscalingv1.NewForConfigOrDie(c)
	cs.AutoscalingV2alpha1Client = clientautoscalingv2alpha1.NewForConfigOrDie(c)
	cs.BatchV1Client = clientbatchv1.NewForConfigOrDie(c)
	cs.BatchV2alpha1Client = clientbatchv2alpha1.NewForConfigOrDie(c)
	cs.CertificatesV1beta1Client = clientcertificatesv1beta1.NewForConfigOrDie(c)
	cs.ExtensionsV1beta1Client = clientextensionsv1beta1.NewForConfigOrDie(c)
	cs.PolicyV1beta1Client = clientpolicyv1beta1.NewForConfigOrDie(c)
	cs.RbacV1beta1Client = clientrbacv1beta1.NewForConfigOrDie(c)
	cs.RbacV1alpha1Client = clientrbacv1alpha1.NewForConfigOrDie(c)
	cs.StorageV1beta1Client = clientstoragev1beta1.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.CoreV1Client = clientcorev1.New(c)
	cs.AppsV1beta1Client = clientappsv1beta1.New(c)
	cs.AuthenticationV1Client = clientauthenticationv1.New(c)
	cs.AuthenticationV1beta1Client = clientauthenticationv1beta1.New(c)
	cs.AuthorizationV1Client = clientauthorizationv1.New(c)
	cs.AuthorizationV1beta1Client = clientauthorizationv1beta1.New(c)
	cs.AutoscalingV1Client = clientautoscalingv1.New(c)
	cs.AutoscalingV2alpha1Client = clientautoscalingv2alpha1.New(c)
	cs.BatchV1Client = clientbatchv1.New(c)
	cs.BatchV2alpha1Client = clientbatchv2alpha1.New(c)
	cs.CertificatesV1beta1Client = clientcertificatesv1beta1.New(c)
	cs.ExtensionsV1beta1Client = clientextensionsv1beta1.New(c)
	cs.PolicyV1beta1Client = clientpolicyv1beta1.New(c)
	cs.RbacV1beta1Client = clientrbacv1beta1.New(c)
	cs.RbacV1alpha1Client = clientrbacv1alpha1.New(c)
	cs.StorageV1beta1Client = clientstoragev1beta1.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
