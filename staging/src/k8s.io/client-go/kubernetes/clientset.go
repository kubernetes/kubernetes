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

package kubernetes

import (
	glog "github.com/golang/glog"
	discovery "k8s.io/client-go/discovery"
	admissionregistrationv1alpha1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1alpha1"
	appsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1"
	appsv1beta2 "k8s.io/client-go/kubernetes/typed/apps/v1beta2"
	authenticationv1 "k8s.io/client-go/kubernetes/typed/authentication/v1"
	authenticationv1beta1 "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	authorizationv1 "k8s.io/client-go/kubernetes/typed/authorization/v1"
	authorizationv1beta1 "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"
	autoscalingv1 "k8s.io/client-go/kubernetes/typed/autoscaling/v1"
	autoscalingv2alpha1 "k8s.io/client-go/kubernetes/typed/autoscaling/v2alpha1"
	batchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	batchv2alpha1 "k8s.io/client-go/kubernetes/typed/batch/v2alpha1"
	certificatesv1beta1 "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	extensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	networkingv1 "k8s.io/client-go/kubernetes/typed/networking/v1"
	policyv1beta1 "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	rbacv1alpha1 "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
	schedulingv1alpha1 "k8s.io/client-go/kubernetes/typed/scheduling/v1alpha1"
	settingsv1alpha1 "k8s.io/client-go/kubernetes/typed/settings/v1alpha1"
	storagev1 "k8s.io/client-go/kubernetes/typed/storage/v1"
	storagev1beta1 "k8s.io/client-go/kubernetes/typed/storage/v1beta1"
	rest "k8s.io/client-go/rest"
	flowcontrol "k8s.io/client-go/util/flowcontrol"
)

type Interface interface {
	Discovery() discovery.DiscoveryInterface
	AdmissionregistrationV1alpha1() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface
	// Deprecated: please explicitly pick a version if possible.
	Admissionregistration() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface
	AppsV1beta1() appsv1beta1.AppsV1beta1Interface
	AppsV1beta2() appsv1beta2.AppsV1beta2Interface
	// Deprecated: please explicitly pick a version if possible.
	Apps() appsv1beta2.AppsV1beta2Interface
	AuthenticationV1() authenticationv1.AuthenticationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authentication() authenticationv1.AuthenticationV1Interface
	AuthenticationV1beta1() authenticationv1beta1.AuthenticationV1beta1Interface
	AuthorizationV1() authorizationv1.AuthorizationV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Authorization() authorizationv1.AuthorizationV1Interface
	AuthorizationV1beta1() authorizationv1beta1.AuthorizationV1beta1Interface
	AutoscalingV1() autoscalingv1.AutoscalingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Autoscaling() autoscalingv1.AutoscalingV1Interface
	AutoscalingV2alpha1() autoscalingv2alpha1.AutoscalingV2alpha1Interface
	BatchV1() batchv1.BatchV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Batch() batchv1.BatchV1Interface
	BatchV2alpha1() batchv2alpha1.BatchV2alpha1Interface
	CertificatesV1beta1() certificatesv1beta1.CertificatesV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Certificates() certificatesv1beta1.CertificatesV1beta1Interface
	CoreV1() corev1.CoreV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Core() corev1.CoreV1Interface
	ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Extensions() extensionsv1beta1.ExtensionsV1beta1Interface
	NetworkingV1() networkingv1.NetworkingV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Networking() networkingv1.NetworkingV1Interface
	PolicyV1beta1() policyv1beta1.PolicyV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Policy() policyv1beta1.PolicyV1beta1Interface
	RbacV1beta1() rbacv1beta1.RbacV1beta1Interface
	// Deprecated: please explicitly pick a version if possible.
	Rbac() rbacv1beta1.RbacV1beta1Interface
	RbacV1alpha1() rbacv1alpha1.RbacV1alpha1Interface
	SchedulingV1alpha1() schedulingv1alpha1.SchedulingV1alpha1Interface
	// Deprecated: please explicitly pick a version if possible.
	Scheduling() schedulingv1alpha1.SchedulingV1alpha1Interface
	SettingsV1alpha1() settingsv1alpha1.SettingsV1alpha1Interface
	// Deprecated: please explicitly pick a version if possible.
	Settings() settingsv1alpha1.SettingsV1alpha1Interface
	StorageV1beta1() storagev1beta1.StorageV1beta1Interface
	StorageV1() storagev1.StorageV1Interface
	// Deprecated: please explicitly pick a version if possible.
	Storage() storagev1.StorageV1Interface
}

// Clientset contains the clients for groups. Each group has exactly one
// version included in a Clientset.
type Clientset struct {
	*discovery.DiscoveryClient
	*admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Client
	*appsv1beta1.AppsV1beta1Client
	*appsv1beta2.AppsV1beta2Client
	*authenticationv1.AuthenticationV1Client
	*authenticationv1beta1.AuthenticationV1beta1Client
	*authorizationv1.AuthorizationV1Client
	*authorizationv1beta1.AuthorizationV1beta1Client
	*autoscalingv1.AutoscalingV1Client
	*autoscalingv2alpha1.AutoscalingV2alpha1Client
	*batchv1.BatchV1Client
	*batchv2alpha1.BatchV2alpha1Client
	*certificatesv1beta1.CertificatesV1beta1Client
	*corev1.CoreV1Client
	*extensionsv1beta1.ExtensionsV1beta1Client
	*networkingv1.NetworkingV1Client
	*policyv1beta1.PolicyV1beta1Client
	*rbacv1beta1.RbacV1beta1Client
	*rbacv1alpha1.RbacV1alpha1Client
	*schedulingv1alpha1.SchedulingV1alpha1Client
	*settingsv1alpha1.SettingsV1alpha1Client
	*storagev1beta1.StorageV1beta1Client
	*storagev1.StorageV1Client
}

// AdmissionregistrationV1alpha1 retrieves the AdmissionregistrationV1alpha1Client
func (c *Clientset) AdmissionregistrationV1alpha1() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AdmissionregistrationV1alpha1Client
}

// Deprecated: Admissionregistration retrieves the default version of AdmissionregistrationClient.
// Please explicitly pick a version.
func (c *Clientset) Admissionregistration() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AdmissionregistrationV1alpha1Client
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() appsv1beta1.AppsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta1Client
}

// AppsV1beta2 retrieves the AppsV1beta2Client
func (c *Clientset) AppsV1beta2() appsv1beta2.AppsV1beta2Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta2Client
}

// Deprecated: Apps retrieves the default version of AppsClient.
// Please explicitly pick a version.
func (c *Clientset) Apps() appsv1beta2.AppsV1beta2Interface {
	if c == nil {
		return nil
	}
	return c.AppsV1beta2Client
}

// AuthenticationV1 retrieves the AuthenticationV1Client
func (c *Clientset) AuthenticationV1() authenticationv1.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// Deprecated: Authentication retrieves the default version of AuthenticationClient.
// Please explicitly pick a version.
func (c *Clientset) Authentication() authenticationv1.AuthenticationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1Client
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() authenticationv1beta1.AuthenticationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthenticationV1beta1Client
}

// AuthorizationV1 retrieves the AuthorizationV1Client
func (c *Clientset) AuthorizationV1() authorizationv1.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// Deprecated: Authorization retrieves the default version of AuthorizationClient.
// Please explicitly pick a version.
func (c *Clientset) Authorization() authorizationv1.AuthorizationV1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1Client
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() authorizationv1beta1.AuthorizationV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.AuthorizationV1beta1Client
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() autoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// Deprecated: Autoscaling retrieves the default version of AutoscalingClient.
// Please explicitly pick a version.
func (c *Clientset) Autoscaling() autoscalingv1.AutoscalingV1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV1Client
}

// AutoscalingV2alpha1 retrieves the AutoscalingV2alpha1Client
func (c *Clientset) AutoscalingV2alpha1() autoscalingv2alpha1.AutoscalingV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.AutoscalingV2alpha1Client
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() batchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// Deprecated: Batch retrieves the default version of BatchClient.
// Please explicitly pick a version.
func (c *Clientset) Batch() batchv1.BatchV1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV1Client
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() batchv2alpha1.BatchV2alpha1Interface {
	if c == nil {
		return nil
	}
	return c.BatchV2alpha1Client
}

// CertificatesV1beta1 retrieves the CertificatesV1beta1Client
func (c *Clientset) CertificatesV1beta1() certificatesv1beta1.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// Deprecated: Certificates retrieves the default version of CertificatesClient.
// Please explicitly pick a version.
func (c *Clientset) Certificates() certificatesv1beta1.CertificatesV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.CertificatesV1beta1Client
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() corev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// Deprecated: Core retrieves the default version of CoreClient.
// Please explicitly pick a version.
func (c *Clientset) Core() corev1.CoreV1Interface {
	if c == nil {
		return nil
	}
	return c.CoreV1Client
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// Deprecated: Extensions retrieves the default version of ExtensionsClient.
// Please explicitly pick a version.
func (c *Clientset) Extensions() extensionsv1beta1.ExtensionsV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.ExtensionsV1beta1Client
}

// NetworkingV1 retrieves the NetworkingV1Client
func (c *Clientset) NetworkingV1() networkingv1.NetworkingV1Interface {
	if c == nil {
		return nil
	}
	return c.NetworkingV1Client
}

// Deprecated: Networking retrieves the default version of NetworkingClient.
// Please explicitly pick a version.
func (c *Clientset) Networking() networkingv1.NetworkingV1Interface {
	if c == nil {
		return nil
	}
	return c.NetworkingV1Client
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() policyv1beta1.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// Deprecated: Policy retrieves the default version of PolicyClient.
// Please explicitly pick a version.
func (c *Clientset) Policy() policyv1beta1.PolicyV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.PolicyV1beta1Client
}

// RbacV1beta1 retrieves the RbacV1beta1Client
func (c *Clientset) RbacV1beta1() rbacv1beta1.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// Deprecated: Rbac retrieves the default version of RbacClient.
// Please explicitly pick a version.
func (c *Clientset) Rbac() rbacv1beta1.RbacV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1beta1Client
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() rbacv1alpha1.RbacV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.RbacV1alpha1Client
}

// SchedulingV1alpha1 retrieves the SchedulingV1alpha1Client
func (c *Clientset) SchedulingV1alpha1() schedulingv1alpha1.SchedulingV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.SchedulingV1alpha1Client
}

// Deprecated: Scheduling retrieves the default version of SchedulingClient.
// Please explicitly pick a version.
func (c *Clientset) Scheduling() schedulingv1alpha1.SchedulingV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.SchedulingV1alpha1Client
}

// SettingsV1alpha1 retrieves the SettingsV1alpha1Client
func (c *Clientset) SettingsV1alpha1() settingsv1alpha1.SettingsV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.SettingsV1alpha1Client
}

// Deprecated: Settings retrieves the default version of SettingsClient.
// Please explicitly pick a version.
func (c *Clientset) Settings() settingsv1alpha1.SettingsV1alpha1Interface {
	if c == nil {
		return nil
	}
	return c.SettingsV1alpha1Client
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() storagev1beta1.StorageV1beta1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1beta1Client
}

// StorageV1 retrieves the StorageV1Client
func (c *Clientset) StorageV1() storagev1.StorageV1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1Client
}

// Deprecated: Storage retrieves the default version of StorageClient.
// Please explicitly pick a version.
func (c *Clientset) Storage() storagev1.StorageV1Interface {
	if c == nil {
		return nil
	}
	return c.StorageV1Client
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
	cs.AdmissionregistrationV1alpha1Client, err = admissionregistrationv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AppsV1beta1Client, err = appsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AppsV1beta2Client, err = appsv1beta2.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1Client, err = authenticationv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthenticationV1beta1Client, err = authenticationv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1Client, err = authorizationv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AuthorizationV1beta1Client, err = authorizationv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV1Client, err = autoscalingv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.AutoscalingV2alpha1Client, err = autoscalingv2alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV1Client, err = batchv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.BatchV2alpha1Client, err = batchv2alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.CertificatesV1beta1Client, err = certificatesv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.CoreV1Client, err = corev1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.ExtensionsV1beta1Client, err = extensionsv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.NetworkingV1Client, err = networkingv1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.PolicyV1beta1Client, err = policyv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1beta1Client, err = rbacv1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.RbacV1alpha1Client, err = rbacv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.SchedulingV1alpha1Client, err = schedulingv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.SettingsV1alpha1Client, err = settingsv1alpha1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageV1beta1Client, err = storagev1beta1.NewForConfig(&configShallowCopy)
	if err != nil {
		return nil, err
	}
	cs.StorageV1Client, err = storagev1.NewForConfig(&configShallowCopy)
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
	cs.AdmissionregistrationV1alpha1Client = admissionregistrationv1alpha1.NewForConfigOrDie(c)
	cs.AppsV1beta1Client = appsv1beta1.NewForConfigOrDie(c)
	cs.AppsV1beta2Client = appsv1beta2.NewForConfigOrDie(c)
	cs.AuthenticationV1Client = authenticationv1.NewForConfigOrDie(c)
	cs.AuthenticationV1beta1Client = authenticationv1beta1.NewForConfigOrDie(c)
	cs.AuthorizationV1Client = authorizationv1.NewForConfigOrDie(c)
	cs.AuthorizationV1beta1Client = authorizationv1beta1.NewForConfigOrDie(c)
	cs.AutoscalingV1Client = autoscalingv1.NewForConfigOrDie(c)
	cs.AutoscalingV2alpha1Client = autoscalingv2alpha1.NewForConfigOrDie(c)
	cs.BatchV1Client = batchv1.NewForConfigOrDie(c)
	cs.BatchV2alpha1Client = batchv2alpha1.NewForConfigOrDie(c)
	cs.CertificatesV1beta1Client = certificatesv1beta1.NewForConfigOrDie(c)
	cs.CoreV1Client = corev1.NewForConfigOrDie(c)
	cs.ExtensionsV1beta1Client = extensionsv1beta1.NewForConfigOrDie(c)
	cs.NetworkingV1Client = networkingv1.NewForConfigOrDie(c)
	cs.PolicyV1beta1Client = policyv1beta1.NewForConfigOrDie(c)
	cs.RbacV1beta1Client = rbacv1beta1.NewForConfigOrDie(c)
	cs.RbacV1alpha1Client = rbacv1alpha1.NewForConfigOrDie(c)
	cs.SchedulingV1alpha1Client = schedulingv1alpha1.NewForConfigOrDie(c)
	cs.SettingsV1alpha1Client = settingsv1alpha1.NewForConfigOrDie(c)
	cs.StorageV1beta1Client = storagev1beta1.NewForConfigOrDie(c)
	cs.StorageV1Client = storagev1.NewForConfigOrDie(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClientForConfigOrDie(c)
	return &cs
}

// New creates a new Clientset for the given RESTClient.
func New(c rest.Interface) *Clientset {
	var cs Clientset
	cs.AdmissionregistrationV1alpha1Client = admissionregistrationv1alpha1.New(c)
	cs.AppsV1beta1Client = appsv1beta1.New(c)
	cs.AppsV1beta2Client = appsv1beta2.New(c)
	cs.AuthenticationV1Client = authenticationv1.New(c)
	cs.AuthenticationV1beta1Client = authenticationv1beta1.New(c)
	cs.AuthorizationV1Client = authorizationv1.New(c)
	cs.AuthorizationV1beta1Client = authorizationv1beta1.New(c)
	cs.AutoscalingV1Client = autoscalingv1.New(c)
	cs.AutoscalingV2alpha1Client = autoscalingv2alpha1.New(c)
	cs.BatchV1Client = batchv1.New(c)
	cs.BatchV2alpha1Client = batchv2alpha1.New(c)
	cs.CertificatesV1beta1Client = certificatesv1beta1.New(c)
	cs.CoreV1Client = corev1.New(c)
	cs.ExtensionsV1beta1Client = extensionsv1beta1.New(c)
	cs.NetworkingV1Client = networkingv1.New(c)
	cs.PolicyV1beta1Client = policyv1beta1.New(c)
	cs.RbacV1beta1Client = rbacv1beta1.New(c)
	cs.RbacV1alpha1Client = rbacv1alpha1.New(c)
	cs.SchedulingV1alpha1Client = schedulingv1alpha1.New(c)
	cs.SettingsV1alpha1Client = settingsv1alpha1.New(c)
	cs.StorageV1beta1Client = storagev1beta1.New(c)
	cs.StorageV1Client = storagev1.New(c)

	cs.DiscoveryClient = discovery.NewDiscoveryClient(c)
	return &cs
}
