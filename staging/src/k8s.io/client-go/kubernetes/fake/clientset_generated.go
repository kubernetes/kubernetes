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

package fake

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	kubernetes "k8s.io/client-go/kubernetes"
	admissionregistrationv1alpha1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1alpha1"
	fakeadmissionregistrationv1alpha1 "k8s.io/client-go/kubernetes/typed/admissionregistration/v1alpha1/fake"
	appsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1"
	fakeappsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1/fake"
	authenticationv1 "k8s.io/client-go/kubernetes/typed/authentication/v1"
	fakeauthenticationv1 "k8s.io/client-go/kubernetes/typed/authentication/v1/fake"
	authenticationv1beta1 "k8s.io/client-go/kubernetes/typed/authentication/v1beta1"
	fakeauthenticationv1beta1 "k8s.io/client-go/kubernetes/typed/authentication/v1beta1/fake"
	authorizationv1 "k8s.io/client-go/kubernetes/typed/authorization/v1"
	fakeauthorizationv1 "k8s.io/client-go/kubernetes/typed/authorization/v1/fake"
	authorizationv1beta1 "k8s.io/client-go/kubernetes/typed/authorization/v1beta1"
	fakeauthorizationv1beta1 "k8s.io/client-go/kubernetes/typed/authorization/v1beta1/fake"
	autoscalingv1 "k8s.io/client-go/kubernetes/typed/autoscaling/v1"
	fakeautoscalingv1 "k8s.io/client-go/kubernetes/typed/autoscaling/v1/fake"
	autoscalingv2alpha1 "k8s.io/client-go/kubernetes/typed/autoscaling/v2alpha1"
	fakeautoscalingv2alpha1 "k8s.io/client-go/kubernetes/typed/autoscaling/v2alpha1/fake"
	batchv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	fakebatchv1 "k8s.io/client-go/kubernetes/typed/batch/v1/fake"
	batchv2alpha1 "k8s.io/client-go/kubernetes/typed/batch/v2alpha1"
	fakebatchv2alpha1 "k8s.io/client-go/kubernetes/typed/batch/v2alpha1/fake"
	certificatesv1beta1 "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	fakecertificatesv1beta1 "k8s.io/client-go/kubernetes/typed/certificates/v1beta1/fake"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	fakecorev1 "k8s.io/client-go/kubernetes/typed/core/v1/fake"
	extensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	fakeextensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1/fake"
	networkingv1 "k8s.io/client-go/kubernetes/typed/networking/v1"
	fakenetworkingv1 "k8s.io/client-go/kubernetes/typed/networking/v1/fake"
	policyv1beta1 "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	fakepolicyv1beta1 "k8s.io/client-go/kubernetes/typed/policy/v1beta1/fake"
	rbacv1alpha1 "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1"
	fakerbacv1alpha1 "k8s.io/client-go/kubernetes/typed/rbac/v1alpha1/fake"
	rbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1"
	fakerbacv1beta1 "k8s.io/client-go/kubernetes/typed/rbac/v1beta1/fake"
	settingsv1alpha1 "k8s.io/client-go/kubernetes/typed/settings/v1alpha1"
	fakesettingsv1alpha1 "k8s.io/client-go/kubernetes/typed/settings/v1alpha1/fake"
	storagev1 "k8s.io/client-go/kubernetes/typed/storage/v1"
	fakestoragev1 "k8s.io/client-go/kubernetes/typed/storage/v1/fake"
	storagev1beta1 "k8s.io/client-go/kubernetes/typed/storage/v1beta1"
	fakestoragev1beta1 "k8s.io/client-go/kubernetes/typed/storage/v1beta1/fake"
	"k8s.io/client-go/testing"
)

// NewSimpleClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := testing.NewObjectTracker(scheme, codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakePtr := testing.Fake{}
	fakePtr.AddReactor("*", "*", testing.ObjectReaction(o))

	fakePtr.AddWatchReactor("*", testing.DefaultWatchReactor(watch.NewFake(), nil))

	return &Clientset{fakePtr}
}

// Clientset implements kubernetes.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type Clientset struct {
	testing.Fake
}

func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return &fakediscovery.FakeDiscovery{Fake: &c.Fake}
}

var _ kubernetes.Interface = &Clientset{}

// AdmissionregistrationV1alpha1 retrieves the AdmissionregistrationV1alpha1Client
func (c *Clientset) AdmissionregistrationV1alpha1() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface {
	return &fakeadmissionregistrationv1alpha1.FakeAdmissionregistrationV1alpha1{Fake: &c.Fake}
}

// Admissionregistration retrieves the AdmissionregistrationV1alpha1Client
func (c *Clientset) Admissionregistration() admissionregistrationv1alpha1.AdmissionregistrationV1alpha1Interface {
	return &fakeadmissionregistrationv1alpha1.FakeAdmissionregistrationV1alpha1{Fake: &c.Fake}
}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() corev1.CoreV1Interface {
	return &fakecorev1.FakeCoreV1{Fake: &c.Fake}
}

// Core retrieves the CoreV1Client
func (c *Clientset) Core() corev1.CoreV1Interface {
	return &fakecorev1.FakeCoreV1{Fake: &c.Fake}
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() appsv1beta1.AppsV1beta1Interface {
	return &fakeappsv1beta1.FakeAppsV1beta1{Fake: &c.Fake}
}

// Apps retrieves the AppsV1beta1Client
func (c *Clientset) Apps() appsv1beta1.AppsV1beta1Interface {
	return &fakeappsv1beta1.FakeAppsV1beta1{Fake: &c.Fake}
}

// AuthenticationV1 retrieves the AuthenticationV1Client
func (c *Clientset) AuthenticationV1() authenticationv1.AuthenticationV1Interface {
	return &fakeauthenticationv1.FakeAuthenticationV1{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationV1Client
func (c *Clientset) Authentication() authenticationv1.AuthenticationV1Interface {
	return &fakeauthenticationv1.FakeAuthenticationV1{Fake: &c.Fake}
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() authenticationv1beta1.AuthenticationV1beta1Interface {
	return &fakeauthenticationv1beta1.FakeAuthenticationV1beta1{Fake: &c.Fake}
}

// AuthorizationV1 retrieves the AuthorizationV1Client
func (c *Clientset) AuthorizationV1() authorizationv1.AuthorizationV1Interface {
	return &fakeauthorizationv1.FakeAuthorizationV1{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationV1Client
func (c *Clientset) Authorization() authorizationv1.AuthorizationV1Interface {
	return &fakeauthorizationv1.FakeAuthorizationV1{Fake: &c.Fake}
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() authorizationv1beta1.AuthorizationV1beta1Interface {
	return &fakeauthorizationv1beta1.FakeAuthorizationV1beta1{Fake: &c.Fake}
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() autoscalingv1.AutoscalingV1Interface {
	return &fakeautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingV1Client
func (c *Clientset) Autoscaling() autoscalingv1.AutoscalingV1Interface {
	return &fakeautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// AutoscalingV2alpha1 retrieves the AutoscalingV2alpha1Client
func (c *Clientset) AutoscalingV2alpha1() autoscalingv2alpha1.AutoscalingV2alpha1Interface {
	return &fakeautoscalingv2alpha1.FakeAutoscalingV2alpha1{Fake: &c.Fake}
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() batchv1.BatchV1Interface {
	return &fakebatchv1.FakeBatchV1{Fake: &c.Fake}
}

// Batch retrieves the BatchV1Client
func (c *Clientset) Batch() batchv1.BatchV1Interface {
	return &fakebatchv1.FakeBatchV1{Fake: &c.Fake}
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() batchv2alpha1.BatchV2alpha1Interface {
	return &fakebatchv2alpha1.FakeBatchV2alpha1{Fake: &c.Fake}
}

// CertificatesV1beta1 retrieves the CertificatesV1beta1Client
func (c *Clientset) CertificatesV1beta1() certificatesv1beta1.CertificatesV1beta1Interface {
	return &fakecertificatesv1beta1.FakeCertificatesV1beta1{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesV1beta1Client
func (c *Clientset) Certificates() certificatesv1beta1.CertificatesV1beta1Interface {
	return &fakecertificatesv1beta1.FakeCertificatesV1beta1{Fake: &c.Fake}
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() extensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() extensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// NetworkingV1 retrieves the NetworkingV1Client
func (c *Clientset) NetworkingV1() networkingv1.NetworkingV1Interface {
	return &fakenetworkingv1.FakeNetworkingV1{Fake: &c.Fake}
}

// Networking retrieves the NetworkingV1Client
func (c *Clientset) Networking() networkingv1.NetworkingV1Interface {
	return &fakenetworkingv1.FakeNetworkingV1{Fake: &c.Fake}
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() policyv1beta1.PolicyV1beta1Interface {
	return &fakepolicyv1beta1.FakePolicyV1beta1{Fake: &c.Fake}
}

// Policy retrieves the PolicyV1beta1Client
func (c *Clientset) Policy() policyv1beta1.PolicyV1beta1Interface {
	return &fakepolicyv1beta1.FakePolicyV1beta1{Fake: &c.Fake}
}

// RbacV1beta1 retrieves the RbacV1beta1Client
func (c *Clientset) RbacV1beta1() rbacv1beta1.RbacV1beta1Interface {
	return &fakerbacv1beta1.FakeRbacV1beta1{Fake: &c.Fake}
}

// Rbac retrieves the RbacV1beta1Client
func (c *Clientset) Rbac() rbacv1beta1.RbacV1beta1Interface {
	return &fakerbacv1beta1.FakeRbacV1beta1{Fake: &c.Fake}
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() rbacv1alpha1.RbacV1alpha1Interface {
	return &fakerbacv1alpha1.FakeRbacV1alpha1{Fake: &c.Fake}
}

// SettingsV1alpha1 retrieves the SettingsV1alpha1Client
func (c *Clientset) SettingsV1alpha1() settingsv1alpha1.SettingsV1alpha1Interface {
	return &fakesettingsv1alpha1.FakeSettingsV1alpha1{Fake: &c.Fake}
}

// Settings retrieves the SettingsV1alpha1Client
func (c *Clientset) Settings() settingsv1alpha1.SettingsV1alpha1Interface {
	return &fakesettingsv1alpha1.FakeSettingsV1alpha1{Fake: &c.Fake}
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() storagev1beta1.StorageV1beta1Interface {
	return &fakestoragev1beta1.FakeStorageV1beta1{Fake: &c.Fake}
}

// StorageV1 retrieves the StorageV1Client
func (c *Clientset) StorageV1() storagev1.StorageV1Interface {
	return &fakestoragev1.FakeStorageV1{Fake: &c.Fake}
}

// Storage retrieves the StorageV1Client
func (c *Clientset) Storage() storagev1.StorageV1Interface {
	return &fakestoragev1.FakeStorageV1{Fake: &c.Fake}
}
