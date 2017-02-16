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
	runtime "k8s.io/apimachinery/pkg/runtime"
	serializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	appsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1"
	fakeappsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/apps/v1beta1/fake"
	authenticationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1"
	fakeauthenticationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1/fake"
	authenticationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1beta1"
	fakeauthenticationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authentication/v1beta1/fake"
	authorizationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1"
	fakeauthorizationv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1/fake"
	authorizationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1beta1"
	fakeauthorizationv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/authorization/v1beta1/fake"
	autoscalingv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v1"
	fakeautoscalingv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v1/fake"
	autoscalingv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1"
	fakeautoscalingv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1/fake"
	batchv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v1"
	fakebatchv1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v1/fake"
	batchv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v2alpha1"
	fakebatchv2alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/batch/v2alpha1/fake"
	certificatesv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
	fakecertificatesv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1/fake"
	corev1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	fakecorev1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1/fake"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	fakeextensionsv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1/fake"
	policyv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/policy/v1beta1"
	fakepolicyv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/policy/v1beta1/fake"
	rbacv1alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1"
	fakerbacv1alpha1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1alpha1/fake"
	rbacv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1beta1"
	fakerbacv1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/rbac/v1beta1/fake"
	storagev1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/storage/v1beta1"
	fakestoragev1beta1 "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/storage/v1beta1/fake"
)

var Scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)
var parameterCodec = runtime.NewParameterCodec(scheme)

func init() {
	corev1.AddToScheme(Scheme)
	appsv1beta1.AddToScheme(Scheme)
	authenticationv1.AddToScheme(Scheme)
	authenticationv1beta1.AddToScheme(Scheme)
	authorizationv1.AddToScheme(Scheme)
	authorizationv1beta1.AddToScheme(Scheme)
	autoscalingv1.AddToScheme(Scheme)
	autoscalingv2alpha1.AddToScheme(Scheme)
	batchv1.AddToScheme(Scheme)
	batchv2alpha1.AddToScheme(Scheme)
	certificatesv1beta1.AddToScheme(Scheme)
	extensionsv1beta1.AddToScheme(Scheme)
	policyv1beta1.AddToScheme(Scheme)
	rbacv1beta1.AddToScheme(Scheme)
	rbacv1alpha1.AddToScheme(Scheme)
	storagev1beta1.AddToScheme(Scheme)

}

// NewSimpleClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := testing.NewObjectTracker(api.Registry, api.Scheme, api.Codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakePtr := testing.Fake{}
	fakePtr.AddReactor("*", "*", testing.ObjectReaction(o, api.Registry.RESTMapper()))

	fakePtr.AddWatchReactor("*", testing.DefaultWatchReactor(watch.NewFake(), nil))

	return &Clientset{fakePtr}
}

// Clientset implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type Clientset struct {
	testing.Fake
}

func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return &fakediscovery.FakeDiscovery{Fake: &c.Fake}
}

var _ clientset.Interface = &Clientset{}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() clientcorev1.CoreV1Interface {
	return &fakeclientcorev1.FakeCoreV1{Fake: &c.Fake}
}

// Core retrieves the CoreV1Client
func (c *Clientset) Core() clientcorev1.CoreV1Interface {
	return &fakeclientcorev1.FakeCoreV1{Fake: &c.Fake}
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() clientappsv1beta1.AppsV1beta1Interface {
	return &fakeclientappsv1beta1.FakeAppsV1beta1{Fake: &c.Fake}
}

// Apps retrieves the AppsV1beta1Client
func (c *Clientset) Apps() clientappsv1beta1.AppsV1beta1Interface {
	return &fakeclientappsv1beta1.FakeAppsV1beta1{Fake: &c.Fake}
}

// AuthenticationV1 retrieves the AuthenticationV1Client
func (c *Clientset) AuthenticationV1() clientauthenticationv1.AuthenticationV1Interface {
	return &fakeclientauthenticationv1.FakeAuthenticationV1{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationV1Client
func (c *Clientset) Authentication() clientauthenticationv1.AuthenticationV1Interface {
	return &fakeclientauthenticationv1.FakeAuthenticationV1{Fake: &c.Fake}
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() clientauthenticationv1beta1.AuthenticationV1beta1Interface {
	return &fakeclientauthenticationv1beta1.FakeAuthenticationV1beta1{Fake: &c.Fake}
}

// AuthorizationV1 retrieves the AuthorizationV1Client
func (c *Clientset) AuthorizationV1() clientauthorizationv1.AuthorizationV1Interface {
	return &fakeclientauthorizationv1.FakeAuthorizationV1{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationV1Client
func (c *Clientset) Authorization() clientauthorizationv1.AuthorizationV1Interface {
	return &fakeclientauthorizationv1.FakeAuthorizationV1{Fake: &c.Fake}
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() clientauthorizationv1beta1.AuthorizationV1beta1Interface {
	return &fakeclientauthorizationv1beta1.FakeAuthorizationV1beta1{Fake: &c.Fake}
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface {
	return &fakeclientautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingV1Client
func (c *Clientset) Autoscaling() clientautoscalingv1.AutoscalingV1Interface {
	return &fakeclientautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// AutoscalingV2alpha1 retrieves the AutoscalingV2alpha1Client
func (c *Clientset) AutoscalingV2alpha1() clientautoscalingv2alpha1.AutoscalingV2alpha1Interface {
	return &fakeclientautoscalingv2alpha1.FakeAutoscalingV2alpha1{Fake: &c.Fake}
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() clientbatchv1.BatchV1Interface {
	return &fakeclientbatchv1.FakeBatchV1{Fake: &c.Fake}
}

// Batch retrieves the BatchV1Client
func (c *Clientset) Batch() clientbatchv1.BatchV1Interface {
	return &fakeclientbatchv1.FakeBatchV1{Fake: &c.Fake}
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() clientbatchv2alpha1.BatchV2alpha1Interface {
	return &fakeclientbatchv2alpha1.FakeBatchV2alpha1{Fake: &c.Fake}
}

// CertificatesV1beta1 retrieves the CertificatesV1beta1Client
func (c *Clientset) CertificatesV1beta1() clientcertificatesv1beta1.CertificatesV1beta1Interface {
	return &fakeclientcertificatesv1beta1.FakeCertificatesV1beta1{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesV1beta1Client
func (c *Clientset) Certificates() clientcertificatesv1beta1.CertificatesV1beta1Interface {
	return &fakeclientcertificatesv1beta1.FakeCertificatesV1beta1{Fake: &c.Fake}
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeclientextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeclientextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() clientpolicyv1beta1.PolicyV1beta1Interface {
	return &fakeclientpolicyv1beta1.FakePolicyV1beta1{Fake: &c.Fake}
}

// Policy retrieves the PolicyV1beta1Client
func (c *Clientset) Policy() clientpolicyv1beta1.PolicyV1beta1Interface {
	return &fakeclientpolicyv1beta1.FakePolicyV1beta1{Fake: &c.Fake}
}

// RbacV1beta1 retrieves the RbacV1beta1Client
func (c *Clientset) RbacV1beta1() clientrbacv1beta1.RbacV1beta1Interface {
	return &fakeclientrbacv1beta1.FakeRbacV1beta1{Fake: &c.Fake}
}

// Rbac retrieves the RbacV1beta1Client
func (c *Clientset) Rbac() clientrbacv1beta1.RbacV1beta1Interface {
	return &fakeclientrbacv1beta1.FakeRbacV1beta1{Fake: &c.Fake}
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() clientrbacv1alpha1.RbacV1alpha1Interface {
	return &fakeclientrbacv1alpha1.FakeRbacV1alpha1{Fake: &c.Fake}
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() clientstoragev1beta1.StorageV1beta1Interface {
	return &fakeclientstoragev1beta1.FakeStorageV1beta1{Fake: &c.Fake}
}

// Storage retrieves the StorageV1beta1Client
func (c *Clientset) Storage() clientstoragev1beta1.StorageV1beta1Interface {
	return &fakeclientstoragev1beta1.FakeStorageV1beta1{Fake: &c.Fake}
}
