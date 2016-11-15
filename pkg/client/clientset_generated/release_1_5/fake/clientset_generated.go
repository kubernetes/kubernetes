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

package fake

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	v1beta1apps "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1beta1"
	fakev1beta1apps "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/apps/v1beta1/fake"
	v1beta1authentication "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1"
	fakev1beta1authentication "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authentication/v1beta1/fake"
	v1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authorization/v1beta1"
	fakev1beta1authorization "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/authorization/v1beta1/fake"
	v1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1"
	fakev1autoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/autoscaling/v1/fake"
	v1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v1"
	fakev1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v1/fake"
	v2alpha1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v2alpha1"
	fakev2alpha1batch "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/batch/v2alpha1/fake"
	v1alpha1certificates "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1"
	fakev1alpha1certificates "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/certificates/v1alpha1/fake"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	fakev1core "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1/fake"
	v1beta1extensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1"
	fakev1beta1extensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1/fake"
	v1beta1policy "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1beta1"
	fakev1beta1policy "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/policy/v1beta1/fake"
	v1alpha1rbac "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/rbac/v1alpha1"
	fakev1alpha1rbac "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/rbac/v1alpha1/fake"
	v1beta1storage "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/storage/v1beta1"
	fakev1beta1storage "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/storage/v1beta1/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	fakediscovery "k8s.io/kubernetes/pkg/client/typed/discovery/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// NewSimpleClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := core.NewObjectTracker(api.Scheme, api.Codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakePtr := core.Fake{}
	fakePtr.AddReactor("*", "*", core.ObjectReaction(o, registered.RESTMapper()))

	fakePtr.AddWatchReactor("*", core.DefaultWatchReactor(watch.NewFake(), nil))

	return &Clientset{fakePtr}
}

// Clientset implements clientset.Interface. Meant to be embedded into a
// struct to get a default implementation. This makes faking out just the method
// you want to test easier.
type Clientset struct {
	core.Fake
}

func (c *Clientset) Discovery() discovery.DiscoveryInterface {
	return &fakediscovery.FakeDiscovery{Fake: &c.Fake}
}

var _ clientset.Interface = &Clientset{}

// CoreV1 retrieves the CoreV1Client
func (c *Clientset) CoreV1() v1core.CoreV1Interface {
	return &fakev1core.FakeCoreV1{Fake: &c.Fake}
}

// Core retrieves the CoreV1Client
func (c *Clientset) Core() v1core.CoreV1Interface {
	return &fakev1core.FakeCoreV1{Fake: &c.Fake}
}

// AppsV1beta1 retrieves the AppsV1beta1Client
func (c *Clientset) AppsV1beta1() v1beta1apps.AppsV1beta1Interface {
	return &fakev1beta1apps.FakeAppsV1beta1{Fake: &c.Fake}
}

// Apps retrieves the AppsV1beta1Client
func (c *Clientset) Apps() v1beta1apps.AppsV1beta1Interface {
	return &fakev1beta1apps.FakeAppsV1beta1{Fake: &c.Fake}
}

// AuthenticationV1beta1 retrieves the AuthenticationV1beta1Client
func (c *Clientset) AuthenticationV1beta1() v1beta1authentication.AuthenticationV1beta1Interface {
	return &fakev1beta1authentication.FakeAuthenticationV1beta1{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationV1beta1Client
func (c *Clientset) Authentication() v1beta1authentication.AuthenticationV1beta1Interface {
	return &fakev1beta1authentication.FakeAuthenticationV1beta1{Fake: &c.Fake}
}

// AuthorizationV1beta1 retrieves the AuthorizationV1beta1Client
func (c *Clientset) AuthorizationV1beta1() v1beta1authorization.AuthorizationV1beta1Interface {
	return &fakev1beta1authorization.FakeAuthorizationV1beta1{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationV1beta1Client
func (c *Clientset) Authorization() v1beta1authorization.AuthorizationV1beta1Interface {
	return &fakev1beta1authorization.FakeAuthorizationV1beta1{Fake: &c.Fake}
}

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() v1autoscaling.AutoscalingV1Interface {
	return &fakev1autoscaling.FakeAutoscalingV1{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingV1Client
func (c *Clientset) Autoscaling() v1autoscaling.AutoscalingV1Interface {
	return &fakev1autoscaling.FakeAutoscalingV1{Fake: &c.Fake}
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() v1batch.BatchV1Interface {
	return &fakev1batch.FakeBatchV1{Fake: &c.Fake}
}

// Batch retrieves the BatchV1Client
func (c *Clientset) Batch() v1batch.BatchV1Interface {
	return &fakev1batch.FakeBatchV1{Fake: &c.Fake}
}

// BatchV2alpha1 retrieves the BatchV2alpha1Client
func (c *Clientset) BatchV2alpha1() v2alpha1batch.BatchV2alpha1Interface {
	return &fakev2alpha1batch.FakeBatchV2alpha1{Fake: &c.Fake}
}

// CertificatesV1alpha1 retrieves the CertificatesV1alpha1Client
func (c *Clientset) CertificatesV1alpha1() v1alpha1certificates.CertificatesV1alpha1Interface {
	return &fakev1alpha1certificates.FakeCertificatesV1alpha1{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesV1alpha1Client
func (c *Clientset) Certificates() v1alpha1certificates.CertificatesV1alpha1Interface {
	return &fakev1alpha1certificates.FakeCertificatesV1alpha1{Fake: &c.Fake}
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface {
	return &fakev1beta1extensions.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsV1beta1Interface {
	return &fakev1beta1extensions.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// PolicyV1beta1 retrieves the PolicyV1beta1Client
func (c *Clientset) PolicyV1beta1() v1beta1policy.PolicyV1beta1Interface {
	return &fakev1beta1policy.FakePolicyV1beta1{Fake: &c.Fake}
}

// Policy retrieves the PolicyV1beta1Client
func (c *Clientset) Policy() v1beta1policy.PolicyV1beta1Interface {
	return &fakev1beta1policy.FakePolicyV1beta1{Fake: &c.Fake}
}

// RbacV1alpha1 retrieves the RbacV1alpha1Client
func (c *Clientset) RbacV1alpha1() v1alpha1rbac.RbacV1alpha1Interface {
	return &fakev1alpha1rbac.FakeRbacV1alpha1{Fake: &c.Fake}
}

// Rbac retrieves the RbacV1alpha1Client
func (c *Clientset) Rbac() v1alpha1rbac.RbacV1alpha1Interface {
	return &fakev1alpha1rbac.FakeRbacV1alpha1{Fake: &c.Fake}
}

// StorageV1beta1 retrieves the StorageV1beta1Client
func (c *Clientset) StorageV1beta1() v1beta1storage.StorageV1beta1Interface {
	return &fakev1beta1storage.FakeStorageV1beta1{Fake: &c.Fake}
}

// Storage retrieves the StorageV1beta1Client
func (c *Clientset) Storage() v1beta1storage.StorageV1beta1Interface {
	return &fakev1beta1storage.FakeStorageV1beta1{Fake: &c.Fake}
}
