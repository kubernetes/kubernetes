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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	internalversionapps "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	fakeinternalversionapps "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion/fake"
	internalversionauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	fakeinternalversionauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion/fake"
	internalversionauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	fakeinternalversionauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion/fake"
	internalversionautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	fakeinternalversionautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion/fake"
	internalversionbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	fakeinternalversionbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion/fake"
	internalversioncertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	fakeinternalversioncertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion/fake"
	internalversioncore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	fakeinternalversioncore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion/fake"
	internalversionextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	fakeinternalversionextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion/fake"
	internalversionpolicy "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	fakeinternalversionpolicy "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion/fake"
	internalversionrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	fakeinternalversionrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion/fake"
	internalversionstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion"
	fakeinternalversionstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion/fake"
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

// CoreInternalVersion retrieves the CoreInternalVersionClient
func (c *Clientset) CoreInternalVersion() internalversioncore.CoreInternalVersionInterface {
	return &fakeinternalversioncore.FakeCoreInternalVersion{Fake: &c.Fake}
}

// Core retrieves the CoreInternalVersionClient
func (c *Clientset) Core() internalversioncore.CoreInternalVersionInterface {
	return &fakeinternalversioncore.FakeCoreInternalVersion{Fake: &c.Fake}
}

// AppsInternalVersion retrieves the AppsInternalVersionClient
func (c *Clientset) AppsInternalVersion() internalversionapps.AppsInternalVersionInterface {
	return &fakeinternalversionapps.FakeAppsInternalVersion{Fake: &c.Fake}
}

// Apps retrieves the AppsInternalVersionClient
func (c *Clientset) Apps() internalversionapps.AppsInternalVersionInterface {
	return &fakeinternalversionapps.FakeAppsInternalVersion{Fake: &c.Fake}
}

// AuthenticationInternalVersion retrieves the AuthenticationInternalVersionClient
func (c *Clientset) AuthenticationInternalVersion() internalversionauthentication.AuthenticationInternalVersionInterface {
	return &fakeinternalversionauthentication.FakeAuthenticationInternalVersion{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationInternalVersionClient
func (c *Clientset) Authentication() internalversionauthentication.AuthenticationInternalVersionInterface {
	return &fakeinternalversionauthentication.FakeAuthenticationInternalVersion{Fake: &c.Fake}
}

// AuthorizationInternalVersion retrieves the AuthorizationInternalVersionClient
func (c *Clientset) AuthorizationInternalVersion() internalversionauthorization.AuthorizationInternalVersionInterface {
	return &fakeinternalversionauthorization.FakeAuthorizationInternalVersion{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationInternalVersionClient
func (c *Clientset) Authorization() internalversionauthorization.AuthorizationInternalVersionInterface {
	return &fakeinternalversionauthorization.FakeAuthorizationInternalVersion{Fake: &c.Fake}
}

// AutoscalingInternalVersion retrieves the AutoscalingInternalVersionClient
func (c *Clientset) AutoscalingInternalVersion() internalversionautoscaling.AutoscalingInternalVersionInterface {
	return &fakeinternalversionautoscaling.FakeAutoscalingInternalVersion{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingInternalVersionClient
func (c *Clientset) Autoscaling() internalversionautoscaling.AutoscalingInternalVersionInterface {
	return &fakeinternalversionautoscaling.FakeAutoscalingInternalVersion{Fake: &c.Fake}
}

// BatchInternalVersion retrieves the BatchInternalVersionClient
func (c *Clientset) BatchInternalVersion() internalversionbatch.BatchInternalVersionInterface {
	return &fakeinternalversionbatch.FakeBatchInternalVersion{Fake: &c.Fake}
}

// Batch retrieves the BatchInternalVersionClient
func (c *Clientset) Batch() internalversionbatch.BatchInternalVersionInterface {
	return &fakeinternalversionbatch.FakeBatchInternalVersion{Fake: &c.Fake}
}

// CertificatesInternalVersion retrieves the CertificatesInternalVersionClient
func (c *Clientset) CertificatesInternalVersion() internalversioncertificates.CertificatesInternalVersionInterface {
	return &fakeinternalversioncertificates.FakeCertificatesInternalVersion{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesInternalVersionClient
func (c *Clientset) Certificates() internalversioncertificates.CertificatesInternalVersionInterface {
	return &fakeinternalversioncertificates.FakeCertificatesInternalVersion{Fake: &c.Fake}
}

// ExtensionsInternalVersion retrieves the ExtensionsInternalVersionClient
func (c *Clientset) ExtensionsInternalVersion() internalversionextensions.ExtensionsInternalVersionInterface {
	return &fakeinternalversionextensions.FakeExtensionsInternalVersion{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsInternalVersionClient
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalVersionInterface {
	return &fakeinternalversionextensions.FakeExtensionsInternalVersion{Fake: &c.Fake}
}

// PolicyInternalVersion retrieves the PolicyInternalVersionClient
func (c *Clientset) PolicyInternalVersion() internalversionpolicy.PolicyInternalVersionInterface {
	return &fakeinternalversionpolicy.FakePolicyInternalVersion{Fake: &c.Fake}
}

// Policy retrieves the PolicyInternalVersionClient
func (c *Clientset) Policy() internalversionpolicy.PolicyInternalVersionInterface {
	return &fakeinternalversionpolicy.FakePolicyInternalVersion{Fake: &c.Fake}
}

// RbacInternalVersion retrieves the RbacInternalVersionClient
func (c *Clientset) RbacInternalVersion() internalversionrbac.RbacInternalVersionInterface {
	return &fakeinternalversionrbac.FakeRbacInternalVersion{Fake: &c.Fake}
}

// Rbac retrieves the RbacInternalVersionClient
func (c *Clientset) Rbac() internalversionrbac.RbacInternalVersionInterface {
	return &fakeinternalversionrbac.FakeRbacInternalVersion{Fake: &c.Fake}
}

// StorageInternalVersion retrieves the StorageInternalVersionClient
func (c *Clientset) StorageInternalVersion() internalversionstorage.StorageInternalVersionInterface {
	return &fakeinternalversionstorage.FakeStorageInternalVersion{Fake: &c.Fake}
}

// Storage retrieves the StorageInternalVersionClient
func (c *Clientset) Storage() internalversionstorage.StorageInternalVersionInterface {
	return &fakeinternalversionstorage.FakeStorageInternalVersion{Fake: &c.Fake}
}
