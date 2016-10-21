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

// CoreInternalversion retrieves the CoreInternalversionClient
func (c *Clientset) CoreInternalversion() internalversioncore.CoreInternalversionInterface {
	return &fakeinternalversioncore.FakeCoreInternalversion{Fake: &c.Fake}
}

// Core retrieves the CoreInternalversionClient
func (c *Clientset) Core() internalversioncore.CoreInternalversionInterface {
	return &fakeinternalversioncore.FakeCoreInternalversion{Fake: &c.Fake}
}

// AppsInternalversion retrieves the AppsInternalversionClient
func (c *Clientset) AppsInternalversion() internalversionapps.AppsInternalversionInterface {
	return &fakeinternalversionapps.FakeAppsInternalversion{Fake: &c.Fake}
}

// Apps retrieves the AppsInternalversionClient
func (c *Clientset) Apps() internalversionapps.AppsInternalversionInterface {
	return &fakeinternalversionapps.FakeAppsInternalversion{Fake: &c.Fake}
}

// AuthenticationInternalversion retrieves the AuthenticationInternalversionClient
func (c *Clientset) AuthenticationInternalversion() internalversionauthentication.AuthenticationInternalversionInterface {
	return &fakeinternalversionauthentication.FakeAuthenticationInternalversion{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationInternalversionClient
func (c *Clientset) Authentication() internalversionauthentication.AuthenticationInternalversionInterface {
	return &fakeinternalversionauthentication.FakeAuthenticationInternalversion{Fake: &c.Fake}
}

// AuthorizationInternalversion retrieves the AuthorizationInternalversionClient
func (c *Clientset) AuthorizationInternalversion() internalversionauthorization.AuthorizationInternalversionInterface {
	return &fakeinternalversionauthorization.FakeAuthorizationInternalversion{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationInternalversionClient
func (c *Clientset) Authorization() internalversionauthorization.AuthorizationInternalversionInterface {
	return &fakeinternalversionauthorization.FakeAuthorizationInternalversion{Fake: &c.Fake}
}

// AutoscalingInternalversion retrieves the AutoscalingInternalversionClient
func (c *Clientset) AutoscalingInternalversion() internalversionautoscaling.AutoscalingInternalversionInterface {
	return &fakeinternalversionautoscaling.FakeAutoscalingInternalversion{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingInternalversionClient
func (c *Clientset) Autoscaling() internalversionautoscaling.AutoscalingInternalversionInterface {
	return &fakeinternalversionautoscaling.FakeAutoscalingInternalversion{Fake: &c.Fake}
}

// BatchInternalversion retrieves the BatchInternalversionClient
func (c *Clientset) BatchInternalversion() internalversionbatch.BatchInternalversionInterface {
	return &fakeinternalversionbatch.FakeBatchInternalversion{Fake: &c.Fake}
}

// Batch retrieves the BatchInternalversionClient
func (c *Clientset) Batch() internalversionbatch.BatchInternalversionInterface {
	return &fakeinternalversionbatch.FakeBatchInternalversion{Fake: &c.Fake}
}

// CertificatesInternalversion retrieves the CertificatesInternalversionClient
func (c *Clientset) CertificatesInternalversion() internalversioncertificates.CertificatesInternalversionInterface {
	return &fakeinternalversioncertificates.FakeCertificatesInternalversion{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesInternalversionClient
func (c *Clientset) Certificates() internalversioncertificates.CertificatesInternalversionInterface {
	return &fakeinternalversioncertificates.FakeCertificatesInternalversion{Fake: &c.Fake}
}

// ExtensionsInternalversion retrieves the ExtensionsInternalversionClient
func (c *Clientset) ExtensionsInternalversion() internalversionextensions.ExtensionsInternalversionInterface {
	return &fakeinternalversionextensions.FakeExtensionsInternalversion{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsInternalversionClient
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInternalversionInterface {
	return &fakeinternalversionextensions.FakeExtensionsInternalversion{Fake: &c.Fake}
}

// PolicyInternalversion retrieves the PolicyInternalversionClient
func (c *Clientset) PolicyInternalversion() internalversionpolicy.PolicyInternalversionInterface {
	return &fakeinternalversionpolicy.FakePolicyInternalversion{Fake: &c.Fake}
}

// Policy retrieves the PolicyInternalversionClient
func (c *Clientset) Policy() internalversionpolicy.PolicyInternalversionInterface {
	return &fakeinternalversionpolicy.FakePolicyInternalversion{Fake: &c.Fake}
}

// RbacInternalversion retrieves the RbacInternalversionClient
func (c *Clientset) RbacInternalversion() internalversionrbac.RbacInternalversionInterface {
	return &fakeinternalversionrbac.FakeRbacInternalversion{Fake: &c.Fake}
}

// Rbac retrieves the RbacInternalversionClient
func (c *Clientset) Rbac() internalversionrbac.RbacInternalversionInterface {
	return &fakeinternalversionrbac.FakeRbacInternalversion{Fake: &c.Fake}
}

// StorageInternalversion retrieves the StorageInternalversionClient
func (c *Clientset) StorageInternalversion() internalversionstorage.StorageInternalversionInterface {
	return &fakeinternalversionstorage.FakeStorageInternalversion{Fake: &c.Fake}
}

// Storage retrieves the StorageInternalversionClient
func (c *Clientset) Storage() internalversionstorage.StorageInternalversionInterface {
	return &fakeinternalversionstorage.FakeStorageInternalversion{Fake: &c.Fake}
}
