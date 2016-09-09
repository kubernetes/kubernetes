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
	unversionedauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned"
	fakeunversionedauthentication "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/unversioned/fake"
	unversionedauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/unversioned"
	fakeunversionedauthorization "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/unversioned/fake"
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned"
	fakeunversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned/fake"
	unversionedbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned"
	fakeunversionedbatch "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/unversioned/fake"
	unversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned"
	fakeunversionedcertificates "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned/fake"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	fakeunversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned/fake"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	fakeunversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned/fake"
	unversionedrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/unversioned"
	fakeunversionedrbac "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/unversioned/fake"
	unversionedstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/unversioned"
	fakeunversionedstorage "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/unversioned/fake"
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

// Core retrieves the CoreClient
func (c *Clientset) Core() unversionedcore.CoreInterface {
	return &fakeunversionedcore.FakeCore{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() unversionedauthentication.AuthenticationInterface {
	return &fakeunversionedauthentication.FakeAuthentication{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationClient
func (c *Clientset) Authorization() unversionedauthorization.AuthorizationInterface {
	return &fakeunversionedauthorization.FakeAuthorization{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() unversionedautoscaling.AutoscalingInterface {
	return &fakeunversionedautoscaling.FakeAutoscaling{Fake: &c.Fake}
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() unversionedbatch.BatchInterface {
	return &fakeunversionedbatch.FakeBatch{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() unversionedcertificates.CertificatesInterface {
	return &fakeunversionedcertificates.FakeCertificates{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() unversionedextensions.ExtensionsInterface {
	return &fakeunversionedextensions.FakeExtensions{Fake: &c.Fake}
}

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() unversionedrbac.RbacInterface {
	return &fakeunversionedrbac.FakeRbac{Fake: &c.Fake}
}

// Storage retrieves the StorageClient
func (c *Clientset) Storage() unversionedstorage.StorageInterface {
	return &fakeunversionedstorage.FakeStorage{Fake: &c.Fake}
}
