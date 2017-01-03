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
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	v1core "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/core/v1"
	fakev1core "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/core/v1/fake"
	v1beta1extensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/extensions/v1beta1"
	fakev1beta1extensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/extensions/v1beta1/fake"
	v1beta1federation "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/federation/v1beta1"
	fakev1beta1federation "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/typed/federation/v1beta1/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
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

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() v1beta1extensions.ExtensionsV1beta1Interface {
	return &fakev1beta1extensions.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() v1beta1extensions.ExtensionsV1beta1Interface {
	return &fakev1beta1extensions.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// FederationV1beta1 retrieves the FederationV1beta1Client
func (c *Clientset) FederationV1beta1() v1beta1federation.FederationV1beta1Interface {
	return &fakev1beta1federation.FakeFederationV1beta1{Fake: &c.Fake}
}

// Federation retrieves the FederationV1beta1Client
func (c *Clientset) Federation() v1beta1federation.FederationV1beta1Interface {
	return &fakev1beta1federation.FakeFederationV1beta1{Fake: &c.Fake}
}
