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
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	internalversioncore "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	fakeinternalversioncore "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion/fake"
	internalversionextensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion"
	fakeinternalversionextensions "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion/fake"
	internalversionfederation "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion"
	fakeinternalversionfederation "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion/fake"
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

// Core retrieves the CoreClient
func (c *Clientset) Core() internalversioncore.CoreInterface {
	return &fakeinternalversioncore.FakeCore{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() internalversionextensions.ExtensionsInterface {
	return &fakeinternalversionextensions.FakeExtensions{Fake: &c.Fake}
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() internalversionfederation.FederationInterface {
	return &fakeinternalversionfederation.FakeFederation{Fake: &c.Fake}
}
