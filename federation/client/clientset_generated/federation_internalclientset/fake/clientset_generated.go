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
	"k8s.io/client-go/testing"
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	autoscalinginternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion"
	fakeautoscalinginternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/autoscaling/internalversion/fake"
	batchinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion"
	fakebatchinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/batch/internalversion/fake"
	coreinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion"
	fakecoreinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/core/internalversion/fake"
	extensionsinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion"
	fakeextensionsinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/extensions/internalversion/fake"
	federationinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion"
	fakefederationinternalversion "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset/typed/federation/internalversion/fake"
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

// Core retrieves the CoreClient
func (c *Clientset) Core() coreinternalversion.CoreInterface {
	return &fakecoreinternalversion.FakeCore{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() autoscalinginternalversion.AutoscalingInterface {
	return &fakeautoscalinginternalversion.FakeAutoscaling{Fake: &c.Fake}
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() batchinternalversion.BatchInterface {
	return &fakebatchinternalversion.FakeBatch{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() extensionsinternalversion.ExtensionsInterface {
	return &fakeextensionsinternalversion.FakeExtensions{Fake: &c.Fake}
}

// Federation retrieves the FederationClient
func (c *Clientset) Federation() federationinternalversion.FederationInterface {
	return &fakefederationinternalversion.FakeFederation{Fake: &c.Fake}
}
