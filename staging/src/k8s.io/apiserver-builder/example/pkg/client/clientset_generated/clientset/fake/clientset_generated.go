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
	clientset "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset"
	innsmouthv1 "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/innsmouth/v1"
	fakeinnsmouthv1 "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/innsmouth/v1/fake"
	miskatonicv1beta1 "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/miskatonic/v1beta1"
	fakemiskatonicv1beta1 "k8s.io/apiserver-builder/example/pkg/client/clientset_generated/clientset/typed/miskatonic/v1beta1/fake"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/testing"
)

// NewSimpleClientset returns a clientset that will respond with the provided objects.
// It's backed by a very simple object tracker that processes creates, updates and deletions as-is,
// without applying any validations and/or defaults. It shouldn't be considered a replacement
// for a real clientset and is mostly useful in simple unit tests.
func NewSimpleClientset(objects ...runtime.Object) *Clientset {
	o := testing.NewObjectTracker(registry, scheme, codecs.UniversalDecoder())
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakePtr := testing.Fake{}
	fakePtr.AddReactor("*", "*", testing.ObjectReaction(o, registry.RESTMapper()))

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

// InnsmouthV1 retrieves the InnsmouthV1Client
func (c *Clientset) InnsmouthV1() innsmouthv1.InnsmouthV1Interface {
	return &fakeinnsmouthv1.FakeInnsmouthV1{Fake: &c.Fake}
}

// Innsmouth retrieves the InnsmouthV1Client
func (c *Clientset) Innsmouth() innsmouthv1.InnsmouthV1Interface {
	return &fakeinnsmouthv1.FakeInnsmouthV1{Fake: &c.Fake}
}

// MiskatonicV1beta1 retrieves the MiskatonicV1beta1Client
func (c *Clientset) MiskatonicV1beta1() miskatonicv1beta1.MiskatonicV1beta1Interface {
	return &fakemiskatonicv1beta1.FakeMiskatonicV1beta1{Fake: &c.Fake}
}

// Miskatonic retrieves the MiskatonicV1beta1Client
func (c *Clientset) Miskatonic() miskatonicv1beta1.MiskatonicV1beta1Interface {
	return &fakemiskatonicv1beta1.FakeMiskatonicV1beta1{Fake: &c.Fake}
}
