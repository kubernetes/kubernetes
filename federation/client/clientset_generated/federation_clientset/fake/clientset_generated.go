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
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	autoscalingv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1"
	fakeautoscalingv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/autoscaling/v1/fake"
	batchv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/batch/v1"
	fakebatchv1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/batch/v1/fake"
	corev1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1"
	fakecorev1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/core/v1/fake"
	extensionsv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1"
	fakeextensionsv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/extensions/v1beta1/fake"
	federationv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1"
	fakefederationv1beta1 "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/typed/federation/v1beta1/fake"
	"k8s.io/kubernetes/pkg/api"
)

var Scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)
var parameterCodec = runtime.NewParameterCodec(scheme)

func init() {
	corev1.AddToScheme(Scheme)
	autoscalingv1.AddToScheme(Scheme)
	batchv1.AddToScheme(Scheme)
	extensionsv1beta1.AddToScheme(Scheme)
	federationv1beta1.AddToScheme(Scheme)

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

// AutoscalingV1 retrieves the AutoscalingV1Client
func (c *Clientset) AutoscalingV1() clientautoscalingv1.AutoscalingV1Interface {
	return &fakeclientautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingV1Client
func (c *Clientset) Autoscaling() clientautoscalingv1.AutoscalingV1Interface {
	return &fakeclientautoscalingv1.FakeAutoscalingV1{Fake: &c.Fake}
}

// BatchV1 retrieves the BatchV1Client
func (c *Clientset) BatchV1() clientbatchv1.BatchV1Interface {
	return &fakeclientbatchv1.FakeBatchV1{Fake: &c.Fake}
}

// Batch retrieves the BatchV1Client
func (c *Clientset) Batch() clientbatchv1.BatchV1Interface {
	return &fakeclientbatchv1.FakeBatchV1{Fake: &c.Fake}
}

// ExtensionsV1beta1 retrieves the ExtensionsV1beta1Client
func (c *Clientset) ExtensionsV1beta1() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeclientextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsV1beta1Client
func (c *Clientset) Extensions() clientextensionsv1beta1.ExtensionsV1beta1Interface {
	return &fakeclientextensionsv1beta1.FakeExtensionsV1beta1{Fake: &c.Fake}
}

// FederationV1beta1 retrieves the FederationV1beta1Client
func (c *Clientset) FederationV1beta1() clientfederationv1beta1.FederationV1beta1Interface {
	return &fakeclientfederationv1beta1.FakeFederationV1beta1{Fake: &c.Fake}
}

// Federation retrieves the FederationV1beta1Client
func (c *Clientset) Federation() clientfederationv1beta1.FederationV1beta1Interface {
	return &fakeclientfederationv1beta1.FakeFederationV1beta1{Fake: &c.Fake}
}
