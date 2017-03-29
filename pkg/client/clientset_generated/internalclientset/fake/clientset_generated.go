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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	appsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion"
	fakeappsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/apps/internalversion/fake"
	authenticationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion"
	fakeauthenticationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion/fake"
	authorizationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion"
	fakeauthorizationinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion/fake"
	autoscalinginternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	fakeautoscalinginternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion/fake"
	batchinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion"
	fakebatchinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/batch/internalversion/fake"
	certificatesinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	fakecertificatesinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion/fake"
	coreinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	fakecoreinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion/fake"
	extensionsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	fakeextensionsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion/fake"
	policyinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion"
	fakepolicyinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/policy/internalversion/fake"
	rbacinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion"
	fakerbacinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion/fake"
	settingsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/settings/internalversion"
	fakesettingsinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/settings/internalversion/fake"
	storageinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion"
	fakestorageinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/storage/internalversion/fake"
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

// Core retrieves the CoreClient
func (c *Clientset) Core() coreinternalversion.CoreInterface {
	return &fakecoreinternalversion.FakeCore{Fake: &c.Fake}
}

// Apps retrieves the AppsClient
func (c *Clientset) Apps() appsinternalversion.AppsInterface {
	return &fakeappsinternalversion.FakeApps{Fake: &c.Fake}
}

// Authentication retrieves the AuthenticationClient
func (c *Clientset) Authentication() authenticationinternalversion.AuthenticationInterface {
	return &fakeauthenticationinternalversion.FakeAuthentication{Fake: &c.Fake}
}

// Authorization retrieves the AuthorizationClient
func (c *Clientset) Authorization() authorizationinternalversion.AuthorizationInterface {
	return &fakeauthorizationinternalversion.FakeAuthorization{Fake: &c.Fake}
}

// Autoscaling retrieves the AutoscalingClient
func (c *Clientset) Autoscaling() autoscalinginternalversion.AutoscalingInterface {
	return &fakeautoscalinginternalversion.FakeAutoscaling{Fake: &c.Fake}
}

// Batch retrieves the BatchClient
func (c *Clientset) Batch() batchinternalversion.BatchInterface {
	return &fakebatchinternalversion.FakeBatch{Fake: &c.Fake}
}

// Certificates retrieves the CertificatesClient
func (c *Clientset) Certificates() certificatesinternalversion.CertificatesInterface {
	return &fakecertificatesinternalversion.FakeCertificates{Fake: &c.Fake}
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() extensionsinternalversion.ExtensionsInterface {
	return &fakeextensionsinternalversion.FakeExtensions{Fake: &c.Fake}
}

// Policy retrieves the PolicyClient
func (c *Clientset) Policy() policyinternalversion.PolicyInterface {
	return &fakepolicyinternalversion.FakePolicy{Fake: &c.Fake}
}

// Rbac retrieves the RbacClient
func (c *Clientset) Rbac() rbacinternalversion.RbacInterface {
	return &fakerbacinternalversion.FakeRbac{Fake: &c.Fake}
}

// Settings retrieves the SettingsClient
func (c *Clientset) Settings() settingsinternalversion.SettingsInterface {
	return &fakesettingsinternalversion.FakeSettings{Fake: &c.Fake}
}

// Storage retrieves the StorageClient
func (c *Clientset) Storage() storageinternalversion.StorageInterface {
	return &fakestorageinternalversion.FakeStorage{Fake: &c.Fake}
}
