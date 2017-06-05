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

package initializer_test

import (
	"testing"
	"time"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// TestWantsAuthorizer ensures that the authorizer is injected
// when the WantsAuthorizer interface is implemented by a plugin.
func TestWantsAuthorizer(t *testing.T) {
	target, err := initializer.New(nil, nil, &TestAuthorizer{})
	if err != nil {
		t.Fatalf("expected to create an instance of initializer but got an error = %s", err.Error())
	}
	wantAuthorizerAdmission := &WantAuthorizerAdmission{}
	target.Initialize(wantAuthorizerAdmission)
	if wantAuthorizerAdmission.auth == nil {
		t.Errorf("expected authorizer to be initialized but found nil")
	}
}

// TestWantsExternalKubeClientSet ensures that the clienset is injected
// when the WantsExternalKubeClientSet interface is implemented by a plugin.
func TestWantsExternalKubeClientSet(t *testing.T) {
	cs := &fake.Clientset{}
	target, err := initializer.New(cs, nil, &TestAuthorizer{})
	if err != nil {
		t.Fatalf("expected to create an instance of initializer but got an error = %s", err.Error())
	}
	wantExternalKubeClientSet := &WantExternalKubeClientSet{}
	target.Initialize(wantExternalKubeClientSet)
	if wantExternalKubeClientSet.cs != cs {
		t.Errorf("expected clientset to be initialized")
	}
}

// TestWantsExternalKubeInformerFactory ensures that the informer factory is injected
// when the WantsExternalKubeInformerFactory interface is implemented by a plugin.
func TestWantsExternalKubeInformerFactory(t *testing.T) {
	cs := &fake.Clientset{}
	sf := informers.NewSharedInformerFactory(cs, time.Duration(1)*time.Second)
	target, err := initializer.New(cs, sf, &TestAuthorizer{})
	if err != nil {
		t.Fatalf("expected to create an instance of initializer but got an error = %s", err.Error())
	}
	wantExternalKubeInformerFactory := &WantExternalKubeInformerFactory{}
	target.Initialize(wantExternalKubeInformerFactory)
	if wantExternalKubeInformerFactory.sf != sf {
		t.Errorf("expected informer factory to be initialized")
	}
}

// WantExternalKubeInformerFactory is a test stub that fulfills the WantsExternalKubeInformerFactory interface
type WantExternalKubeInformerFactory struct {
	sf informers.SharedInformerFactory
}

func (self *WantExternalKubeInformerFactory) SetExternalKubeInformerFactory(sf informers.SharedInformerFactory) {
	self.sf = sf
}
func (self *WantExternalKubeInformerFactory) Admit(a admission.Attributes) error { return nil }
func (self *WantExternalKubeInformerFactory) Handles(o admission.Operation) bool { return false }
func (self *WantExternalKubeInformerFactory) Validate() error                    { return nil }

var _ admission.Interface = &WantExternalKubeInformerFactory{}
var _ initializer.WantsExternalKubeInformerFactory = &WantExternalKubeInformerFactory{}

// WantExternalKubeClientSet is a test stub that fulfills the WantsExternalKubeClientSet interface
type WantExternalKubeClientSet struct {
	cs kubernetes.Interface
}

func (self *WantExternalKubeClientSet) SetExternalKubeClientSet(cs kubernetes.Interface) { self.cs = cs }
func (self *WantExternalKubeClientSet) Admit(a admission.Attributes) error               { return nil }
func (self *WantExternalKubeClientSet) Handles(o admission.Operation) bool               { return false }
func (self *WantExternalKubeClientSet) Validate() error                                  { return nil }

var _ admission.Interface = &WantExternalKubeClientSet{}
var _ initializer.WantsExternalKubeClientSet = &WantExternalKubeClientSet{}

// WantAuthorizerAdmission is a test stub that fulfills the WantsAuthorizer interface.
type WantAuthorizerAdmission struct {
	auth authorizer.Authorizer
}

func (self *WantAuthorizerAdmission) SetAuthorizer(a authorizer.Authorizer) { self.auth = a }
func (self *WantAuthorizerAdmission) Admit(a admission.Attributes) error    { return nil }
func (self *WantAuthorizerAdmission) Handles(o admission.Operation) bool    { return false }
func (self *WantAuthorizerAdmission) Validate() error                       { return nil }

var _ admission.Interface = &WantAuthorizerAdmission{}
var _ initializer.WantsAuthorizer = &WantAuthorizerAdmission{}

// TestAuthorizer is a test stub for testing that fulfills the authorizer interface.
type TestAuthorizer struct{}

func (t *TestAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "", nil
}
