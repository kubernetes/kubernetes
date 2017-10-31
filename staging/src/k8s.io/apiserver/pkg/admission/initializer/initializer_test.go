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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// TestWantsScheme ensures that the scheme is injected when
// the WantsScheme interface is implemented by a plugin.
func TestWantsScheme(t *testing.T) {
	scheme := runtime.NewScheme()
	target, err := initializer.New(nil, nil, nil, scheme)
	if err != nil {
		t.Fatal(err)
	}
	wantSchemeAdmission := &WantSchemeAdmission{}
	target.Initialize(wantSchemeAdmission)
	if wantSchemeAdmission.scheme != scheme {
		t.Errorf("expected scheme to be initialized")
	}
}

// TestWantsAuthorizer ensures that the authorizer is injected
// when the WantsAuthorizer interface is implemented by a plugin.
func TestWantsAuthorizer(t *testing.T) {
	target, err := initializer.New(nil, nil, &TestAuthorizer{}, nil)
	if err != nil {
		t.Fatal(err)
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
	target, err := initializer.New(cs, nil, &TestAuthorizer{}, nil)
	if err != nil {
		t.Fatal(err)
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
	target, err := initializer.New(cs, sf, &TestAuthorizer{}, nil)
	if err != nil {
		t.Fatal(err)
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

// TestAuthorizer is a test stub that fulfills the WantsAuthorizer interface.
type TestAuthorizer struct{}

func (t *TestAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "", nil
}

// wantClientCert is a test stub for testing that fulfulls the WantsClientCert interface.
type clientCertWanter struct {
	gotCert, gotKey []byte
}

func (s *clientCertWanter) SetClientCert(cert, key []byte)     { s.gotCert, s.gotKey = cert, key }
func (s *clientCertWanter) Admit(a admission.Attributes) error { return nil }
func (s *clientCertWanter) Handles(o admission.Operation) bool { return false }
func (s *clientCertWanter) Validate() error                    { return nil }

// WantSchemeAdmission is a test stub that fulfills the WantsScheme interface.
type WantSchemeAdmission struct {
	scheme *runtime.Scheme
}

func (self *WantSchemeAdmission) SetScheme(s *runtime.Scheme)        { self.scheme = s }
func (self *WantSchemeAdmission) Admit(a admission.Attributes) error { return nil }
func (self *WantSchemeAdmission) Handles(o admission.Operation) bool { return false }
func (self *WantSchemeAdmission) Validate() error                    { return nil }

var _ admission.Interface = &WantSchemeAdmission{}
var _ initializer.WantsScheme = &WantSchemeAdmission{}
