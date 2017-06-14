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

package admission

import (
	"net/url"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// TestAuthorizer is a testing struct for testing that fulfills the authorizer interface.
type TestAuthorizer struct{}

func (t *TestAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "", nil
}

var _ authorizer.Authorizer = &TestAuthorizer{}

type doNothingAdmission struct{}

func (doNothingAdmission) Admit(a admission.Attributes) error { return nil }
func (doNothingAdmission) Handles(o admission.Operation) bool { return false }
func (doNothingAdmission) Validate() error                    { return nil }

// WantAuthorizerAdmission is a testing struct that fulfills the WantsAuthorizer
// interface.
type WantAuthorizerAdmission struct {
	doNothingAdmission
	auth authorizer.Authorizer
}

func (self *WantAuthorizerAdmission) SetAuthorizer(a authorizer.Authorizer) {
	self.auth = a
}

var _ admission.Interface = &WantAuthorizerAdmission{}
var _ WantsAuthorizer = &WantAuthorizerAdmission{}

// TestWantsAuthorizer ensures that the authorizer is injected when the WantsAuthorizer
// interface is implemented.
func TestWantsAuthorizer(t *testing.T) {
	initializer := NewPluginInitializer(nil, nil, nil, &TestAuthorizer{}, nil, nil, nil)
	wantAuthorizerAdmission := &WantAuthorizerAdmission{}
	initializer.Initialize(wantAuthorizerAdmission)
	if wantAuthorizerAdmission.auth == nil {
		t.Errorf("expected authorizer to be initialized but found nil")
	}
}

type WantsCloudConfigAdmissionPlugin struct {
	doNothingAdmission
	cloudConfig []byte
}

func (self *WantsCloudConfigAdmissionPlugin) SetCloudConfig(cloudConfig []byte) {
	self.cloudConfig = cloudConfig
}

func TestCloudConfigAdmissionPlugin(t *testing.T) {
	cloudConfig := []byte("cloud-configuration")
	initializer := NewPluginInitializer(nil, nil, nil, &TestAuthorizer{}, cloudConfig, nil, nil)
	wantsCloudConfigAdmission := &WantsCloudConfigAdmissionPlugin{}
	initializer.Initialize(wantsCloudConfigAdmission)

	if wantsCloudConfigAdmission.cloudConfig == nil {
		t.Errorf("Expected cloud config to be initialized but found nil")
	}
}

type fakeServiceResolver struct{}

func (f *fakeServiceResolver) ResolveEndpoint(namespace, name string) (*url.URL, error) {
	return nil, nil
}

type serviceWanter struct {
	doNothingAdmission
	got ServiceResolver
}

func (s *serviceWanter) SetServiceResolver(sr ServiceResolver) { s.got = sr }

func TestWantsServiceResolver(t *testing.T) {
	sw := &serviceWanter{}
	fsr := &fakeServiceResolver{}
	i := &PluginInitializer{}
	i.SetServiceResolver(fsr).Initialize(sw)
	if got, ok := sw.got.(*fakeServiceResolver); !ok || got != fsr {
		t.Errorf("plumbing fail - %v %v#", ok, got)
	}
}

type clientCertWanter struct {
	doNothingAdmission
	gotCert, gotKey []byte
}

func (s *clientCertWanter) SetClientCert(cert, key []byte) { s.gotCert, s.gotKey = cert, key }

func TestWantsClientCert(t *testing.T) {
	i := &PluginInitializer{}
	ccw := &clientCertWanter{}
	i.SetClientCert([]byte("cert"), []byte("key")).Initialize(ccw)
	if string(ccw.gotCert) != "cert" || string(ccw.gotKey) != "key" {
		t.Errorf("plumbing fail - %v %v", ccw.gotCert, ccw.gotKey)
	}
}
