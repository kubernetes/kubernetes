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
	"testing"

	"k8s.io/kubernetes/pkg/auth/authorizer"
)

// TestAuthorizer is a testing struct for testing that fulfills the authorizer interface.
type TestAuthorizer struct{}

func (t *TestAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "", nil
}

var _ authorizer.Authorizer = &TestAuthorizer{}

// WantAuthorizerAdmission is a testing struct that fulfills the WantsAuthorizer
// interface.
type WantAuthorizerAdmission struct {
	auth authorizer.Authorizer
}

func (self *WantAuthorizerAdmission) SetAuthorizer(a authorizer.Authorizer) {
	self.auth = a
}
func (self *WantAuthorizerAdmission) Admit(a Attributes) error { return nil }
func (self *WantAuthorizerAdmission) Handles(o Operation) bool { return false }
func (self *WantAuthorizerAdmission) Validate() error          { return nil }

var _ Interface = &WantAuthorizerAdmission{}
var _ WantsAuthorizer = &WantAuthorizerAdmission{}

// TestWantsAuthorizer ensures that the authorizer is injected when the WantsAuthorizer
// interface is implemented.
func TestWantsAuthorizer(t *testing.T) {
	initializer := NewPluginInitializer(nil, &TestAuthorizer{})
	wantAuthorizerAdmission := &WantAuthorizerAdmission{}
	initializer.Initialize([]Interface{wantAuthorizerAdmission})
	if wantAuthorizerAdmission.auth == nil {
		t.Errorf("expected authorizer to be initialized but found nil")
	}
}
