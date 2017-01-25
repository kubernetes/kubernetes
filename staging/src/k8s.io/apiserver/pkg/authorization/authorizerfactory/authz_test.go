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

package authorizerfactory

import (
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// NewAlwaysAllowAuthorizer must return a struct which implements authorizer.Authorizer
// and always return nil.
func TestNewAlwaysAllowAuthorizer(t *testing.T) {
	aaa := NewAlwaysAllowAuthorizer()
	if authorized, _, _ := aaa.Authorize(nil); !authorized {
		t.Errorf("AlwaysAllowAuthorizer.Authorize did not authorize successfully.")
	}
}

// NewAlwaysDenyAuthorizer must return a struct which implements authorizer.Authorizer
// and always return an error as everything is forbidden.
func TestNewAlwaysDenyAuthorizer(t *testing.T) {
	ada := NewAlwaysDenyAuthorizer()
	if authorized, _, _ := ada.Authorize(nil); authorized {
		t.Errorf("AlwaysDenyAuthorizer.Authorize returned nil instead of error.")
	}
}

func TestPrivilegedGroupAuthorizer(t *testing.T) {
	auth := NewPrivilegedGroups("allow-01", "allow-01")

	yes := authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"no", "allow-01"}}}
	no := authorizer.AttributesRecord{User: &user.DefaultInfo{Groups: []string{"no", "deny-01"}}}

	if authorized, _, _ := auth.Authorize(yes); !authorized {
		t.Errorf("failed")
	}
	if authorized, _, _ := auth.Authorize(no); authorized {
		t.Errorf("failed")
	}
}
