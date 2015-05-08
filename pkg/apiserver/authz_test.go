/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"testing"
)

// NewAlwaysAllowAuthorizer must return a struct which implements authorizer.Authorizer
// and always return nil.
func TestNewAlwaysAllowAuthorizer(t *testing.T) {
	aaa := NewAlwaysAllowAuthorizer()
	if result := aaa.Authorize(nil); result != nil {
		t.Errorf("AlwaysAllowAuthorizer.Authorize did not return nil. (%s)", result)
	}
}

// NewAlwaysDenyAuthorizer must return a struct which implements authorizer.Authorizer
// and always return an error as everything is forbidden.
func TestNewAlwaysDenyAuthorizer(t *testing.T) {
	ada := NewAlwaysDenyAuthorizer()
	if result := ada.Authorize(nil); result == nil {
		t.Errorf("AlwaysDenyAuthorizer.Authorize returned nil instead of error.")
	}
}

// NewAuthorizerFromAuthorizationConfig has multiple return possibilities. This test
// validates that errors are returned only when proper.
func TestNewAuthorizerFromAuthorizationConfig(t *testing.T) {
	// Unknown modes should return errors
	if _, err := NewAuthorizerFromAuthorizationConfig("DoesNotExist", ""); err == nil {
		t.Errorf("NewAuthorizerFromAuthorizationConfig using a fake mode should have returned an error")
	}

	// ModeAlwaysAllow and ModeAlwaysDeny should return without authorizationPolicyFile
	// but error if one is given
	for _, config := range []string{ModeAlwaysAllow, ModeAlwaysDeny} {
		if _, err := NewAuthorizerFromAuthorizationConfig(config, ""); err != nil {
			t.Errorf("NewAuthorizerFromAuthorizationConfig with %s returned an error: %s", err, config)
		}
		if _, err := NewAuthorizerFromAuthorizationConfig(config, "shoulderror"); err == nil {
			t.Errorf("NewAuthorizerFromAuthorizationConfig with %s should have returned an error", config)
		}
	}

	// ModeABAC requires a policy file
	if _, err := NewAuthorizerFromAuthorizationConfig(ModeABAC, ""); err == nil {
		t.Errorf("NewAuthorizerFromAuthorizationConfig using a fake mode should have returned an error")
	}
	// ModeABAC should not error if a valid policy path is provided
	if _, err := NewAuthorizerFromAuthorizationConfig(ModeABAC, "../auth/authorizer/abac/example_policy_file.jsonl"); err != nil {
		t.Errorf("NewAuthorizerFromAuthorizationConfig errored while using a valid policy file: %s", err)
	}
}
