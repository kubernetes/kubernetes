/*
Copyright 2014 The Kubernetes Authors.

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

// NewAuthorizerFromAuthorizationConfig has multiple return possibilities. This test
// validates that errors are returned only when proper.
func TestNewAuthorizerFromAuthorizationConfig(t *testing.T) {

	examplePolicyFile := "../auth/authorizer/abac/example_policy_file.jsonl"

	tests := []struct {
		modes   []string
		config  AuthorizationConfig
		wantErr bool
		msg     string
	}{
		{
			// Unknown modes should return errors
			modes:   []string{"DoesNotExist"},
			wantErr: true,
			msg:     "using a fake mode should have returned an error",
		},
		{
			// ModeAlwaysAllow and ModeAlwaysDeny should return without authorizationPolicyFile
			// but error if one is given
			modes: []string{ModeAlwaysAllow, ModeAlwaysDeny},
			msg:   "returned an error for valid config",
		},
		{
			// ModeABAC requires a policy file
			modes:   []string{ModeAlwaysAllow, ModeAlwaysDeny, ModeABAC},
			wantErr: true,
			msg:     "specifying ABAC with no policy file should return an error",
		},
		{
			// ModeABAC should not error if a valid policy path is provided
			modes:  []string{ModeAlwaysAllow, ModeAlwaysDeny, ModeABAC},
			config: AuthorizationConfig{PolicyFile: examplePolicyFile},
			msg:    "errored while using a valid policy file",
		},
		{

			// Authorization Policy file cannot be used without ModeABAC
			modes:   []string{ModeAlwaysAllow, ModeAlwaysDeny},
			config:  AuthorizationConfig{PolicyFile: examplePolicyFile},
			wantErr: true,
			msg:     "should have errored when Authorization Policy File is used without ModeABAC",
		},
		{
			// At least one authorizationMode is necessary
			modes:   []string{},
			config:  AuthorizationConfig{PolicyFile: examplePolicyFile},
			wantErr: true,
			msg:     "should have errored when no authorization modes are passed",
		},
		{
			// ModeWebhook requires at minimum a target.
			modes:   []string{ModeWebhook},
			wantErr: true,
			msg:     "should have errored when config was empty with ModeWebhook",
		},
		{
			// Cannot provide webhook flags without ModeWebhook
			modes:   []string{ModeAlwaysAllow},
			config:  AuthorizationConfig{WebhookConfigFile: "authz_webhook_config.yml"},
			wantErr: true,
			msg:     "should have errored when Webhook config file is used without ModeWebhook",
		},
	}

	for _, tt := range tests {
		_, err := NewAuthorizerFromAuthorizationConfig(tt.modes, tt.config)
		if tt.wantErr && (err == nil) {
			t.Errorf("NewAuthorizerFromAuthorizationConfig %s", tt.msg)
		} else if !tt.wantErr && (err != nil) {
			t.Errorf("NewAuthorizerFromAuthorizationConfig %s: %v", tt.msg, err)
		}
	}
}
