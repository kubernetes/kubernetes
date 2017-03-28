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

package authorizer

import (
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"testing"
)

// New has multiple return possibilities. This test
// validates that errors are returned only when proper.
func TestNew(t *testing.T) {
	examplePolicyFile := "../../auth/authorizer/abac/example_policy_file.jsonl"

	tests := []struct {
		config  AuthorizationConfig
		wantErr bool
		msg     string
	}{
		{
			// Unknown modes should return errors
			config:  AuthorizationConfig{AuthorizationModes: []string{"DoesNotExist"}},
			wantErr: true,
			msg:     "using a fake mode should have returned an error",
		},
		{
			// ModeAlwaysAllow and ModeAlwaysDeny should return without authorizationPolicyFile
			// but error if one is given
			config: AuthorizationConfig{AuthorizationModes: []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny}},
			msg:    "returned an error for valid config",
		},
		{
			// ModeABAC requires a policy file
			config:  AuthorizationConfig{AuthorizationModes: []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny, modes.ModeABAC}},
			wantErr: true,
			msg:     "specifying ABAC with no policy file should return an error",
		},
		{
			// ModeABAC should not error if a valid policy path is provided
			config: AuthorizationConfig{
				AuthorizationModes: []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny, modes.ModeABAC},
				PolicyFile:         examplePolicyFile,
			},
			msg: "errored while using a valid policy file",
		},
		{

			// Authorization Policy file cannot be used without ModeABAC
			config: AuthorizationConfig{
				AuthorizationModes: []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny},
				PolicyFile:         examplePolicyFile,
			},
			wantErr: true,
			msg:     "should have errored when Authorization Policy File is used without ModeABAC",
		},
		{
			// At least one authorizationMode is necessary
			config:  AuthorizationConfig{PolicyFile: examplePolicyFile},
			wantErr: true,
			msg:     "should have errored when no authorization modes are passed",
		},
		{
			// ModeWebhook requires at minimum a target.
			config:  AuthorizationConfig{AuthorizationModes: []string{modes.ModeWebhook}},
			wantErr: true,
			msg:     "should have errored when config was empty with ModeWebhook",
		},
		{
			// Cannot provide webhook flags without ModeWebhook
			config: AuthorizationConfig{
				AuthorizationModes: []string{modes.ModeAlwaysAllow},
				WebhookConfigFile:  "authz_webhook_config.yml",
			},
			wantErr: true,
			msg:     "should have errored when Webhook config file is used without ModeWebhook",
		},
	}

	for _, tt := range tests {
		_, err := tt.config.New()
		if tt.wantErr && (err == nil) {
			t.Errorf("New %s", tt.msg)
		} else if !tt.wantErr && (err != nil) {
			t.Errorf("New %s: %v", tt.msg, err)
		}
	}
}
