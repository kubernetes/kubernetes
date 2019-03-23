/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"testing"

	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
)

func TestAuthzValidate(t *testing.T) {
	examplePolicyFile := "../../auth/authorizer/abac/example_policy_file.jsonl"

	testCases := []struct {
		name              string
		modes             []string
		policyFile        string
		webhookConfigFile string
		expectErr         bool
	}{
		{
			name:      "Unknown modes should return errors",
			modes:     []string{"DoesNotExist"},
			expectErr: true,
		},
		{
			name:      "At least one authorizationMode is necessary",
			modes:     []string{},
			expectErr: true,
		},
		{
			name:      "ModeAlwaysAllow and ModeAlwaysDeny should return without authorizationPolicyFile",
			modes:     []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny},
			expectErr: false,
		},
		{
			name:      "ModeABAC requires a policy file",
			modes:     []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny, modes.ModeABAC},
			expectErr: true,
		},
		{
			name:              "Authorization Policy file cannot be used without ModeABAC",
			modes:             []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny},
			policyFile:        examplePolicyFile,
			webhookConfigFile: "",
			expectErr:         true,
		},
		{
			name:              "ModeABAC should not error if a valid policy path is provided",
			modes:             []string{modes.ModeAlwaysAllow, modes.ModeAlwaysDeny, modes.ModeABAC},
			policyFile:        examplePolicyFile,
			webhookConfigFile: "",
			expectErr:         false,
		},
		{
			name:      "ModeWebhook requires a config file",
			modes:     []string{modes.ModeWebhook},
			expectErr: true,
		},
		{
			name:              "Cannot provide webhook config file without ModeWebhook",
			modes:             []string{modes.ModeAlwaysAllow},
			webhookConfigFile: "authz_webhook_config.yaml",
			expectErr:         true,
		},
		{
			name:              "ModeWebhook should not error if a valid config file is provided",
			modes:             []string{modes.ModeWebhook},
			webhookConfigFile: "authz_webhook_config.yaml",
			expectErr:         false,
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			options := NewBuiltInAuthorizationOptions()
			options.Modes = testcase.modes
			options.WebhookConfigFile = testcase.webhookConfigFile
			options.PolicyFile = testcase.policyFile

			errs := options.Validate()
			if len(errs) > 0 && !testcase.expectErr {
				t.Errorf("got unexpected err %v", errs)
			}
			if testcase.expectErr && len(errs) == 0 {
				t.Errorf("should return an error")
			}
		})
	}
}
