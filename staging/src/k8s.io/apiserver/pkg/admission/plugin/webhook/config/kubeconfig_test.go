/*
Copyright 2019 The Kubernetes Authors.

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

package config

import (
	"bytes"
	"strings"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	testcases := []struct {
		name             string
		input            string
		expectErr        string
		expectKubeconfig string
	}{
		{
			name:      "empty",
			input:     "",
			expectErr: `'Kind' is missing in ''`,
		},
		{
			name:      "unknown kind",
			input:     `{"kind":"Unknown","apiVersion":"v1"}`,
			expectErr: `no kind "Unknown" is registered for version "v1"`,
		},
		{
			name: "valid v1alpha1",
			input: `
kind: WebhookAdmission
apiVersion: apiserver.config.k8s.io/v1alpha1
kubeConfigFile: /foo
`,
			expectKubeconfig: "/foo",
		},
		{
			name: "valid v1",
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
`,
			expectKubeconfig: "/foo",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			kubeconfig, err := LoadConfig(bytes.NewBufferString(tc.input))
			if len(tc.expectErr) > 0 {
				if err == nil {
					t.Fatal("expected err, got none")
				}
				if !strings.Contains(err.Error(), tc.expectErr) {
					t.Fatalf("expected err containing %q, got %v", tc.expectErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if kubeconfig != tc.expectKubeconfig {
				t.Fatalf("expected %q, got %q", tc.expectKubeconfig, kubeconfig)
			}
		})
	}
}
