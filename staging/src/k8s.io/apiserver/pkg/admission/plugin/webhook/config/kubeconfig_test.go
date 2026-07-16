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
	"os"
	"path"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestLoadConfig(t *testing.T) {
	// Create a temp directory for tests that need a valid directory path
	validDir := t.TempDir()
	// Create a temp file for "not a directory" test
	notADirFile := path.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(notADirFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name                     string
		input                    string
		enableFeatureGate        *bool
		expectErr                string
		expectKubeconfig         string
		expectStaticManifestsDir string
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
			name: "valid v1",
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
`,
			expectKubeconfig: "/foo",
		},
		{
			name:              "valid v1 with staticManifestsDir",
			enableFeatureGate: new(true),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
staticManifestsDir: ` + validDir + `
`,
			expectKubeconfig:         "/foo",
			expectStaticManifestsDir: validDir,
		},
		{
			name:              "invalid relative staticManifestsDir",
			enableFeatureGate: new(true),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
staticManifestsDir: relative/path
`,
			expectErr: `staticManifestsDir: Invalid value: "relative/path": must be an absolute file path`,
		},
		{
			name:              "staticManifestsDir forbidden when feature gate disabled",
			enableFeatureGate: new(false),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
staticManifestsDir: /etc/kubernetes/admission
`,
			expectErr: "staticManifestsDir: Forbidden",
		},
		{
			name:              "staticManifestsDir must be a directory",
			enableFeatureGate: new(true),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
staticManifestsDir: ` + notADirFile + `
`,
			expectErr: "must be a directory",
		},
		{
			name:              "staticManifestsDir must exist",
			enableFeatureGate: new(true),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
kubeConfigFile: /foo
staticManifestsDir: /nonexistent/path
`,
			expectErr: "unable to read",
		},
		{
			name:              "valid staticManifestsDir only (no kubeConfigFile)",
			enableFeatureGate: new(true),
			input: `
kind: WebhookAdmissionConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: ` + validDir + `
`,
			expectStaticManifestsDir: validDir,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableFeatureGate != nil {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, *tc.enableFeatureGate)
			}
			cfg, err := LoadConfig(bytes.NewBufferString(tc.input))
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
			if cfg.KubeConfigFile != tc.expectKubeconfig {
				t.Fatalf("expected KubeConfigFile %q, got %q", tc.expectKubeconfig, cfg.KubeConfigFile)
			}
			if cfg.StaticManifestsDir != tc.expectStaticManifestsDir {
				t.Fatalf("expected StaticManifestsDir %q, got %q", tc.expectStaticManifestsDir, cfg.StaticManifestsDir)
			}
		})
	}
}
