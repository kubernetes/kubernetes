/*
Copyright The Kubernetes Authors.

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

func TestLoadValidatingConfig(t *testing.T) {
	validDir := t.TempDir()
	notADirFile := path.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(notADirFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name                     string
		input                    string
		enableFeatureGate        *bool
		expectErr                string
		expectStaticManifestsDir string
	}{
		{
			name:      "empty input",
			input:     "",
			expectErr: `'Kind' is missing`,
		},
		{
			name:      "unknown kind",
			input:     `{"kind":"Unknown","apiVersion":"v1"}`,
			expectErr: `no kind "Unknown" is registered`,
		},
		{
			name: "valid config with staticManifestsDir",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: ` + validDir + `
`,
			enableFeatureGate:        new(true),
			expectStaticManifestsDir: validDir,
		},
		{
			name: "valid config with empty staticManifestsDir",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
`,
		},
		{
			name: "forbidden when feature gate disabled",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: /etc/kubernetes/policies
`,
			enableFeatureGate: new(false),
			expectErr:         "staticManifestsDir: Forbidden",
		},
		{
			name: "invalid relative path",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: relative/path
`,
			enableFeatureGate: new(true),
			expectErr:         "must be an absolute file path",
		},
		{
			name: "path does not exist",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: /nonexistent/path
`,
			enableFeatureGate: new(true),
			expectErr:         "unable to read",
		},
		{
			name: "path is not a directory",
			input: `
kind: ValidatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: ` + notADirFile + `
`,
			enableFeatureGate: new(true),
			expectErr:         "must be a directory",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableFeatureGate != nil {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, *tc.enableFeatureGate)
			}
			cfg, err := LoadValidatingConfig(bytes.NewBufferString(tc.input))
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
			if cfg.StaticManifestsDir != tc.expectStaticManifestsDir {
				t.Fatalf("expected StaticManifestsDir %q, got %q", tc.expectStaticManifestsDir, cfg.StaticManifestsDir)
			}
		})
	}
}

func TestLoadMutatingConfig(t *testing.T) {
	validDir := t.TempDir()
	notADirFile := path.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(notADirFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	testcases := []struct {
		name                     string
		input                    string
		enableFeatureGate        *bool
		expectErr                string
		expectStaticManifestsDir string
	}{
		{
			name:      "empty input",
			input:     "",
			expectErr: `'Kind' is missing`,
		},
		{
			name:      "unknown kind",
			input:     `{"kind":"Unknown","apiVersion":"v1"}`,
			expectErr: `no kind "Unknown" is registered`,
		},
		{
			name: "valid config with staticManifestsDir",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: ` + validDir + `
`,
			enableFeatureGate:        new(true),
			expectStaticManifestsDir: validDir,
		},
		{
			name: "valid config with empty staticManifestsDir",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
`,
		},
		{
			name: "forbidden when feature gate disabled",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: /etc/kubernetes/policies
`,
			enableFeatureGate: new(false),
			expectErr:         "staticManifestsDir: Forbidden",
		},
		{
			name: "invalid relative path",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: relative/path
`,
			enableFeatureGate: new(true),
			expectErr:         "must be an absolute file path",
		},
		{
			name: "path does not exist",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: /nonexistent/path
`,
			enableFeatureGate: new(true),
			expectErr:         "unable to read",
		},
		{
			name: "path is not a directory",
			input: `
kind: MutatingAdmissionPolicyConfiguration
apiVersion: apiserver.config.k8s.io/v1
staticManifestsDir: ` + notADirFile + `
`,
			enableFeatureGate: new(true),
			expectErr:         "must be a directory",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.enableFeatureGate != nil {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, *tc.enableFeatureGate)
			}
			cfg, err := LoadMutatingConfig(bytes.NewBufferString(tc.input))
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
			if cfg.StaticManifestsDir != tc.expectStaticManifestsDir {
				t.Fatalf("expected StaticManifestsDir %q, got %q", tc.expectStaticManifestsDir, cfg.StaticManifestsDir)
			}
		})
	}
}
