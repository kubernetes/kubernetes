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

package manifest

import (
	"os"
	"path"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestValidateStaticManifestsDir(t *testing.T) {
	validDir := t.TempDir()
	notADirFile := path.Join(t.TempDir(), "file.txt")
	if err := os.WriteFile(notADirFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name              string
		dir               string
		enableFeatureGate bool
		wantErr           string
	}{
		{
			name:              "empty dir is valid",
			dir:               "",
			enableFeatureGate: true,
		},
		{
			name:              "valid directory",
			dir:               validDir,
			enableFeatureGate: true,
		},
		{
			name:              "feature gate disabled",
			dir:               "/some/path",
			enableFeatureGate: false,
			wantErr:           "Forbidden",
		},
		{
			name:              "relative path",
			dir:               "relative/path",
			enableFeatureGate: true,
			wantErr:           "must be an absolute file path",
		},
		{
			name:              "nonexistent path",
			dir:               "/nonexistent/path",
			enableFeatureGate: true,
			wantErr:           "unable to read",
		},
		{
			name:              "not a directory",
			dir:               notADirFile,
			enableFeatureGate: true,
			wantErr:           "must be a directory",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, tc.enableFeatureGate)
			err := ValidateStaticManifestsDir(tc.dir)
			if len(tc.wantErr) > 0 {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
