/*
Copyright 2026 The Kubernetes Authors.

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

package version

import "testing"

func TestValidateDynamicVersion(t *testing.T) {
	testCases := []struct {
		desc           string
		dynamicVersion string
		defaultVersion string
		expectErr      bool
	}{
		{
			desc:           "empty version is rejected",
			dynamicVersion: "",
			defaultVersion: "v1.30.0",
			expectErr:      true,
		},
		{
			desc:           "identical to default is a no-op even if not semver",
			dynamicVersion: "not-a-semver",
			defaultVersion: "not-a-semver",
			expectErr:      false,
		},
		{
			desc:           "invalid semver is rejected",
			dynamicVersion: "not-a-semver",
			defaultVersion: "v1.30.0",
			expectErr:      true,
		},
		{
			desc:           "placeholder default matches v0.0.0 dynamic version",
			dynamicVersion: "v0.0.0-dev+abcdef",
			defaultVersion: "v0.0.0-master+$Format:%H$",
			expectErr:      false,
		},
		{
			desc:           "placeholder default rejects a non v0.0.0 dynamic version",
			dynamicVersion: "v1.30.0",
			defaultVersion: "v0.0.0-master+$Format:%H$",
			expectErr:      true,
		},
		{
			desc:           "matching major minor patch with different build metadata is accepted",
			dynamicVersion: "v1.30.2-beta.0+abcdef",
			defaultVersion: "v1.30.2",
			expectErr:      false,
		},
		{
			desc:           "mismatched major is rejected",
			dynamicVersion: "v2.30.2",
			defaultVersion: "v1.30.2",
			expectErr:      true,
		},
		{
			desc:           "mismatched minor is rejected",
			dynamicVersion: "v1.31.2",
			defaultVersion: "v1.30.2",
			expectErr:      true,
		},
		{
			desc:           "mismatched patch is rejected",
			dynamicVersion: "v1.30.3",
			defaultVersion: "v1.30.2",
			expectErr:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := validateDynamicVersion(tc.dynamicVersion, tc.defaultVersion)
			if tc.expectErr && err == nil {
				t.Error("expected an error, got nil")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestSetDynamicVersion(t *testing.T) {
	original := dynamicGitVersion.Load().(string)
	t.Cleanup(func() { dynamicGitVersion.Store(original) })

	if err := SetDynamicVersion("not valid"); err == nil {
		t.Error("expected an error for an invalid dynamic version, got nil")
	}
	if got := Get().GitVersion; got != original {
		t.Errorf("GitVersion changed after a rejected SetDynamicVersion call: got %q, want %q", got, original)
	}

	if err := SetDynamicVersion(gitVersion); err != nil {
		t.Errorf("unexpected error setting dynamic version to the default: %v", err)
	}
	if got := Get().GitVersion; got != gitVersion {
		t.Errorf("GitVersion mismatch: got %q, want %q", got, gitVersion)
	}
}
