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
	"bytes"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/server"
	serverstore "k8s.io/apiserver/pkg/server/storage"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/compatibility"
	"k8s.io/klog/v2"
)

type fakeGroupRegistry struct{}

func (f fakeGroupRegistry) IsGroupRegistered(group string) bool {
	return group == "apiregistration.k8s.io"
}

func TestAPIEnablementOptionsValidate(t *testing.T) {
	testCases := []struct {
		name          string
		runtimeConfig cliflag.ConfigurationMap
		expectErr     string
	}{
		{
			name: "test when options is nil",
		},
		{
			name:          "test when invalid runtime-config with only api/all=false",
			runtimeConfig: cliflag.ConfigurationMap{"api/all": "false"},
			expectErr:     "invalid runtime-config with only api/all=false",
		},
		{
			name:          "test when ConfigurationMap key is invalid",
			runtimeConfig: cliflag.ConfigurationMap{"apiall": "false"},
			expectErr:     "runtime-config invalid key",
		},
		{
			name:          "test when unknown api groups",
			runtimeConfig: cliflag.ConfigurationMap{"api/v1": "true", "api/v1beta2": "true"},
			expectErr:     "unknown api groups api/v1,api/v1beta2",
		},
		{
			name:          "test when valid api groups",
			runtimeConfig: cliflag.ConfigurationMap{"apiregistration.k8s.io/v1beta1": "true"},
		},
		{
			name:          "test when invalid api groups",
			runtimeConfig: cliflag.ConfigurationMap{"apiregistration.k8s.io/v1beta1": "true"},
		},
	}
	testGroupRegistry := fakeGroupRegistry{}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			testOptions := &APIEnablementOptions{
				RuntimeConfig: testcase.runtimeConfig,
			}
			errs := testOptions.Validate(testGroupRegistry)
			if len(testcase.expectErr) != 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) {
				t.Errorf("got err: %v, expected err: %s", errs, testcase.expectErr)
			}

			if len(testcase.expectErr) == 0 && len(errs) != 0 {
				t.Errorf("got err: %s, expected err nil", errs)
			}
		})
	}
}

type fakeGroupVersionRegistry struct {
	versions []schema.GroupVersion
}

func (f fakeGroupVersionRegistry) PrioritizedVersionsAllGroups() []schema.GroupVersion {
	return f.versions
}

func (f fakeGroupVersionRegistry) PrioritizedVersionsForGroup(group string) []schema.GroupVersion {
	var result []schema.GroupVersion
	for _, gv := range f.versions {
		if gv.Group == group {
			result = append(result, gv)
		}
	}
	return result
}

func (f fakeGroupVersionRegistry) IsGroupRegistered(group string) bool {
	for _, gv := range f.versions {
		if gv.Group == group {
			return true
		}
	}
	return false
}

func (f fakeGroupVersionRegistry) IsVersionRegistered(gv schema.GroupVersion) bool {
	for _, version := range f.versions {
		if version == gv {
			return true
		}
	}
	return false
}

func (f fakeGroupVersionRegistry) GroupVersions() []schema.GroupVersion {
	return f.versions
}

type fakeEffectiveVersion struct {
	binaryVersion    *version.Version
	emulationVersion *version.Version
}

func (f fakeEffectiveVersion) BinaryVersion() *version.Version {
	return f.binaryVersion
}

func (f fakeEffectiveVersion) EmulationVersion() *version.Version {
	return f.emulationVersion
}

func (f fakeEffectiveVersion) MinCompatibilityVersion() *version.Version {
	return nil
}

func (f fakeEffectiveVersion) EqualTo(other compatibility.EffectiveVersion) bool {
	return f.binaryVersion.EqualTo(other.BinaryVersion()) && f.emulationVersion.EqualTo(other.EmulationVersion())
}

func (f fakeEffectiveVersion) String() string {
	return "fake"
}

func (f fakeEffectiveVersion) Info() *apimachineryversion.Info {
	return nil
}

func (f fakeEffectiveVersion) AllowedEmulationVersionRange() string {
	return "fake range"
}

func (f fakeEffectiveVersion) AllowedMinCompatibilityVersionRange() string {
	return "fake range"
}

func (f fakeEffectiveVersion) Validate() []error {
	return nil
}

func TestAPIEnablementOptionsApplyToVersionComparison(t *testing.T) {
	// Helper function to capture klog output
	captureKlogOutput := func(fn func()) string {
		var buf bytes.Buffer
		klog.SetOutput(&buf)
		klog.LogToStderr(false)
		defer func() {
			klog.SetOutput(nil)
			klog.LogToStderr(true)
		}()

		fn()
		klog.Flush()
		return buf.String()
	}

	testCases := []struct {
		name                 string
		binaryVersion        string
		emulationVersion     string
		alphaAPIsPresent     bool
		versionEnabled       bool
		expectWarning        bool
		expectWarningContent string
	}{
		{
			name:             "same major.minor versions, different patch - no warning",
			binaryVersion:    "1.34.1",
			emulationVersion: "1.34.0",
			alphaAPIsPresent: true,
			versionEnabled:   true,
			expectWarning:    false,
		},
		{
			name:             "same major.minor versions, no patch in emulation - no warning",
			binaryVersion:    "1.34.1",
			emulationVersion: "1.34",
			alphaAPIsPresent: true,
			versionEnabled:   true,
			expectWarning:    false,
		},
		{
			name:             "identical versions - no warning",
			binaryVersion:    "1.34.1",
			emulationVersion: "1.34.1",
			alphaAPIsPresent: true,
			versionEnabled:   true,
			expectWarning:    false,
		},
		{
			name:             "different major versions but not enabled - should not warn",
			binaryVersion:    "1.34.1",
			emulationVersion: "1.33.0",
			alphaAPIsPresent: true,
			expectWarning:    false,
		},
		{
			name:                 "different major versions - should warn",
			binaryVersion:        "1.34.1",
			emulationVersion:     "1.33.0",
			alphaAPIsPresent:     true,
			expectWarning:        true,
			versionEnabled:       true,
			expectWarningContent: "alpha api enabled with emulated version",
		},
		{
			name:                 "different minor versions - should warn",
			binaryVersion:        "1.34.1",
			emulationVersion:     "1.33.5",
			alphaAPIsPresent:     true,
			expectWarning:        true,
			versionEnabled:       true,
			expectWarningContent: "alpha api enabled with emulated version",
		},
		{
			name:             "different major.minor but no alpha APIs - no warning",
			binaryVersion:    "1.34.1",
			emulationVersion: "1.33.0",
			alphaAPIsPresent: false,
			expectWarning:    false,
			versionEnabled:   true,
		},
		{
			name:             "same major.minor with alpha APIs - no warning",
			binaryVersion:    "1.34.5",
			emulationVersion: "1.34.0",
			alphaAPIsPresent: true,
			expectWarning:    false,
			versionEnabled:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			binaryVer := version.MustParse(tc.binaryVersion)
			emulationVer := version.MustParse(tc.emulationVersion)

			effectiveVersion := fakeEffectiveVersion{
				binaryVersion:    binaryVer,
				emulationVersion: emulationVer,
			}

			var versions []schema.GroupVersion
			if tc.alphaAPIsPresent {
				versions = []schema.GroupVersion{
					{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"},
					{Group: "storage.k8s.io", Version: "v1alpha1"},
				}
			} else {
				versions = []schema.GroupVersion{
					{Group: "rbac.authorization.k8s.io", Version: "v1"},
					{Group: "storage.k8s.io", Version: "v1beta1"},
				}
			}

			registry := fakeGroupVersionRegistry{versions: versions}
			config := &server.Config{EffectiveVersion: effectiveVersion}
			options := &APIEnablementOptions{RuntimeConfig: make(cliflag.ConfigurationMap)}

			// Enable the API
			resourceConfig := serverstore.NewResourceConfig()
			if tc.versionEnabled {
				resourceConfig.ExplicitGroupVersionConfigs[schema.GroupVersion{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"}] = true
			}

			// Capture log output during ApplyTo execution
			logOutput := captureKlogOutput(func() {
				err := options.ApplyTo(config, resourceConfig, registry)
				if err != nil {
					t.Errorf("ApplyTo failed: %v", err)
				}
			})

			// Verify warning expectations
			if tc.expectWarning {
				if !strings.Contains(logOutput, tc.expectWarningContent) {
					t.Errorf("Expected warning containing '%s', but got log output: %s", tc.expectWarningContent, logOutput)
				}
				if !strings.Contains(logOutput, "W") { // klog warning prefix
					t.Errorf("Expected warning log level, but got log output: %s", logOutput)
				}
			} else if strings.Contains(logOutput, "alpha api enabled") {
				t.Errorf("Expected no warning, but got log output: %s", logOutput)
			}
		})
	}
}

func TestAPIEnablementOptionsApplyToErrorCases(t *testing.T) {
	// Create a default effective version for test configs
	defaultEffectiveVersion := fakeEffectiveVersion{
		binaryVersion:    version.MustParse("1.34.0"),
		emulationVersion: version.MustParse("1.34.0"),
	}

	testCases := []struct {
		name          string
		options       *APIEnablementOptions
		config        *server.Config
		expectError   bool
		errorContains string
	}{
		{
			name:    "nil options should not error",
			options: nil,
			config: &server.Config{
				EffectiveVersion: defaultEffectiveVersion,
			},
			expectError: false,
		},
		{
			name: "invalid runtime config value should error",
			options: &APIEnablementOptions{
				RuntimeConfig: cliflag.ConfigurationMap{
					"api/all": "invalid-value", // Must be "true" or "false"
				},
			},
			config: &server.Config{
				EffectiveVersion: defaultEffectiveVersion,
			},
			expectError:   true,
			errorContains: "invalid value",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			registry := fakeGroupVersionRegistry{versions: []schema.GroupVersion{
				{Group: "rbac.authorization.k8s.io", Version: "v1"},
			}}

			err := tc.options.ApplyTo(tc.config, serverstore.NewResourceConfig(), registry)

			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				} else if tc.errorContains != "" && !strings.Contains(err.Error(), tc.errorContains) {
					t.Errorf("Expected error containing '%s', but got: %v", tc.errorContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
			}
		})
	}
}
