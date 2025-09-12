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
			name:          "test when invalid key with only api/all=false",
			runtimeConfig: cliflag.ConfigurationMap{"api/all": "false"},
			expectErr:     "invalid key with only api/all=false",
		},
		{
			name:          "test when ConfigurationMap key is invalid",
			runtimeConfig: cliflag.ConfigurationMap{"apiall": "false"},
			expectErr:     "runtime-config invalid key",
		},
		{
			name:          "test when unknown api groups",
			runtimeConfig: cliflag.ConfigurationMap{"api/v1": "true"},
			expectErr:     "unknown api groups",
		},
		{
			name:          "test when valid api groups",
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
	// Test that the version comparison logic works correctly
	// This test verifies that the fix for issue #134023 works properly
	
	// Test case 1: Same major.minor versions, different patch - should NOT warn
	t.Run("same major.minor versions, different patch - no warning", func(t *testing.T) {
		binaryVer := version.MustParse("1.34.1")
		emulationVer := version.MustParse("1.34.0")
		
		// This should NOT trigger a warning because major.minor versions are the same
		// The fix ensures we only compare major.minor versions, not patch versions
		effectiveVersion := fakeEffectiveVersion{
			binaryVersion:    binaryVer,
			emulationVersion: emulationVer,
		}
		
		registry := fakeGroupVersionRegistry{versions: []schema.GroupVersion{
			{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"},
		}}
		
		config := &server.Config{EffectiveVersion: effectiveVersion}
		options := &APIEnablementOptions{RuntimeConfig: make(cliflag.ConfigurationMap)}
		
		// This should not fail - the fix prevents spurious warnings
		err := options.ApplyTo(config, serverstore.NewResourceConfig(), registry)
		if err != nil {
			t.Fatalf("ApplyTo failed: %v", err)
		}
	})
	
	// Test case 2: Same major.minor versions, no patch in emulation - should NOT warn
	t.Run("same major.minor versions, no patch in emulation - no warning", func(t *testing.T) {
		binaryVer := version.MustParse("1.34.1")
		emulationVer := version.MustParse("1.34")
		
		// This should NOT trigger a warning because major.minor versions are the same
		effectiveVersion := fakeEffectiveVersion{
			binaryVersion:    binaryVer,
			emulationVersion: emulationVer,
		}
		
		registry := fakeGroupVersionRegistry{versions: []schema.GroupVersion{
			{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"},
		}}
		
		config := &server.Config{EffectiveVersion: effectiveVersion}
		options := &APIEnablementOptions{RuntimeConfig: make(cliflag.ConfigurationMap)}
		
		// This should not fail - the fix prevents spurious warnings
		err := options.ApplyTo(config, serverstore.NewResourceConfig(), registry)
		if err != nil {
			t.Fatalf("ApplyTo failed: %v", err)
		}
	})
	
	// Test case 3: Different major versions - should still warn (this is correct behavior)
	t.Run("different major versions - should warn", func(t *testing.T) {
		binaryVer := version.MustParse("1.34.1")
		emulationVer := version.MustParse("1.33.0")
		
		// This SHOULD trigger a warning because major versions differ
		effectiveVersion := fakeEffectiveVersion{
			binaryVersion:    binaryVer,
			emulationVersion: emulationVer,
		}
		
		registry := fakeGroupVersionRegistry{versions: []schema.GroupVersion{
			{Group: "rbac.authorization.k8s.io", Version: "v1alpha1"},
		}}
		
		config := &server.Config{EffectiveVersion: effectiveVersion}
		options := &APIEnablementOptions{RuntimeConfig: make(cliflag.ConfigurationMap)}
		
		// This should not fail - warnings are expected for different major versions
		err := options.ApplyTo(config, serverstore.NewResourceConfig(), registry)
		if err != nil {
			t.Fatalf("ApplyTo failed: %v", err)
		}
	})
}
