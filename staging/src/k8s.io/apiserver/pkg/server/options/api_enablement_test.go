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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apimachineryversion "k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	serverversion "k8s.io/apiserver/pkg/util/version"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

type fakeGroupRegistry struct{}

func (f fakeGroupRegistry) IsGroupRegistered(group string) bool {
	return group == "apiregistration.k8s.io"
}

func TestAPIEnablementOptionsValidate(t *testing.T) {
	t.Cleanup(serverversion.Effective.SetBinaryVersionForTests(apimachineryversion.MustParse("v1.30.0")))
	binaryVersion := serverversion.Effective.BinaryVersion().String()
	testCases := []struct {
		name          string
		testOptions   *APIEnablementOptions
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
				RuntimeConfig:    testcase.runtimeConfig,
				EmulationVersion: binaryVersion,
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

func TestAPIEnablementOptionsValidateEmulationVersion(t *testing.T) {
	t.Cleanup(serverversion.Effective.SetBinaryVersionForTests(apimachineryversion.MustParse("v1.30.0")))
	binaryVersion := serverversion.Effective.BinaryVersion().String()
	testCases := []struct {
		name             string
		emulationVersion string
		expectErr        string
	}{
		{
			name:      "test empty version",
			expectErr: "could not parse \"\" as version",
		},
		{
			name:             "test misformat version",
			emulationVersion: "a.b.c",
			expectErr:        "could not parse \"a.b.c\" as version",
		},
		{
			name:             "test binaryVersion",
			emulationVersion: binaryVersion,
		},
		{
			name:             "test valid emulation version",
			emulationVersion: "v1.29.0",
		},
		{
			name:             "test non semantic emulation version",
			emulationVersion: "1.29",
		},
		{
			name:             "test invalid emulation version",
			emulationVersion: "1.31.0",
			expectErr:        "emulation version 1.31.0 is not between [1.29.0, 1.30.0]",
		},
	}
	testGroupRegistry := fakeGroupRegistry{}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.EmulationVersion, true)()
			testOptions := &APIEnablementOptions{
				EmulationVersion: testcase.emulationVersion,
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
