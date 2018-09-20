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

package seccomp

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

type testStrategy struct {
	defaultProfile  *string
	allowedProfiles []string
}

var (
	seccompFooProfile = "foo"
	withoutSeccomp    = testStrategy{}
	allowAnyNoDefault = testStrategy{allowedProfiles: []string{SeccompAllowAny}}
	allowAnyDefault   = testStrategy{
		allowedProfiles: []string{SeccompAllowAny},
		defaultProfile:  &seccompFooProfile,
	}
	allowAnyAndSpecificDefault = testStrategy{
		allowedProfiles: []string{"bar", SeccompAllowAny},
		defaultProfile:  &seccompFooProfile,
	}
	allowSpecificNoDefault = testStrategy{allowedProfiles: []string{"foo"}}
	allowMultipleNoDefault = testStrategy{
		allowedProfiles: []string{"foo", "bar"},
	}
	allowMultipleDefault = testStrategy{
		allowedProfiles: []string{"foo", "bar"},
		defaultProfile:  &seccompFooProfile,
	}
)

func TestNewStrategy(t *testing.T) {
	seccompFooProfile := "foo"

	tests := map[string]struct {
		strategy                      testStrategy
		expectedAllowedProfilesString string
		expectedAllowAny              bool
		expectedAllowedProfiles       map[string]bool
		expectedDefaultProfile        *string
	}{
		"no seccomp": {
			strategy:                      withoutSeccomp,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "",
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"allow any, no default": {
			strategy:                      allowAnyNoDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"allow any, default": {
			strategy:                      allowAnyDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        &seccompFooProfile,
		},
		"allow any and specific, default": {
			strategy:                      allowAnyAndSpecificDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: "bar, " + SeccompAllowAny,
			expectedAllowedProfiles: map[string]bool{
				"bar": true,
			},
			expectedDefaultProfile: &seccompFooProfile,
		},
		"allow multiple specific, no default": {
			strategy:                      allowMultipleNoDefault,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "foo, bar",
			expectedAllowedProfiles: map[string]bool{
				"foo": true,
				"bar": true,
			},
		},
		"allow multiple specific, default": {
			strategy:                      allowMultipleDefault,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "foo, bar",
			expectedAllowedProfiles: map[string]bool{
				"foo": true,
				"bar": true,
			},
			expectedDefaultProfile: &seccompFooProfile,
		},
	}
	for k, v := range tests {
		s := NewSeccompStrategy(v.strategy.defaultProfile, v.strategy.allowedProfiles)
		internalStrat, _ := s.(*seccompStrategy)

		if internalStrat.allowAnyProfile != v.expectedAllowAny {
			t.Errorf("%s expected allowAnyProfile to be %t but found %t", k, v.expectedAllowAny, internalStrat.allowAnyProfile)
		}
		if internalStrat.allowedProfilesString != v.expectedAllowedProfilesString {
			t.Errorf("%s expected allowedProfilesString to be %s but found %s", k, v.expectedAllowedProfilesString, internalStrat.allowedProfilesString)
		}
		if v.expectedDefaultProfile != nil {
			if internalStrat.defaultProfile == nil {
				t.Errorf("%s expected defaultProfile to be %s but found <nil>", k, *v.expectedDefaultProfile)
			} else if *internalStrat.defaultProfile != *v.expectedDefaultProfile {
				t.Errorf("%s expected defaultProfile to be %s but found %s", k, *v.expectedDefaultProfile, *internalStrat.defaultProfile)
			}
		}
		if v.expectedDefaultProfile == nil && internalStrat.defaultProfile != nil {
			t.Errorf("%s expected defaultProfile to be <nil> but found %s", k, *internalStrat.defaultProfile)
		}
		if !reflect.DeepEqual(v.expectedAllowedProfiles, internalStrat.allowedProfiles) {
			t.Errorf("%s expected expectedAllowedProfiles to be %#v but found %#v", k, v.expectedAllowedProfiles, internalStrat.allowedProfiles)
		}
	}
}

func TestGenerate(t *testing.T) {
	seccompBarProfile := "bar"

	tests := map[string]struct {
		pspStrategy       testStrategy
		podSeccompProfile *string
		expectedProfile   *string
	}{
		"no seccomp, no pod annotations": {
			pspStrategy:       withoutSeccomp,
			podSeccompProfile: nil,
			expectedProfile:   nil,
		},
		"no seccomp, pod annotations": {
			pspStrategy:       withoutSeccomp,
			podSeccompProfile: &seccompFooProfile,
			expectedProfile:   &seccompFooProfile,
		},
		"seccomp with no default, no pod annotations": {
			pspStrategy:       allowAnyNoDefault,
			podSeccompProfile: nil,
			expectedProfile:   nil,
		},
		"seccomp with no default, pod annotations": {
			pspStrategy:       allowAnyNoDefault,
			podSeccompProfile: &seccompFooProfile,
			expectedProfile:   &seccompFooProfile,
		},
		"seccomp with default, no pod annotations": {
			pspStrategy:       allowAnyDefault,
			podSeccompProfile: nil,
			expectedProfile:   &seccompFooProfile,
		},
		"seccomp with default, pod annotations": {
			pspStrategy:       allowAnyDefault,
			podSeccompProfile: &seccompBarProfile,
			expectedProfile:   &seccompBarProfile,
		},
	}
	for k, v := range tests {
		s := NewSeccompStrategy(v.pspStrategy.defaultProfile, v.pspStrategy.allowedProfiles)
		actual, err := s.Generate(nil, nil)
		if err != nil {
			t.Errorf("%s received error during generation %#v", k, err)
			continue
		}
		if actual != v.expectedProfile {
			t.Errorf("%s expected profile %s but received %s", k, *v.expectedProfile, *actual)
		}
	}
}

func TestValidate(t *testing.T) {
	seccompFooProfile := "foo"
	seccompBarProfile := "bar"

	tests := map[string]struct {
		pspSpec        testStrategy
		seccompProfile *string
		expectedError  string
	}{
		"no pod annotations, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: nil,
			expectedError:  "Forbidden: <nil> is not an allowed seccomp profile. Valid values are foo",
		},
		"no pod annotations, no required profiles": {
			pspSpec:        withoutSeccomp,
			seccompProfile: nil,
			expectedError:  "",
		},
		"valid pod annotations, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: &seccompFooProfile,
			expectedError:  "",
		},
		"invalid pod annotations, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: &seccompBarProfile,
			expectedError:  "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"pod annotations, no required profiles": {
			pspSpec:        withoutSeccomp,
			seccompProfile: &seccompFooProfile,
			expectedError:  "Forbidden: seccomp must not be set",
		},
		"pod annotations, allow any": {
			pspSpec:        allowAnyNoDefault,
			seccompProfile: &seccompFooProfile,
			expectedError:  "",
		},
		"no pod annotations, allow any": {
			pspSpec:        allowAnyNoDefault,
			seccompProfile: nil,
			expectedError:  "",
		},
	}
	for k, v := range tests {
		s := NewSeccompStrategy(v.pspSpec.defaultProfile, v.pspSpec.allowedProfiles)
		errs := s.Validate(field.NewPath(""), v.seccompProfile)
		if v.expectedError == "" && len(errs) != 0 {
			t.Errorf("%s expected no errors but received %#v", k, errs.ToAggregate().Error())
		}
		if v.expectedError != "" && len(errs) == 0 {
			t.Errorf("%s expected error %s but received none", k, v.expectedError)
		}
		if v.expectedError != "" && len(errs) > 1 {
			t.Errorf("%s received multiple errors: %s", k, errs.ToAggregate().Error())
		}
		if v.expectedError != "" && len(errs) == 1 && !strings.Contains(errs.ToAggregate().Error(), v.expectedError) {
			t.Errorf("%s expected error %s but received %s", k, v.expectedError, errs.ToAggregate().Error())
		}
	}
}
