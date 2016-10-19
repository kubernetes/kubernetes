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

	"k8s.io/kubernetes/pkg/api"
)

var (
	withoutSeccomp    = map[string]string{"foo": "bar"}
	allowAnyNoDefault = map[string]string{
		AllowedProfilesAnnotationKey: "*",
	}
	allowAnyDefault = map[string]string{
		AllowedProfilesAnnotationKey: "*",
		DefaultProfileAnnotationKey:  "foo",
	}
	allowAnyAndSpecificDefault = map[string]string{
		AllowedProfilesAnnotationKey: "*,bar",
		DefaultProfileAnnotationKey:  "foo",
	}
	allowSpecific = map[string]string{
		AllowedProfilesAnnotationKey: "foo",
	}
)

func TestNewStrategy(t *testing.T) {
	tests := map[string]struct {
		annotations                   map[string]string
		expectedAllowedProfilesString string
		expectedAllowAny              bool
		expectedAllowedProfiles       map[string]bool
		expectedDefaultProfile        string
	}{
		"no seccomp": {
			annotations:                   withoutSeccomp,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "",
			expectedAllowedProfiles:       nil,
			expectedDefaultProfile:        "",
		},
		"allow any, no default": {
			annotations:                   allowAnyNoDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: "*",
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        "",
		},
		"allow any, default": {
			annotations:                   allowAnyDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: "*",
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        "foo",
		},
		"allow any and specific, default": {
			annotations:                   allowAnyAndSpecificDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: "*,bar",
			expectedAllowedProfiles: map[string]bool{
				"bar": true,
			},
			expectedDefaultProfile: "foo",
		},
	}
	for k, v := range tests {
		strat := NewStrategy(v.annotations)
		internalStrat, _ := strat.(*strategy)

		if internalStrat.allowAnyProfile != v.expectedAllowAny {
			t.Errorf("%s expected allowAnyProfile to be %t but found %t", k, v.expectedAllowAny, internalStrat.allowAnyProfile)
		}
		if internalStrat.allowedProfilesString != v.expectedAllowedProfilesString {
			t.Errorf("%s expected allowedProfilesString to be %s but found %s", k, v.expectedAllowedProfilesString, internalStrat.allowedProfilesString)
		}
		if internalStrat.defaultProfile != v.expectedDefaultProfile {
			t.Errorf("%s expected defaultProfile to be %s but found %s", k, v.expectedDefaultProfile, internalStrat.defaultProfile)
		}
		if !reflect.DeepEqual(v.expectedAllowedProfiles, internalStrat.allowedProfiles) {
			t.Errorf("%s expected expectedAllowedProfiles to be %#v but found %#v", k, v.expectedAllowedProfiles, internalStrat.allowedProfiles)
		}
	}
}

func TestGenerate(t *testing.T) {
	tests := map[string]struct {
		pspAnnotations  map[string]string
		podAnnotations  map[string]string
		expectedProfile string
	}{
		"no seccomp, no pod annotations": {
			pspAnnotations:  withoutSeccomp,
			podAnnotations:  nil,
			expectedProfile: "",
		},
		"no seccomp, pod annotations": {
			pspAnnotations: withoutSeccomp,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedProfile: "foo",
		},
		"seccomp with no default, no pod annotations": {
			pspAnnotations:  allowAnyNoDefault,
			podAnnotations:  nil,
			expectedProfile: "",
		},
		"seccomp with no default, pod annotations": {
			pspAnnotations: allowAnyNoDefault,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedProfile: "foo",
		},
		"seccomp with default, no pod annotations": {
			pspAnnotations:  allowAnyDefault,
			podAnnotations:  nil,
			expectedProfile: "foo",
		},
		"seccomp with default, pod annotations": {
			pspAnnotations: allowAnyDefault,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedProfile: "bar",
		},
	}
	for k, v := range tests {
		strat := NewStrategy(v.pspAnnotations)
		actual, err := strat.Generate(v.podAnnotations, nil)
		if err != nil {
			t.Errorf("%s received error during generation %#v", k, err)
			continue
		}
		if actual != v.expectedProfile {
			t.Errorf("%s expected profile %s but received %s", k, v.expectedProfile, actual)
		}
	}
}

func TestValidatePod(t *testing.T) {
	tests := map[string]struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		expectedError  string
	}{
		"no pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: nil,
			expectedError:  "Forbidden:  is not an allowed seccomp profile. Valid values are foo",
		},
		"no pod annotations, no required profiles": {
			pspAnnotations: withoutSeccomp,
			podAnnotations: nil,
			expectedError:  "",
		},
		"valid pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"invalid pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"pod annotations, no required profiles": {
			pspAnnotations: withoutSeccomp,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "Forbidden: seccomp may not be set",
		},
		"pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefault,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"no pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefault,
			podAnnotations: nil,
			expectedError:  "",
		},
	}
	for k, v := range tests {
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Annotations: v.podAnnotations,
			},
		}
		strat := NewStrategy(v.pspAnnotations)
		errs := strat.ValidatePod(pod)
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

func TestValidateContainer(t *testing.T) {
	tests := map[string]struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		expectedError  string
	}{
		"no pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: nil,
			expectedError:  "Forbidden:  is not an allowed seccomp profile. Valid values are foo",
		},
		"no pod annotations, no required profiles": {
			pspAnnotations: withoutSeccomp,
			podAnnotations: nil,
			expectedError:  "",
		},
		"valid pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "",
		},
		"invalid pod annotations, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"pod annotations, no required profiles": {
			pspAnnotations: withoutSeccomp,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "Forbidden: seccomp may not be set",
		},
		"pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefault,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "",
		},
		"no pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefault,
			podAnnotations: nil,
			expectedError:  "",
		},
		"container inherits valid pod annotation": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"container inherits invalid pod annotation": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
	}
	for k, v := range tests {
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Annotations: v.podAnnotations,
			},
		}
		container := &api.Container{
			Name: "container",
		}

		strat := NewStrategy(v.pspAnnotations)
		errs := strat.ValidateContainer(pod, container)
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
