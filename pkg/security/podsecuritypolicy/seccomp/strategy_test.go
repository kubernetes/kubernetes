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

	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type testStrategy struct {
	defaultProfile  *string
	allowedProfiles []string
}

var (
	// Field-specific strategies
	seccompFooProfile = "foo"
	seccompBarProfile = "bar"
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

	// Annotations-defined strategies (DEPRECATED)
	withoutSeccompAnn    = map[string]string{}
	allowAnyNoDefaultAnn = map[string]string{
		AllowedProfilesAnnotationKey: "*",
	}
	allowAnyDefaultAnn = map[string]string{
		AllowedProfilesAnnotationKey: "*",
		DefaultProfileAnnotationKey:  "foo",
	}
	allowAnyAndSpecificDefaultAnn = map[string]string{
		AllowedProfilesAnnotationKey: "*,bar",
		DefaultProfileAnnotationKey:  "foo",
	}
	allowSpecificAnn = map[string]string{
		AllowedProfilesAnnotationKey: "foo",
	}
)

func TestNewStrategy(t *testing.T) {
	tests := map[string]struct {
		annotations                   map[string]string
		strategy                      testStrategy
		expectedAllowedProfilesString string
		expectedAllowAny              bool
		expectedAllowedProfiles       map[string]bool
		expectedDefaultProfile        *string
	}{
		"fields - no seccomp": {
			strategy:                      withoutSeccomp,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "",
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"fields - allow any, no default": {
			strategy:                      allowAnyNoDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"fields - allow any, default": {
			strategy:                      allowAnyDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        &seccompFooProfile,
		},
		"fields - allow any and specific, default": {
			strategy:                      allowAnyAndSpecificDefault,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: "bar," + SeccompAllowAny,
			expectedAllowedProfiles: map[string]bool{
				"bar": true,
			},
			expectedDefaultProfile: &seccompFooProfile,
		},
		"annotations - no seccomp": {
			annotations:                   withoutSeccompAnn,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: "",
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"annotations - allow any, no default": {
			annotations:                   allowAnyNoDefaultAnn,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        nil,
		},
		"annotations - allow any, default": {
			annotations:                   allowAnyDefaultAnn,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny,
			expectedAllowedProfiles:       map[string]bool{},
			expectedDefaultProfile:        &seccompFooProfile,
		},
		"annotations - allow any and specific, default": {
			annotations:                   allowAnyAndSpecificDefaultAnn,
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny + ",bar",
			expectedAllowedProfiles: map[string]bool{
				"bar": true,
			},
			expectedDefaultProfile: &seccompFooProfile,
		},
		"mix - ignore allowed profiles annotation if the field are set": {
			annotations:                   allowAnyAndSpecificDefaultAnn,
			strategy:                      allowSpecificNoDefault,
			expectedAllowAny:              false,
			expectedAllowedProfilesString: seccompFooProfile,
			expectedAllowedProfiles: map[string]bool{
				"foo": true,
			},
			expectedDefaultProfile: &seccompFooProfile,
		},
		"mix - ignore default profile annotation if the field are set": {
			annotations: allowAnyAndSpecificDefaultAnn,
			strategy: testStrategy{
				defaultProfile: &seccompBarProfile,
			},
			expectedAllowAny:              true,
			expectedAllowedProfilesString: SeccompAllowAny + ",bar",
			expectedAllowedProfiles: map[string]bool{
				"bar": true,
			},
			expectedDefaultProfile: &seccompBarProfile,
		},
	}
	for k, v := range tests {
		policy := &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.annotations,
			},
			Spec: policy.PodSecurityPolicySpec{
				DefaultSeccompProfile:  v.strategy.defaultProfile,
				AllowedSeccompProfiles: v.strategy.allowedProfiles,
			},
		}
		s := NewSeccompStrategy(policy)
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
	tests := map[string]struct {
		pspAnnotations    map[string]string
		podAnnotations    map[string]string
		pspStrategy       testStrategy
		podSeccompProfile *string
		containerProfile  *string
		expectedProfile   *string
	}{
		"fields - no seccomp, no pod fields": {
			pspStrategy:       withoutSeccomp,
			podSeccompProfile: nil,
			expectedProfile:   nil,
		},
		"fields - no seccomp, pod fields": {
			pspStrategy:       withoutSeccomp,
			podSeccompProfile: &seccompFooProfile,
			expectedProfile:   &seccompFooProfile,
		},
		"fields - no seccomp, pod fields, container fields": {
			pspStrategy:       withoutSeccomp,
			podSeccompProfile: &seccompFooProfile,
			containerProfile:  &seccompBarProfile,
			expectedProfile:   &seccompBarProfile,
		},
		"fields - seccomp with no default, no pod fields": {
			pspStrategy:       allowAnyNoDefault,
			podSeccompProfile: nil,
			expectedProfile:   nil,
		},
		"fields - seccomp with no default, pod fields": {
			pspStrategy:       allowAnyNoDefault,
			podSeccompProfile: &seccompFooProfile,
			expectedProfile:   &seccompFooProfile,
		},
		"fields - seccomp with default, no pod fields": {
			pspStrategy:       allowAnyDefault,
			podSeccompProfile: nil,
			expectedProfile:   &seccompFooProfile,
		},
		"fields - seccomp with default, pod fields": {
			pspStrategy:       allowAnyDefault,
			podSeccompProfile: &seccompBarProfile,
			expectedProfile:   &seccompBarProfile,
		},
		"fields - seccomp with default, container fields": {
			pspStrategy:      allowAnyDefault,
			containerProfile: &seccompBarProfile,
			expectedProfile:  &seccompBarProfile,
		},
		"annotations - no seccomp, no pod annotations": {
			pspAnnotations:  withoutSeccompAnn,
			podAnnotations:  nil,
			expectedProfile: nil,
		},
		"annotations - no seccomp, pod annotations": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedProfile: &seccompFooProfile,
		},
		"annotations - no seccomp, pod and container annotations": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey:                           "foo",
				api.SeccompContainerAnnotationKeyPrefix + "container": "bar",
			},
			expectedProfile: &seccompBarProfile,
		},
		"annotations - seccomp with no default, no pod annotations": {
			pspAnnotations:  allowAnyNoDefaultAnn,
			podAnnotations:  nil,
			expectedProfile: nil,
		},
		"annotations - seccomp with no default, pod annotations": {
			pspAnnotations: allowAnyNoDefaultAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedProfile: &seccompFooProfile,
		},
		"annotations - seccomp with default, no pod annotations": {
			pspAnnotations:  allowAnyDefaultAnn,
			podAnnotations:  nil,
			expectedProfile: &seccompFooProfile,
		},
		"annotations - seccomp with default, pod annotations": {
			pspAnnotations: allowAnyDefaultAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedProfile: &seccompBarProfile,
		},
		"annotations - seccomp with default, container annotations": {
			pspAnnotations: allowAnyDefaultAnn,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "bar",
			},
			expectedProfile: &seccompBarProfile,
		},
	}
	for k, v := range tests {
		// FIXME: missing cases for container-local security context
		pod := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.podAnnotations,
			},
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					SeccompProfile: v.podSeccompProfile,
				},
				Containers: []api.Container{
					{
						Name: "container",
						SecurityContext: &api.SecurityContext{
							SeccompProfile: v.containerProfile,
						},
					},
				},
			},
		}
		policy := &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.pspAnnotations,
			},
			Spec: policy.PodSecurityPolicySpec{
				DefaultSeccompProfile:  v.pspStrategy.defaultProfile,
				AllowedSeccompProfiles: v.pspStrategy.allowedProfiles,
			},
		}
		s := NewSeccompStrategy(policy)
		actual, err := s.Generate(pod, &pod.Spec.Containers[0])
		if err != nil {
			t.Errorf("%s received error during generation %#v", k, err)
			continue
		}
		if v.expectedProfile != nil {
			if actual == nil {
				t.Errorf("%s expected profile to be %s but found <nil>", k, *v.expectedProfile)
			} else if *actual != *v.expectedProfile {
				t.Errorf("%s expected defaultProfile to be %s but found %s", k, *v.expectedProfile, *actual)
			}
		}
		if v.expectedProfile == nil && actual != nil {
			t.Errorf("%s expected defaultProfile to be <nil> but found %s", k, *actual)
		}
	}
}

func TestValidatePod(t *testing.T) {
	tests := map[string]struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		pspSpec        testStrategy
		seccompProfile *string
		expectedError  string
	}{
		"fields - no pod fields, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: nil,
			expectedError:  "Forbidden: <nil> is not an allowed seccomp profile. Valid values are foo",
		},
		"fields - no pod fields, no required profiles": {
			pspSpec:        withoutSeccomp,
			seccompProfile: nil,
			expectedError:  "",
		},
		"fields - valid pod fields, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: &seccompFooProfile,
			expectedError:  "",
		},
		"fields - invalid pod fields, required profiles": {
			pspSpec:        allowSpecificNoDefault,
			seccompProfile: &seccompBarProfile,
			expectedError:  "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"fields - pod fields, no required profiles": {
			pspSpec:        withoutSeccomp,
			seccompProfile: &seccompFooProfile,
			expectedError:  "Forbidden: seccomp must not be set",
		},
		"fields - pod fields, allow any": {
			pspSpec:        allowAnyNoDefault,
			seccompProfile: &seccompFooProfile,
			expectedError:  "",
		},
		"fields - no pod fields, allow any": {
			pspSpec:        allowAnyNoDefault,
			seccompProfile: nil,
			expectedError:  "",
		},
		"annotations - no pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: nil,
			expectedError:  "Forbidden: <nil> is not an allowed seccomp profile. Valid values are foo",
		},
		"annotations - no pod annotations, no required profiles": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: nil,
			expectedError:  "",
		},
		"annotations - valid pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"annotations - invalid pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"annotations - pod annotations, no required profiles": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "Forbidden: seccomp must not be set",
		},
		"annotations - pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefaultAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"annotations - no pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefaultAnn,
			podAnnotations: nil,
			expectedError:  "",
		},
	}
	for k, v := range tests {
		pod := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.podAnnotations,
			},
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					SeccompProfile: v.seccompProfile,
				},
			},
		}
		// FIXME: add fields + annotations test cases
		policy := &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.pspAnnotations,
			},
			Spec: policy.PodSecurityPolicySpec{
				DefaultSeccompProfile:  v.pspSpec.defaultProfile,
				AllowedSeccompProfiles: v.pspSpec.allowedProfiles,
			},
		}
		s := NewSeccompStrategy(policy)
		errs := s.ValidatePod(pod, nil)
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
		pspAnnotations   map[string]string
		podAnnotations   map[string]string
		pspStrategy      testStrategy
		podProfile       *string
		containerProfile *string
		expectedError    string
	}{
		"fields - no pod fields, required profiles": {
			pspStrategy:   allowSpecificNoDefault,
			podProfile:    nil,
			expectedError: "Forbidden: <nil> is not an allowed seccomp profile. Valid values are foo",
		},
		"fields - no pod fields, no required profiles": {
			pspStrategy:   withoutSeccomp,
			podProfile:    nil,
			expectedError: "",
		},
		"fields - valid container profile, required profiles": {
			pspStrategy:      allowSpecificNoDefault,
			containerProfile: &seccompFooProfile,
			expectedError:    "",
		},
		"fields - invalid container profile, required profiles": {
			pspStrategy:      allowSpecificNoDefault,
			containerProfile: &seccompBarProfile,
			expectedError:    "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"fields - container profile, no required profiles": {
			pspStrategy:      withoutSeccomp,
			containerProfile: &seccompFooProfile,
			expectedError:    "Forbidden: seccomp must not be set",
		},
		"fields - container profile, allow any": {
			pspStrategy:      allowAnyNoDefault,
			containerProfile: &seccompFooProfile,
			expectedError:    "",
		},
		"fields - no container profile, allow any": {
			pspStrategy:   allowAnyNoDefault,
			podProfile:    nil,
			expectedError: "",
		},
		"fields - container inherits valid pod profile": {
			pspStrategy:   allowSpecificNoDefault,
			podProfile:    &seccompFooProfile,
			expectedError: "",
		},
		"fields - container inherits invalid pod profile": {
			pspStrategy:   allowSpecificNoDefault,
			podProfile:    &seccompBarProfile,
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"annotations - no pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: nil,
			expectedError:  "Forbidden: <nil> is not an allowed seccomp profile. Valid values are foo",
		},
		"annotations - no pod annotations, no required profiles": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: nil,
			expectedError:  "",
		},
		"annotations - valid pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "",
		},
		"annotations - invalid pod annotations, required profiles": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
		"annotations - pod annotations, no required profiles": {
			pspAnnotations: withoutSeccompAnn,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "Forbidden: seccomp must not be set",
		},
		"annotations - pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefaultAnn,
			podAnnotations: map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "container": "foo",
			},
			expectedError: "",
		},
		"annotations - no pod annotations, allow any": {
			pspAnnotations: allowAnyNoDefaultAnn,
			podAnnotations: nil,
			expectedError:  "",
		},
		"annotations - container inherits valid pod annotation": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			expectedError: "",
		},
		"annotations - container inherits invalid pod annotation": {
			pspAnnotations: allowSpecificAnn,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "bar",
			},
			expectedError: "Forbidden: bar is not an allowed seccomp profile. Valid values are foo",
		},
	}
	for k, v := range tests {
		policy := &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.pspAnnotations,
			},
			Spec: policy.PodSecurityPolicySpec{
				DefaultSeccompProfile:  v.pspStrategy.defaultProfile,
				AllowedSeccompProfiles: v.pspStrategy.allowedProfiles,
			},
		}
		pod := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.podAnnotations,
			},
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					SeccompProfile: v.podProfile,
				},
				Containers: []api.Container{
					{
						Name: "container",
						SecurityContext: &api.SecurityContext{
							SeccompProfile: v.containerProfile,
						},
					},
				},
			},
		}
		s := NewSeccompStrategy(policy)
		errs := s.ValidateContainer(pod, &pod.Spec.Containers[0], nil)
		if v.expectedError == "" && len(errs) != 0 {
			t.Errorf("'%s' expected no errors but received '%#v'", k, errs.ToAggregate().Error())
		}
		if v.expectedError != "" && len(errs) == 0 {
			t.Errorf("'%s' expected error '%s' but received none", k, v.expectedError)
		}
		if v.expectedError != "" && len(errs) > 1 {
			t.Errorf("'%s' received multiple errors: '%s'", k, errs.ToAggregate().Error())
		}
		if v.expectedError != "" && len(errs) == 1 && !strings.Contains(errs.ToAggregate().Error(), v.expectedError) {
			t.Errorf("'%s' expected error '%s' but received '%s'", k, v.expectedError, errs.ToAggregate().Error())
		}
	}
}
