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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
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
	allowSpecificLocalhost = map[string]string{
		AllowedProfilesAnnotationKey: v1.SeccompLocalhostProfileNamePrefix + "foo",
	}
	allowSpecificDockerDefault = map[string]string{
		AllowedProfilesAnnotationKey: v1.DeprecatedSeccompProfileDockerDefault,
	}
	allowSpecificRuntimeDefault = map[string]string{
		AllowedProfilesAnnotationKey: v1.SeccompProfileRuntimeDefault,
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
		s := NewStrategy(v.annotations)
		internalStrat, _ := s.(*strategy)

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
	bar := "bar"
	tests := map[string]struct {
		pspAnnotations  map[string]string
		podAnnotations  map[string]string
		seccompProfile  *api.SeccompProfile
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
		"seccomp with default, pod field": {
			pspAnnotations: allowAnyDefault,
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &bar,
			},
			expectedProfile: "localhost/bar",
		},
	}
	for k, v := range tests {
		s := NewStrategy(v.pspAnnotations)
		actual, err := s.Generate(v.podAnnotations, &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					SeccompProfile: v.seccompProfile,
				},
			},
		})

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
	foo := "foo"
	tests := map[string]struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		seccompProfile *api.SeccompProfile
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
		"valid pod annotations and field, required profiles": {
			pspAnnotations: allowSpecific,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: "foo",
			},
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &foo,
			},
			expectedError: "",
		},
		"valid pod field and no annotation, required profiles": {
			pspAnnotations: allowSpecific,
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &foo,
			},
			expectedError: "Forbidden: localhost/foo is not an allowed seccomp profile. Valid values are foo",
		},
		"valid pod field and no annotation, required profiles (localhost)": {
			pspAnnotations: allowSpecificLocalhost,
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &foo,
			},
			expectedError: "",
		},
		"docker/default PSP annotation automatically allows runtime/default pods": {
			pspAnnotations: allowSpecificDockerDefault,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: v1.SeccompProfileRuntimeDefault,
			},
			expectedError: "",
		},
		"runtime/default PSP annotation automatically allows docker/default pods": {
			pspAnnotations: allowSpecificRuntimeDefault,
			podAnnotations: map[string]string{
				api.SeccompPodAnnotationKey: v1.DeprecatedSeccompProfileDockerDefault,
			},
			expectedError: "",
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
		s := NewStrategy(v.pspAnnotations)
		errs := s.ValidatePod(pod)
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
	foo := "foo"
	bar := "bar"
	tests := map[string]struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		seccompProfile *api.SeccompProfile
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
		"valid container field and no annotation, required profiles": {
			pspAnnotations: allowSpecificLocalhost,
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &foo,
			},
			expectedError: "",
		},
		"invalid container field and no annotation, required profiles": {
			pspAnnotations: allowSpecificLocalhost,
			seccompProfile: &api.SeccompProfile{
				Type:             api.SeccompProfileTypeLocalhost,
				LocalhostProfile: &bar,
			},
			expectedError: "Forbidden: localhost/bar is not an allowed seccomp profile. Valid values are localhost/foo",
		},
	}
	for k, v := range tests {
		pod := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: v.podAnnotations,
			},
		}
		container := &api.Container{
			Name: "container",
			SecurityContext: &api.SecurityContext{
				SeccompProfile: v.seccompProfile,
			},
		}

		s := NewStrategy(v.pspAnnotations)
		errs := s.ValidateContainer(pod, container)
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
