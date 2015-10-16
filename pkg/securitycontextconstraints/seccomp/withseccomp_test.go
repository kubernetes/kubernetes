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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestNewWithSeccompProfile(t *testing.T) {
	tests := map[string]struct {
		allowedProfiles []string
	}{
		"empty":    {allowedProfiles: []string{}},
		"nil":      {allowedProfiles: nil},
		"wildcard": {allowedProfiles: []string{allowAnyProfile}},
		"values":   {allowedProfiles: []string{"foo", "bar", "*"}},
	}

	for k, v := range tests {
		_, err := NewWithSeccompProfile(v.allowedProfiles)

		if err != nil {
			t.Errorf("%s failed with error %v", k, err)
		}
	}
}

func TestGenerate(t *testing.T) {
	tests := map[string]struct {
		allowedProfiles []string
		expectedProfile string
	}{
		"empty allowed profiles": {
			allowedProfiles: []string{},
			expectedProfile: "",
		},
		"nil allowed profiles": {
			allowedProfiles: nil,
			expectedProfile: "",
		},
		"allow wildcard only": {
			allowedProfiles: []string{allowAnyProfile},
			expectedProfile: "",
		},
		"allow values": {
			allowedProfiles: []string{"foo", "bar"},
			expectedProfile: "foo",
		},
		"allow wildcard and values": {
			allowedProfiles: []string{"*", "foo", "bar"},
			expectedProfile: "foo",
		},
	}

	for k, v := range tests {
		strategy, err := NewWithSeccompProfile(v.allowedProfiles)
		if err != nil {
			t.Errorf("%s failed to create strategy with error %v", k, err)
			continue
		}

		actualProfile, generationError := strategy.Generate(nil)
		if generationError != nil {
			t.Errorf("%s received generation error %v", k, generationError)
			continue
		}

		if v.expectedProfile != actualProfile {
			t.Errorf("%s expected %s but received %s", k, v.expectedProfile, actualProfile)
		}
	}
}

func TestValidatePod(t *testing.T) {
	newPod := func(podProfile string) *api.Pod {
		pod := &api.Pod{}

		if podProfile != "" {
			pod.Annotations = map[string]string{
				api.SeccompPodAnnotationKey: podProfile,
			}
		}
		return pod
	}

	tests := map[string]struct {
		allowedProfiles []string
		pod             *api.Pod
		expectedMsg     string
	}{
		"empty allowed profiles, no pod profile": {
			allowedProfiles: nil,
			pod:             newPod(""),
			expectedMsg:     "",
		},
		"empty allowed profiles, pod profile": {
			allowedProfiles: nil,
			pod:             newPod("foo"),
			expectedMsg:     "seccomp may not be set",
		},
		"good pod profile": {
			allowedProfiles: []string{"foo"},
			pod:             newPod("foo"),
			expectedMsg:     "",
		},
		"bad pod profile": {
			allowedProfiles: []string{"foo"},
			pod:             newPod("bar"),
			expectedMsg:     "bar is not a valid seccomp profile",
		},
		"wildcard allows pod profile": {
			allowedProfiles: []string{"*"},
			pod:             newPod("foo"),
			expectedMsg:     "",
		},
		"wildcard allows no profile": {
			allowedProfiles: []string{"*"},
			pod:             newPod(""),
			expectedMsg:     "",
		},
	}

	for name, tc := range tests {
		strategy, err := NewWithSeccompProfile(tc.allowedProfiles)
		if err != nil {
			t.Errorf("%s failed to create strategy with error %v", name, err)
			continue
		}

		errs := strategy.ValidatePod(tc.pod)

		//should've passed but didn't
		if len(tc.expectedMsg) == 0 && len(errs) > 0 {
			t.Errorf("%s expected no errors but received %v", name, errs)
		}
		//should've failed but didn't
		if len(tc.expectedMsg) != 0 && len(errs) == 0 {
			t.Errorf("%s expected error %s but received no errors", name, tc.expectedMsg)
		}
		//failed with additional messages
		if len(tc.expectedMsg) != 0 && len(errs) > 1 {
			t.Errorf("%s expected error %s but received multiple errors: %v", name, tc.expectedMsg, errs)
		}
		//check that we got the right message
		if len(tc.expectedMsg) != 0 && len(errs) == 1 {
			if !strings.Contains(errs[0].Error(), tc.expectedMsg) {
				t.Errorf("%s expected error to contain %s but it did not: %v", name, tc.expectedMsg, errs)
			}
		}
	}
}

func TestValidateContainer(t *testing.T) {
	newPod := func(profile string) *api.Pod {
		pod := &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test",
					},
				},
			},
		}

		if profile != "" {
			pod.Annotations = map[string]string{
				api.SeccompContainerAnnotationKeyPrefix + "test": profile,
			}
		}
		return pod
	}

	tests := map[string]struct {
		allowedProfiles []string
		pod             *api.Pod
		expectedMsg     string
	}{
		"empty allowed profiles, no container profile": {
			allowedProfiles: nil,
			pod:             newPod(""),
			expectedMsg:     "",
		},
		"empty allowed profiles, container profile": {
			allowedProfiles: nil,
			pod:             newPod("foo"),
			expectedMsg:     "seccomp may not be set",
		},
		"good container profile": {
			allowedProfiles: []string{"foo"},
			pod:             newPod("foo"),
			expectedMsg:     "",
		},
		"bad container profile": {
			allowedProfiles: []string{"foo"},
			pod:             newPod("bar"),
			expectedMsg:     "bar is not a valid seccomp profile",
		},
		"wildcard allows container profile": {
			allowedProfiles: []string{"*"},
			pod:             newPod("foo"),
			expectedMsg:     "",
		},
		"wildcard allows no profile": {
			allowedProfiles: []string{"*"},
			pod:             newPod(""),
			expectedMsg:     "",
		},
	}

	for name, tc := range tests {
		strategy, err := NewWithSeccompProfile(tc.allowedProfiles)
		if err != nil {
			t.Errorf("%s failed to create strategy with error %v", name, err)
			continue
		}

		errs := strategy.ValidateContainer(tc.pod, &tc.pod.Spec.Containers[0])

		//should've passed but didn't
		if len(tc.expectedMsg) == 0 && len(errs) > 0 {
			t.Errorf("%s expected no errors but received %v", name, errs)
		}
		//should've failed but didn't
		if len(tc.expectedMsg) != 0 && len(errs) == 0 {
			t.Errorf("%s expected error %s but received no errors", name, tc.expectedMsg)
		}
		//failed with additional messages
		if len(tc.expectedMsg) != 0 && len(errs) > 1 {
			t.Errorf("%s expected error %s but received multiple errors: %v", name, tc.expectedMsg, errs)
		}
		//check that we got the right message
		if len(tc.expectedMsg) != 0 && len(errs) == 1 {
			if !strings.Contains(errs[0].Error(), tc.expectedMsg) {
				t.Errorf("%s expected error to contain %s but it did not: %v", name, tc.expectedMsg, errs)
			}
		}
	}
}
