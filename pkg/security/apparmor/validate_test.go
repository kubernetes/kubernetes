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

package apparmor

import (
	"errors"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
)

func TestValidateBadHost(t *testing.T) {
	hostErr := errors.New("expected host error")
	v := &validator{
		validateHostErr: hostErr,
	}

	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{v1.DeprecatedAppArmorBetaProfileRuntimeDefault, false},
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + "docker-default", false},
	}

	for _, test := range tests {
		err := v.Validate(getPodWithProfile(test.profile))
		if test.expectValid {
			assert.NoError(t, err, "Pod with profile %q should be valid", test.profile)
		} else {
			assert.Equal(t, hostErr, err, "Pod with profile %q should trigger a host validation error", test.profile)
		}
	}
}

func TestValidateValidHost(t *testing.T) {
	v := &validator{}

	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{v1.DeprecatedAppArmorBetaProfileRuntimeDefault, true},
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + "docker-default", true},
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + "foo-container", true},
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + "/usr/sbin/ntpd", true},
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + "", false}, // Empty profile explicitly forbidden.
		{v1.DeprecatedAppArmorBetaProfileNamePrefix + " ", false},
	}

	for _, test := range tests {
		err := v.Validate(getPodWithProfile(test.profile))
		if test.expectValid {
			assert.NoError(t, err, "Pod with profile %q should be valid", test.profile)
		} else {
			assert.Error(t, err, fmt.Sprintf("Pod with profile %q should trigger a validation error", test.profile))
		}
	}

	// Test multi-container pod.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "init":  v1.DeprecatedAppArmorBetaProfileNamePrefix + "foo-container",
				v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "test1": v1.DeprecatedAppArmorBetaProfileRuntimeDefault,
				v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "test2": v1.DeprecatedAppArmorBetaProfileNamePrefix + "docker-default",
			},
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{Name: "init"},
			},
			Containers: []v1.Container{
				{Name: "test1"},
				{Name: "test2"},
				{Name: "no-profile"},
			},
		},
	}
	assert.NoError(t, v.Validate(pod), "Multi-container pod should validate")
}

func getPodWithProfile(profile string) *v1.Pod {
	annotations := map[string]string{
		v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "test": profile,
	}
	if profile == "" {
		annotations = map[string]string{
			"foo": "bar",
		}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: annotations,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "test",
				},
			},
		},
	}
}
