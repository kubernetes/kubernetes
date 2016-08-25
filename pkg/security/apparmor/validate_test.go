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
	"testing"

	"k8s.io/kubernetes/pkg/api"

	"github.com/stretchr/testify/assert"
)

func TestGetAppArmorFS(t *testing.T) {
	// This test only passes on systems running AppArmor with the default configuration.
	// The test should be manually run if modifying the getAppArmorFS function.
	t.Skip()

	const expectedPath = "/sys/kernel/security/apparmor"
	actualPath, err := getAppArmorFS()
	assert.NoError(t, err)
	assert.Equal(t, expectedPath, actualPath)
}

func TestValidateHost(t *testing.T) {
	// This test only passes on systems running AppArmor with the default configuration.
	// The test should be manually run if modifying the getAppArmorFS function.
	t.Skip()

	assert.NoError(t, validateHost("docker"))
	assert.Error(t, validateHost("rkt"))
}

func TestValidateProfile(t *testing.T) {
	loadedProfiles := map[string]bool{
		"docker-default":                                true,
		"foo-bar":                                       true,
		"baz":                                           true,
		"/usr/sbin/ntpd":                                true,
		"/usr/lib/connman/scripts/dhclient-script":      true,
		"/usr/lib/NetworkManager/nm-dhcp-client.action": true,
		"/usr/bin/evince-previewer//sanitized_helper":   true,
	}
	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{ProfileRuntimeDefault, true},
		{"baz", false}, // Missing local prefix.
		{ProfileNamePrefix + "/usr/sbin/ntpd", true},
		{ProfileNamePrefix + "foo-bar", true},
		{ProfileNamePrefix + "unloaded", false}, // Not loaded.
		{ProfileNamePrefix + "", false},
	}

	for _, test := range tests {
		err := validateProfile(test.profile, loadedProfiles)
		if test.expectValid {
			assert.NoError(t, err, "Profile %s should be valid", test.profile)
		} else {
			assert.Error(t, err, "Profile %s should not be valid", test.profile)
		}
	}
}

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
		{ProfileRuntimeDefault, false},
		{ProfileNamePrefix + "docker-default", false},
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
	v := &validator{
		appArmorFS: "./testdata/",
	}

	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{ProfileRuntimeDefault, true},
		{ProfileNamePrefix + "docker-default", true},
		{ProfileNamePrefix + "foo-container", true},
		{ProfileNamePrefix + "/usr/sbin/ntpd", true},
		{"docker-default", false},
		{ProfileNamePrefix + "foo", false},
		{ProfileNamePrefix + "", false},
	}

	for _, test := range tests {
		err := v.Validate(getPodWithProfile(test.profile))
		if test.expectValid {
			assert.NoError(t, err, "Pod with profile %q should be valid", test.profile)
		} else {
			assert.Error(t, err, "Pod with profile %q should trigger a validation error", test.profile)
		}
	}

	// Test multi-container pod.
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				ContainerAnnotationKeyPrefix + "init":  ProfileNamePrefix + "foo-container",
				ContainerAnnotationKeyPrefix + "test1": ProfileRuntimeDefault,
				ContainerAnnotationKeyPrefix + "test2": ProfileNamePrefix + "docker-default",
			},
		},
		Spec: api.PodSpec{
			InitContainers: []api.Container{
				{Name: "init"},
			},
			Containers: []api.Container{
				{Name: "test1"},
				{Name: "test2"},
				{Name: "no-profile"},
			},
		},
	}
	assert.NoError(t, v.Validate(pod), "Multi-container pod should validate")
	for k, val := range pod.Annotations {
		pod.Annotations[k] = val + "-bad"
		assert.Error(t, v.Validate(pod), "Multi-container pod with invalid profile %s:%s", k, pod.Annotations[k])
		pod.Annotations[k] = val // Restore.
	}
}

func TestParseProfileName(t *testing.T) {
	tests := []struct{ line, expected string }{
		{"foo://bar/baz (kill)", "foo://bar/baz"},
		{"foo-bar (enforce)", "foo-bar"},
		{"/usr/foo/bar/baz (complain)", "/usr/foo/bar/baz"},
	}
	for _, test := range tests {
		name := parseProfileName(test.line)
		assert.Equal(t, test.expected, name, "Parsing %s", test.line)
	}
}

func getPodWithProfile(profile string) *api.Pod {
	annotations := map[string]string{
		ContainerAnnotationKeyPrefix + "test": profile,
	}
	if profile == "" {
		annotations = map[string]string{
			"foo": "bar",
		}
	}
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: annotations,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
				},
			},
		},
	}
}
