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
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"

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

func TestAdmitHost(t *testing.T) {
	// This test only passes on systems running AppArmor with the default configuration.
	// The test should be manually run if modifying the getAppArmorFS function.
	t.Skip()

	assert.NoError(t, validateHost("docker"))
	assert.Error(t, validateHost("rkt"))
}

func TestAdmitProfile(t *testing.T) {
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
		{"runtime/default", true},
		{"baz", false}, // Missing local prefix.
		{"localhost//usr/sbin/ntpd", true},
		{"localhost/foo-bar", true},
		{"localhost/unloaded", false}, // Not loaded.
		{"localhost/", false},
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

func TestAdmitBadHost(t *testing.T) {
	hostErr := errors.New("expected host error")
	v := &validator{
		validateHostErr: hostErr,
	}

	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{"runtime/default", false},
		{"localhost/docker-default", false},
	}

	for _, test := range tests {
		result := v.Admit(&lifecycle.PodAdmitAttributes{
			Pod: getPodWithProfile(test.profile),
		})
		if test.expectValid {
			assert.True(t, result.Admit, "Pod with profile %q should be admitted", test.profile)
		} else {
			assert.False(t, result.Admit, "Pod with profile %q should be rejected", test.profile)
			assert.Equal(t, rejectReason, result.Reason, "Pod with profile %q", test.profile)
			assert.Contains(t, result.Message, hostErr.Error(), "Pod with profile %q", test.profile)
		}
	}
}

func TestAdmitValidHost(t *testing.T) {
	v := &validator{
		appArmorFS: "./testdata/",
	}

	tests := []struct {
		profile     string
		expectValid bool
	}{
		{"", true},
		{"runtime/default", true},
		{"localhost/docker-default", true},
		{"localhost/foo-container", true},
		{"localhost//usr/sbin/ntpd", true},
		{"docker-default", false},
		{"localhost/foo", false},
		{"localhost/", false},
	}

	for _, test := range tests {
		result := v.Admit(&lifecycle.PodAdmitAttributes{
			Pod: getPodWithProfile(test.profile),
		})
		if test.expectValid {
			assert.True(t, result.Admit, "Pod with profile %q should be admitted", test.profile)
		} else {
			assert.False(t, result.Admit, "Pod with profile %q should be rejected", test.profile)
			assert.Equal(t, rejectReason, result.Reason, "Pod with profile %q", test.profile)
			assert.NotEmpty(t, result.Message, "Pod with profile %q", test.profile)
		}
	}

	// Test multi-container pod.
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				"container.apparmor.security.alpha.kubernetes.io/init":  "localhost/foo-container",
				"container.apparmor.security.alpha.kubernetes.io/test1": "runtime/default",
				"container.apparmor.security.alpha.kubernetes.io/test2": "localhost/docker-default",
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
	assert.True(t, v.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}).Admit,
		"Multi-container pod should be admitted")
	for k, val := range pod.Annotations {
		pod.Annotations[k] = val + "-bad"

		result := v.Admit(&lifecycle.PodAdmitAttributes{Pod: pod})
		assert.False(t, result.Admit, "Multi-container pod with invalid profile should be rejected")
		assert.Equal(t, rejectReason, result.Reason, "Multi-container pod with invalid profile")
		assert.NotEmpty(t, result.Message, "Multi-container pod with invalid profile")

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
		"container.apparmor.security.alpha.kubernetes.io/test": profile,
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
