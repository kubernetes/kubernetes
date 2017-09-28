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

package util

import (
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"testing"
)

func TestEmptyVersion(t *testing.T) {

	ver, err := KubernetesReleaseVersion("")
	if err == nil {
		t.Error("KubernetesReleaseVersion returned successfully, but error expected")
	}
	if ver != "" {
		t.Error("KubernetesReleaseVersion returned value, expected only error")
	}
}

func TestValidVersion(t *testing.T) {
	validVersions := []string{
		"v1.3.0",
		"v1.4.0-alpha.0",
		"v1.4.5",
		"v1.4.0-beta.0",
		"v2.0.0",
		"v1.6.0-alpha.0.536+d60d9f3269288f",
		"v1.5.0-alpha.0.1078+1044b6822497da-pull",
		"v1.5.0-alpha.1.822+49b9e32fad9f32-pull-gke-gci",
		"v1.6.1+coreos.0",
	}
	for _, s := range validVersions {
		ver, err := KubernetesReleaseVersion(s)
		t.Log("Valid: ", s, ver, err)
		if err != nil {
			t.Errorf("KubernetesReleaseVersion unexpected error for version %q: %v", s, err)
		}
		if ver != s {
			t.Errorf("KubernetesReleaseVersion should return same valid version string. %q != %q", s, ver)
		}
	}
}

func TestInvalidVersion(t *testing.T) {
	invalidVersions := []string{
		"v1.3",
		"1.4",
		"b1.4.0",
		"c1.4.5+git",
		"something1.2",
	}
	for _, s := range invalidVersions {
		ver, err := KubernetesReleaseVersion(s)
		t.Log("Invalid: ", s, ver, err)
		if err == nil {
			t.Errorf("KubernetesReleaseVersion error expected for version %q, but returned successfully", s)
		}
		if ver != "" {
			t.Errorf("KubernetesReleaseVersion should return empty string in case of error. Returned %q for version %q", ver, s)
		}
	}
}

func TestValidConvenientForUserVersion(t *testing.T) {
	validVersions := []string{
		"1.4.0",
		"1.4.5+git",
		"1.6.1_coreos.0",
	}
	for _, s := range validVersions {
		ver, err := KubernetesReleaseVersion(s)
		t.Log("Valid: ", s, ver, err)
		if err != nil {
			t.Errorf("KubernetesReleaseVersion unexpected error for version %q: %v", s, err)
		}
		if ver != "v"+s {
			t.Errorf("KubernetesReleaseVersion should return semantic version string. %q vs. %q", s, ver)
		}
	}
}

func TestVersionFromNetwork(t *testing.T) {
	type T struct {
		Content       string
		Status        int
		Expected      string
		ErrorExpected bool
	}
	cases := map[string]T{
		"stable":     {"stable-1", http.StatusOK, "v1.4.6", false}, // recursive pointer to stable-1
		"stable-1":   {"v1.4.6", http.StatusOK, "v1.4.6", false},
		"stable-1.3": {"v1.3.10", http.StatusOK, "v1.3.10", false},
		"latest":     {"v1.6.0-alpha.0", http.StatusOK, "v1.6.0-alpha.0", false},
		"latest-1.3": {"v1.3.11-beta.0", http.StatusOK, "v1.3.11-beta.0", false},
		"empty":      {"", http.StatusOK, "", true},
		"garbage":    {"<?xml version='1.0'?><Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>", http.StatusOK, "", true},
		"unknown":    {"The requested URL was not found on this server.", http.StatusNotFound, "", true},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimSuffix(path.Base(r.URL.Path), ".txt")
		res, found := cases[key]
		if found {
			http.Error(w, res.Content, res.Status)
		} else {
			http.Error(w, "Unknown test case key!", http.StatusNotFound)
		}
	}))
	defer server.Close()

	kubeReleaseBucketURL = server.URL

	for k, v := range cases {
		ver, err := KubernetesReleaseVersion(k)
		t.Logf("Key: %q. Result: %q, Error: %v", k, ver, err)
		switch {
		case err != nil && !v.ErrorExpected:
			t.Errorf("KubernetesReleaseVersion: unexpected error for %q. Error: %v", k, err)
		case err == nil && v.ErrorExpected:
			t.Errorf("KubernetesReleaseVersion: error expected for key %q, but result is %q", k, ver)
		case ver != v.Expected:
			t.Errorf("KubernetesReleaseVersion: unexpected result for key %q. Expected: %q Actual: %q", k, v.Expected, ver)
		}
	}
}

func TestVersionToTag(t *testing.T) {
	type T struct {
		input    string
		expected string
	}
	cases := []T{
		// NOP
		{"", ""},
		// Official releases
		{"v1.0.0", "v1.0.0"},
		// CI or custom builds
		{"v10.1.2-alpha.1.100+0123456789abcdef+SOMETHING", "v10.1.2-alpha.1.100_0123456789abcdef_SOMETHING"},
		// random and invalid input: should return safe value
		{"v1,0!0+üñµ", "v1_0_0____"},
	}

	for _, tc := range cases {
		tag := KubernetesVersionToImageTag(tc.input)
		t.Logf("KubernetesVersionToImageTag: Input: %q. Result: %q. Expected: %q", tc.input, tag, tc.expected)
		if tag != tc.expected {
			t.Errorf("failed KubernetesVersionToImageTag: Input: %q. Result: %q. Expected: %q", tc.input, tag, tc.expected)
		}
	}
}

func TestSplitVersion(t *testing.T) {
	type T struct {
		input  string
		bucket string
		label  string
		valid  bool
	}
	cases := []T{
		// Release area
		{"v1.7.0", "https://dl.k8s.io/release", "v1.7.0", true},
		{"v1.8.0-alpha.2.1231+afabd012389d53a", "https://dl.k8s.io/release", "v1.8.0-alpha.2.1231+afabd012389d53a", true},
		{"release/v1.7.0", "https://dl.k8s.io/release", "v1.7.0", true},
		{"release/latest-1.7", "https://dl.k8s.io/release", "latest-1.7", true},
		// CI builds area, lookup actual builds at ci-cross/*.txt
		{"ci-cross/latest", "https://dl.k8s.io/ci-cross", "latest", true},
		{"ci/latest-1.7", "https://dl.k8s.io/ci-cross", "latest-1.7", true},
		// unknown label in default (release) area: splitVersion validate only areas.
		{"unknown-1", "https://dl.k8s.io/release", "unknown-1", true},
		// unknown area, not valid input.
		{"unknown/latest-1", "", "", false},
	}
	// kubeReleaseBucketURL can be overriden during network tests, thus ensure
	// it will contain value corresponding to expected outcome for this unit test
	kubeReleaseBucketURL = "https://dl.k8s.io"

	for _, tc := range cases {
		bucket, label, err := splitVersion(tc.input)
		switch {
		case err != nil && tc.valid:
			t.Errorf("splitVersion: unexpected error for %q. Error: %v", tc.input, err)
		case err == nil && !tc.valid:
			t.Errorf("splitVersion: error expected for key %q, but result is %q, %q", tc.input, bucket, label)
		case bucket != tc.bucket:
			t.Errorf("splitVersion: unexpected bucket result for key %q. Expected: %q Actual: %q", tc.input, tc.bucket, bucket)
		case label != tc.label:
			t.Errorf("splitVersion: unexpected label result for key %q. Expected: %q Actual: %q", tc.input, tc.label, label)
		}

	}
}

func TestKubernetesIsCIVersion(t *testing.T) {
	type T struct {
		input    string
		expected bool
	}
	cases := []T{
		{"", false},
		// Official releases
		{"v1.0.0", false},
		{"release/v1.0.0", false},
		// CI builds
		{"ci/latest-1", true},
		{"ci-cross/latest", true},
		{"ci/v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci-cross/v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
	}

	for _, tc := range cases {
		result := KubernetesIsCIVersion(tc.input)
		t.Logf("KubernetesIsCIVersion: Input: %q. Result: %v. Expected: %v", tc.input, result, tc.expected)
		if result != tc.expected {
			t.Errorf("failed KubernetesIsCIVersion: Input: %q. Result: %v. Expected: %v", tc.input, result, tc.expected)
		}
	}

}

// Validate KubernetesReleaseVersion but with bucket prefixes
func TestCIBuildVersion(t *testing.T) {
	type T struct {
		input    string
		expected string
		valid    bool
	}
	cases := []T{
		// Official releases
		{"v1.7.0", "v1.7.0", true},
		{"release/v1.8.0", "v1.8.0", true},
		{"1.4.0-beta.0", "v1.4.0-beta.0", true},
		{"release/0invalid", "", false},
		// CI or custom builds
		{"ci/v1.9.0-alpha.1.123+acbcbfd53bfa0a", "v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci-cross/v1.9.0-alpha.1.123+acbcbfd53bfa0a", "v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci/1.9.0-alpha.1.123+acbcbfd53bfa0a", "v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci-cross/1.9.0-alpha.1.123+acbcbfd53bfa0a", "v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci/0invalid", "", false},
	}

	for _, tc := range cases {
		ver, err := KubernetesReleaseVersion(tc.input)
		t.Logf("Input: %q. Result: %q, Error: %v", tc.input, ver, err)
		switch {
		case err != nil && tc.valid:
			t.Errorf("KubernetesReleaseVersion: unexpected error for input %q. Error: %v", tc.input, err)
		case err == nil && !tc.valid:
			t.Errorf("KubernetesReleaseVersion: error expected for input %q, but result is %q", tc.input, ver)
		case ver != tc.expected:
			t.Errorf("KubernetesReleaseVersion: unexpected result for input %q. Expected: %q Actual: %q", tc.input, tc.expected, ver)
		}
	}
}

func TestNormalizedBuildVersionVersion(t *testing.T) {
	type T struct {
		input    string
		expected string
	}
	cases := []T{
		{"v1.7.0", "v1.7.0"},
		{"v1.8.0-alpha.2.1231+afabd012389d53a", "v1.8.0-alpha.2.1231+afabd012389d53a"},
		{"1.7.0", "v1.7.0"},
		{"unknown-1", ""},
	}

	for _, tc := range cases {
		output := normalizedBuildVersion(tc.input)
		if output != tc.expected {
			t.Errorf("normalizedBuildVersion: unexpected output %q for input %q. Expected: %q", output, tc.input, tc.expected)
		}
	}
}
