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
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestMain(m *testing.M) {
	KubernetesReleaseVersion = kubernetesReleaseVersionTest
	os.Exit(m.Run())
}

func kubernetesReleaseVersionTest(version string) (string, error) {
	fetcher := func(string, time.Duration) (string, error) {
		return constants.DefaultKubernetesPlaceholderVersion.String(), nil
	}
	return kubernetesReleaseVersion(version, fetcher)
}

func TestKubernetesReleaseVersion(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		expectedOutput string
		expectedError  bool
	}{
		{
			name:           "empty input",
			input:          "",
			expectedOutput: "",
			expectedError:  true,
		},
		{
			name:           "label as input",
			input:          "stable",
			expectedOutput: normalizedBuildVersion(constants.DefaultKubernetesPlaceholderVersion.String()),
			expectedError:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			output, err := KubernetesReleaseVersion(tc.input)
			if (err != nil) != tc.expectedError {
				t.Errorf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if output != tc.expectedOutput {
				t.Errorf("expected output: %s, got: %s", tc.expectedOutput, output)
			}
		})
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
		"1.7.1",
	}
	for _, s := range validVersions {
		t.Run(s, func(t *testing.T) {
			ver, err := kubernetesReleaseVersion(s, errorFetcher)
			t.Log("Valid: ", s, ver, err)
			if err != nil {
				t.Errorf("kubernetesReleaseVersion unexpected error for version %q: %v", s, err)
			}
			if ver != s && ver != "v"+s {
				t.Errorf("kubernetesReleaseVersion should return same valid version string. %q != %q", s, ver)
			}
		})
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
		t.Run(s, func(t *testing.T) {
			ver, err := kubernetesReleaseVersion(s, errorFetcher)
			t.Log("Invalid: ", s, ver, err)
			if err == nil {
				t.Errorf("kubernetesReleaseVersion error expected for version %q, but returned successfully", s)
			}
			if ver != "" {
				t.Errorf("kubernetesReleaseVersion should return empty string in case of error. Returned %q for version %q", ver, s)
			}
		})
	}
}

func TestValidConvenientForUserVersion(t *testing.T) {
	validVersions := []string{
		"1.4.0",
		"1.4.5+git",
		"1.6.1_coreos.0",
	}
	for _, s := range validVersions {
		t.Run(s, func(t *testing.T) {
			ver, err := kubernetesReleaseVersion(s, errorFetcher)
			t.Log("Valid: ", s, ver, err)
			if err != nil {
				t.Errorf("kubernetesReleaseVersion unexpected error for version %q: %v", s, err)
			}
			if ver != "v"+s {
				t.Errorf("kubernetesReleaseVersion should return semantic version string. %q vs. %q", s, ver)
			}
		})
	}
}

func TestVersionFromNetwork(t *testing.T) {
	type T struct {
		Content              string
		Expected             string
		FetcherErrorExpected bool
		ErrorExpected        bool
	}

	currentVersion := normalizedBuildVersion(constants.CurrentKubernetesVersion.String())

	cases := map[string]T{
		"stable":          {"stable-1", "v1.4.6", false, false}, // recursive pointer to stable-1
		"stable-1":        {"v1.4.6", "v1.4.6", false, false},
		"stable-1.3":      {"v1.3.10", "v1.3.10", false, false},
		"latest":          {"v1.6.0-alpha.0", "v1.6.0-alpha.0", false, false},
		"latest-1.3":      {"v1.3.11-beta.0", "v1.3.11-beta.0", false, false},
		"latest-1.5":      {"", currentVersion, true, false}, // fallback to currentVersion on fetcher error
		"invalid-version": {"", "", false, true},             // invalid version cannot be parsed
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {

			fileFetcher := func(url string, timeout time.Duration) (string, error) {
				key := strings.TrimSuffix(path.Base(url), ".txt")
				res, found := cases[key]
				if found {
					if v.FetcherErrorExpected {
						return "error", errors.New("expected error")
					}
					return res.Content, nil
				}
				return "Unknown test case key!", errors.New("unknown test case key")
			}

			ver, err := kubernetesReleaseVersion(k, fileFetcher)
			t.Logf("Key: %q. Result: %q, Error: %v", k, ver, err)
			switch {
			case err != nil && !v.ErrorExpected:
				t.Errorf("kubernetesReleaseVersion: unexpected error for %q. Error: %+v", k, err)
			case err == nil && v.ErrorExpected:
				t.Errorf("kubernetesReleaseVersion: error expected for key %q, but result is %q", k, ver)
			case ver != v.Expected:
				t.Errorf("kubernetesReleaseVersion: unexpected result for key %q. Expected: %q Actual: %q", k, v.Expected, ver)
			}
		})
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
		t.Run(fmt.Sprintf("input:%s/expected:%s", tc.input, tc.expected), func(t *testing.T) {
			tag := KubernetesVersionToImageTag(tc.input)
			t.Logf("kubernetesVersionToImageTag: Input: %q. Result: %q. Expected: %q", tc.input, tag, tc.expected)
			if tag != tc.expected {
				t.Errorf("failed KubernetesVersionToImageTag: Input: %q. Result: %q. Expected: %q", tc.input, tag, tc.expected)
			}
		})
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
		// CI builds area
		{"ci/latest", "https://storage.googleapis.com/k8s-release-dev/ci", "latest", true},
		{"ci/latest-1.7", "https://storage.googleapis.com/k8s-release-dev/ci", "latest-1.7", true},
		// unknown label in default (release) area: splitVersion validate only areas.
		{"unknown-1", "https://dl.k8s.io/release", "unknown-1", true},
		// unknown area, not valid input.
		{"unknown/latest-1", "", "", false},
		// invalid input
		{"", "", "", false},
		{"ci/", "", "", false},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("input:%s/label:%s", tc.input, tc.label), func(t *testing.T) {
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
		})
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
		{"ci/v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci/", false},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("input:%s/expected:%t", tc.input, tc.expected), func(t *testing.T) {
			result := KubernetesIsCIVersion(tc.input)
			t.Logf("kubernetesIsCIVersion: Input: %q. Result: %v. Expected: %v", tc.input, result, tc.expected)
			if result != tc.expected {
				t.Errorf("failed KubernetesIsCIVersion: Input: %q. Result: %v. Expected: %v", tc.input, result, tc.expected)
			}
		})
	}
}

// Validate kubernetesReleaseVersion but with bucket prefixes
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
		{"ci/1.9.0-alpha.1.123+acbcbfd53bfa0a", "v1.9.0-alpha.1.123+acbcbfd53bfa0a", true},
		{"ci/0invalid", "", false},
		{"0invalid", "", false},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("input:%s/expected:%s", tc.input, tc.expected), func(t *testing.T) {

			fileFetcher := func(url string, timeout time.Duration) (string, error) {
				if tc.valid {
					return tc.expected, nil
				}
				return "Unknown test case key!", errors.New("unknown test case key")
			}

			ver, err := kubernetesReleaseVersion(tc.input, fileFetcher)
			t.Logf("Input: %q. Result: %q, Error: %v", tc.input, ver, err)
			switch {
			case err != nil && tc.valid:
				t.Errorf("kubernetesReleaseVersion: unexpected error for input %q. Error: %v", tc.input, err)
			case err == nil && !tc.valid:
				t.Errorf("kubernetesReleaseVersion: error expected for input %q, but result is %q", tc.input, ver)
			case ver != tc.expected:
				t.Errorf("kubernetesReleaseVersion: unexpected result for input %q. Expected: %q Actual: %q", tc.input, tc.expected, ver)
			}
		})
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
		t.Run(fmt.Sprintf("input:%s/expected:%s", tc.input, tc.expected), func(t *testing.T) {
			output := normalizedBuildVersion(tc.input)
			if output != tc.expected {
				t.Errorf("normalizedBuildVersion: unexpected output %q for input %q. Expected: %q", output, tc.input, tc.expected)
			}
		})
	}
}

func TestKubeadmVersion(t *testing.T) {
	type T struct {
		name         string
		input        string
		output       string
		outputError  bool
		parsingError bool
	}
	cases := []T{
		{
			name:   "valid version with label and metadata",
			input:  "v1.8.0-alpha.2.1231+afabd012389d53a",
			output: "v1.8.0-alpha.2",
		},
		{
			name:   "valid version with label and extra metadata",
			input:  "v1.8.0-alpha.2.1231+afabd012389d53a.extra",
			output: "v1.8.0-alpha.2",
		},
		{
			name:   "valid patch version with label and extra metadata",
			input:  "v1.11.3-beta.0.38+135cc4c1f47994",
			output: "v1.11.2",
		},
		{
			name:   "valid version with label extra",
			input:  "v1.8.0-alpha.2.1231",
			output: "v1.8.0-alpha.2",
		},
		{
			name:   "valid patch version with label",
			input:  "v1.9.11-beta.0",
			output: "v1.9.10",
		},
		{
			name:   "handle version with partial label",
			input:  "v1.8.0-alpha",
			output: "v1.8.0-alpha.0",
		},
		{
			name:   "handle version missing 'v'",
			input:  "1.11.0",
			output: "v1.11.0",
		},
		{
			name:   "valid version without label and metadata",
			input:  "v1.8.0",
			output: "v1.8.0",
		},
		{
			name:   "valid patch version without label and metadata",
			input:  "v1.8.2",
			output: "v1.8.2",
		},
		{
			name:         "invalid version",
			input:        "foo",
			parsingError: true,
		},
		{
			name:         "invalid version with stray dash",
			input:        "v1.9.11-",
			parsingError: true,
		},
		{
			name:         "invalid version without patch release",
			input:        "v1.9",
			parsingError: true,
		},
		{
			name:         "invalid version with label and stray dot",
			input:        "v1.8.0-alpha.2.",
			parsingError: true,
		},
		{
			name:        "invalid version with label and metadata",
			input:       "v1.8.0-alpha.2.1231+afabd012389d53a",
			output:      "v1.8.0-alpha.3",
			outputError: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			output, err := kubeadmVersion(tc.input)
			if (err != nil) != tc.parsingError {
				t.Fatalf("expected error: %v, got: %v", tc.parsingError, err != nil)
			}
			if (output != tc.output) != tc.outputError {
				t.Fatalf("expected output: %s, got: %s, for input: %s", tc.output, output, tc.input)
			}
		})
	}
}

func TestValidateStableVersion(t *testing.T) {
	type T struct {
		name          string
		remoteVersion string
		clientVersion string
		output        string
		expectedError bool
	}
	cases := []T{
		{
			name:          "valid: remote version is newer; return stable label [1]",
			remoteVersion: "v1.12.0",
			clientVersion: "v1.11.0",
			output:        "stable-1.11",
		},
		{
			name:          "valid: remote version is newer; return stable label [2]",
			remoteVersion: "v2.0.0",
			clientVersion: "v1.11.0",
			output:        "stable-1.11",
		},
		{
			name:          "valid: remote version is newer; return stable label [3]",
			remoteVersion: "v2.1.5",
			clientVersion: "v1.11.5",
			output:        "stable-1.11",
		},
		{
			name:          "valid: return the remote version as it is part of the same release",
			remoteVersion: "v1.11.5",
			clientVersion: "v1.11.0",
			output:        "v1.11.5",
		},
		{
			name:          "valid: return the same version",
			remoteVersion: "v1.11.0",
			clientVersion: "v1.11.0",
			output:        "v1.11.0",
		},
		{
			name:          "invalid: client version is empty",
			remoteVersion: "v1.12.1",
			clientVersion: "",
			expectedError: true,
		},
		{
			name:          "invalid: error parsing the remote version",
			remoteVersion: "invalid-version",
			clientVersion: "v1.12.0",
			expectedError: true,
		},
		{
			name:          "invalid: error parsing the client version",
			remoteVersion: "v1.12.0",
			clientVersion: "invalid-version",
			expectedError: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			output, err := validateStableVersion(tc.remoteVersion, tc.clientVersion)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v", tc.expectedError, err != nil)
			}
			if output != tc.output {
				t.Fatalf("expected output: %s, got: %s", tc.output, output)
			}
		})
	}
}

func errorFetcher(url string, timeout time.Duration) (string, error) {
	return "should not make internet calls", errors.Errorf("should not make internet calls, tried to request url: %s", url)
}

func TestFetchFromURL(t *testing.T) {
	tests := []struct {
		name      string
		url       string
		expected  string
		timeout   time.Duration
		code      int
		body      string
		expectErr bool
	}{
		{
			name:      "normal success",
			url:       "/normal",
			code:      http.StatusOK,
			body:      "normal response",
			expected:  "normal response",
			expectErr: false,
		},
		{
			name:      "HTTP error status",
			url:       "/error",
			code:      http.StatusBadRequest,
			body:      "bad request",
			expected:  "bad request",
			expectErr: true,
		},
		{
			name:      "Request timeout",
			url:       "/timeout",
			timeout:   time.Millisecond * 50,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.code != 0 {
					w.WriteHeader(tt.code)
				}
				if tt.body != "" {
					if _, err := w.Write([]byte(tt.body)); err != nil {
						t.Error("Write body failed.")
					}
				}
				if tt.timeout == time.Millisecond*50 {
					time.Sleep(time.Millisecond * 200)
					w.WriteHeader(http.StatusOK)
					if _, err := w.Write([]byte("Delayed response")); err != nil {
						t.Error("Write body failed.")
					}
				}
			})

			ts := httptest.NewServer(handler)
			defer ts.Close()

			url := ts.URL + tt.url
			result, err := fetchFromURL(url, tt.timeout)
			if (err != nil) != tt.expectErr {
				t.Errorf("expected error: %v, got: %v", tt.expectErr, err)
			}
			if tt.expected != result {
				t.Errorf("expected result: %q, got: %q", tt.expected, result)
			}
		})
	}
}
