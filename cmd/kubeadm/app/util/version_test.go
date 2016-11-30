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
		t.Error("KubernetesReleaseVersion returned succesfully, but error expected")
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
		"1.4.0",
		"1.4.5+git",
		"something1.2",
	}
	for _, s := range invalidVersions {
		ver, err := KubernetesReleaseVersion(s)
		t.Log("Invalid: ", s, ver, err)
		if err == nil {
			t.Errorf("KubernetesReleaseVersion error expected for version %q, but returned succesfully", s)
		}
		if ver != "" {
			t.Errorf("KubernetesReleaseVersion should return empty string in case of error. Returned %q for version %q", ver, s)
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
