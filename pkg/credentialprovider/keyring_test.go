/*
Copyright 2014 The Kubernetes Authors.

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

package credentialprovider

import (
	"encoding/base64"
	"fmt"
	"reflect"
	"testing"
)

func TestUrlsMatch(t *testing.T) {
	tests := []struct {
		globUrl       string
		targetUrl     string
		matchExpected bool
	}{
		// match when there is no path component
		{
			globUrl:       "*.kubernetes.io",
			targetUrl:     "prefix.kubernetes.io",
			matchExpected: true,
		},
		{
			globUrl:       "prefix.*.io",
			targetUrl:     "prefix.kubernetes.io",
			matchExpected: true,
		},
		{
			globUrl:       "prefix.kubernetes.*",
			targetUrl:     "prefix.kubernetes.io",
			matchExpected: true,
		},
		{
			globUrl:       "*-good.kubernetes.io",
			targetUrl:     "prefix-good.kubernetes.io",
			matchExpected: true,
		},
		// match with path components
		{
			globUrl:       "*.kubernetes.io/blah",
			targetUrl:     "prefix.kubernetes.io/blah",
			matchExpected: true,
		},
		{
			globUrl:       "prefix.*.io/foo",
			targetUrl:     "prefix.kubernetes.io/foo/bar",
			matchExpected: true,
		},
		// match with path components and ports
		{
			globUrl:       "*.kubernetes.io:1111/blah",
			targetUrl:     "prefix.kubernetes.io:1111/blah",
			matchExpected: true,
		},
		{
			globUrl:       "prefix.*.io:1111/foo",
			targetUrl:     "prefix.kubernetes.io:1111/foo/bar",
			matchExpected: true,
		},
		// no match when number of parts mismatch
		{
			globUrl:       "*.kubernetes.io",
			targetUrl:     "kubernetes.io",
			matchExpected: false,
		},
		{
			globUrl:       "*.*.kubernetes.io",
			targetUrl:     "prefix.kubernetes.io",
			matchExpected: false,
		},
		{
			globUrl:       "*.*.kubernetes.io",
			targetUrl:     "kubernetes.io",
			matchExpected: false,
		},
		// no match when some parts mismatch
		{
			globUrl:       "kubernetes.io",
			targetUrl:     "kubernetes.com",
			matchExpected: false,
		},
		{
			globUrl:       "k*.io",
			targetUrl:     "quay.io",
			matchExpected: false,
		},
		// no match when ports mismatch
		{
			globUrl:       "*.kubernetes.io:1234/blah",
			targetUrl:     "prefix.kubernetes.io:1111/blah",
			matchExpected: false,
		},
		{
			globUrl:       "prefix.*.io/foo",
			targetUrl:     "prefix.kubernetes.io:1111/foo/bar",
			matchExpected: false,
		},
	}
	for _, test := range tests {
		matched, _ := urlsMatchStr(test.globUrl, test.targetUrl)
		if matched != test.matchExpected {
			t.Errorf("Expected match result of %s and %s to be %t, but was %t",
				test.globUrl, test.targetUrl, test.matchExpected, matched)
		}
	}
}

func TestDockerKeyringForGlob(t *testing.T) {
	tests := []struct {
		globUrl   string
		targetUrl string
	}{
		{
			globUrl:   "https://hello.kubernetes.io",
			targetUrl: "hello.kubernetes.io",
		},
		{
			globUrl:   "https://*.docker.io",
			targetUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://prefix.*.io",
			targetUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://prefix.docker.*",
			targetUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://*.docker.io/path",
			targetUrl: "prefix.docker.io/path",
		},
		{
			globUrl:   "https://prefix.*.io/path",
			targetUrl: "prefix.docker.io/path/subpath",
		},
		{
			globUrl:   "https://prefix.docker.*/path",
			targetUrl: "prefix.docker.io/path",
		},
		{
			globUrl:   "https://*.docker.io:8888",
			targetUrl: "prefix.docker.io:8888",
		},
		{
			globUrl:   "https://prefix.*.io:8888",
			targetUrl: "prefix.docker.io:8888",
		},
		{
			globUrl:   "https://prefix.docker.*:8888",
			targetUrl: "prefix.docker.io:8888",
		},
		{
			globUrl:   "https://*.docker.io/path:1111",
			targetUrl: "prefix.docker.io/path:1111",
		},
		{
			globUrl:   "https://*.docker.io/v1/",
			targetUrl: "prefix.docker.io/path:1111",
		},
		{
			globUrl:   "https://*.docker.io/v2/",
			targetUrl: "prefix.docker.io/path:1111",
		},
		{
			globUrl:   "https://prefix.docker.*/path:1111",
			targetUrl: "prefix.docker.io/path:1111",
		},
		{
			globUrl:   "prefix.docker.io:1111",
			targetUrl: "prefix.docker.io:1111/path",
		},
		{
			globUrl:   "*.docker.io:1111",
			targetUrl: "prefix.docker.io:1111/path",
		},
	}
	for i, test := range tests {
		email := "foo@bar.baz"
		username := "foo"
		password := "bar"
		auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
		sampleDockerConfig := fmt.Sprintf(`{
   "%s": {
     "email": %q,
     "auth": %q
   }
}`, test.globUrl, email, auth)

		keyring := &BasicDockerKeyring{}
		if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
			t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
		} else {
			keyring.Add(cfg)
		}

		creds, ok := keyring.Lookup(test.targetUrl + "/foo/bar")
		if !ok {
			t.Errorf("%d: Didn't find expected URL: %s", i, test.targetUrl)
			continue
		}
		val := creds[0]

		if username != val.Username {
			t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
		}
		if password != val.Password {
			t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
		}
		if email != val.Email {
			t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
		}
	}
}

func TestKeyringMiss(t *testing.T) {
	tests := []struct {
		globUrl   string
		lookupUrl string
	}{
		{
			globUrl:   "https://hello.kubernetes.io",
			lookupUrl: "world.mesos.org/foo/bar",
		},
		{
			globUrl:   "https://*.docker.com",
			lookupUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://suffix.*.io",
			lookupUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://prefix.docker.c*",
			lookupUrl: "prefix.docker.io",
		},
		{
			globUrl:   "https://prefix.*.io/path:1111",
			lookupUrl: "prefix.docker.io/path/subpath:1111",
		},
		{
			globUrl:   "suffix.*.io",
			lookupUrl: "prefix.docker.io",
		},
	}
	for _, test := range tests {
		email := "foo@bar.baz"
		username := "foo"
		password := "bar"
		auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
		sampleDockerConfig := fmt.Sprintf(`{
   "%s": {
     "email": %q,
     "auth": %q
   }
}`, test.globUrl, email, auth)

		keyring := &BasicDockerKeyring{}
		if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
			t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
		} else {
			keyring.Add(cfg)
		}

		_, ok := keyring.Lookup(test.lookupUrl + "/foo/bar")
		if ok {
			t.Errorf("Expected not to find URL %s, but found", test.lookupUrl)
		}
	}

}

func TestKeyringMissWithDockerHubCredentials(t *testing.T) {
	url := defaultRegistryKey
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, url, email, auth)

	keyring := &BasicDockerKeyring{}
	if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
		t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
	} else {
		keyring.Add(cfg)
	}

	val, ok := keyring.Lookup("world.mesos.org/foo/bar")
	if ok {
		t.Errorf("Found unexpected credential: %+v", val)
	}
}

func TestKeyringHitWithUnqualifiedDockerHub(t *testing.T) {
	url := defaultRegistryKey
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, url, email, auth)

	keyring := &BasicDockerKeyring{}
	if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
		t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
	} else {
		keyring.Add(cfg)
	}

	creds, ok := keyring.Lookup("google/docker-registry")
	if !ok {
		t.Errorf("Didn't find expected URL: %s", url)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if username != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestKeyringHitWithUnqualifiedLibraryDockerHub(t *testing.T) {
	url := defaultRegistryKey
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, url, email, auth)

	keyring := &BasicDockerKeyring{}
	if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
		t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
	} else {
		keyring.Add(cfg)
	}

	creds, ok := keyring.Lookup("jenkins")
	if !ok {
		t.Errorf("Didn't find expected URL: %s", url)
		return
	}
	if len(creds) > 1 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if username != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestKeyringHitWithQualifiedDockerHub(t *testing.T) {
	url := defaultRegistryKey
	email := "foo@bar.baz"
	username := "foo"
	password := "bar"
	auth := base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("%s:%s", username, password)))
	sampleDockerConfig := fmt.Sprintf(`{
   "https://%s": {
     "email": %q,
     "auth": %q
   }
}`, url, email, auth)

	keyring := &BasicDockerKeyring{}
	if cfg, err := readDockerConfigFileFromBytes([]byte(sampleDockerConfig)); err != nil {
		t.Errorf("Error processing json blob %q, %v", sampleDockerConfig, err)
	} else {
		keyring.Add(cfg)
	}

	creds, ok := keyring.Lookup(url + "/google/docker-registry")
	if !ok {
		t.Errorf("Didn't find expected URL: %s", url)
		return
	}
	if len(creds) > 2 {
		t.Errorf("Got more hits than expected: %s", creds)
	}
	val := creds[0]

	if username != val.Username {
		t.Errorf("Unexpected username value, want: %s, got: %s", username, val.Username)
	}
	if password != val.Password {
		t.Errorf("Unexpected password value, want: %s, got: %s", password, val.Password)
	}
	if email != val.Email {
		t.Errorf("Unexpected email value, want: %s, got: %s", email, val.Email)
	}
}

func TestIsDefaultRegistryMatch(t *testing.T) {
	samples := []map[bool]string{
		{true: "foo/bar"},
		{true: "docker.io/foo/bar"},
		{true: "index.docker.io/foo/bar"},
		{true: "foo"},
		{false: ""},
		{false: "registry.tld/foo/bar"},
		{false: "registry:5000/foo/bar"},
		{false: "myhostdocker.io/foo/bar"},
	}
	for _, sample := range samples {
		for expected, imageName := range sample {
			if got := isDefaultRegistryMatch(imageName); got != expected {
				t.Errorf("Expected '%s' to be %t, got %t", imageName, expected, got)
			}
		}
	}
}

type testProvider struct {
	Count int
}

// Enabled implements dockerConfigProvider
func (d *testProvider) Enabled() bool {
	return true
}

// Provide implements dockerConfigProvider
func (d *testProvider) Provide(image string) DockerConfig {
	d.Count++
	return DockerConfig{}
}

func TestProvidersDockerKeyring(t *testing.T) {
	provider := &testProvider{
		Count: 0,
	}
	keyring := &providersDockerKeyring{
		Providers: []DockerConfigProvider{
			provider,
		},
	}

	if provider.Count != 0 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	keyring.Lookup("foo")
	if provider.Count != 1 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	keyring.Lookup("foo")
	if provider.Count != 2 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	keyring.Lookup("foo")
	if provider.Count != 3 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
}

func TestDockerKeyringLookup(t *testing.T) {
	ada := AuthConfig{
		Username: "ada",
		Password: "smash",
		Email:    "ada@example.com",
	}

	grace := AuthConfig{
		Username: "grace",
		Password: "squash",
		Email:    "grace@example.com",
	}

	dk := &BasicDockerKeyring{}
	dk.Add(DockerConfig{
		"bar.example.com/pong": DockerConfigEntry{
			Username: grace.Username,
			Password: grace.Password,
			Email:    grace.Email,
		},
		"bar.example.com": DockerConfigEntry{
			Username: ada.Username,
			Password: ada.Password,
			Email:    ada.Email,
		},
	})

	tests := []struct {
		image string
		match []AuthConfig
		ok    bool
	}{
		// direct match
		{"bar.example.com", []AuthConfig{ada}, true},

		// direct match deeper than other possible matches
		{"bar.example.com/pong", []AuthConfig{grace, ada}, true},

		// no direct match, deeper path ignored
		{"bar.example.com/ping", []AuthConfig{ada}, true},

		// match first part of path token
		{"bar.example.com/pongz", []AuthConfig{grace, ada}, true},

		// match regardless of sub-path
		{"bar.example.com/pong/pang", []AuthConfig{grace, ada}, true},

		// no host match
		{"example.com", []AuthConfig{}, false},
		{"foo.example.com", []AuthConfig{}, false},
	}

	for i, tt := range tests {
		match, ok := dk.Lookup(tt.image)
		if tt.ok != ok {
			t.Errorf("case %d: expected ok=%t, got %t", i, tt.ok, ok)
		}

		if !reflect.DeepEqual(tt.match, match) {
			t.Errorf("case %d: expected match=%#v, got %#v", i, tt.match, match)
		}
	}
}

// This validates that dockercfg entries with a scheme and url path are properly matched
// by images that only match the hostname.
// NOTE: the above covers the case of a more specific match trumping just hostname.
func TestIssue3797(t *testing.T) {
	rex := AuthConfig{
		Username: "rex",
		Password: "tiny arms",
		Email:    "rex@example.com",
	}

	dk := &BasicDockerKeyring{}
	dk.Add(DockerConfig{
		"https://quay.io/v1/": DockerConfigEntry{
			Username: rex.Username,
			Password: rex.Password,
			Email:    rex.Email,
		},
	})

	tests := []struct {
		image string
		match []AuthConfig
		ok    bool
	}{
		// direct match
		{"quay.io", []AuthConfig{rex}, true},

		// partial matches
		{"quay.io/foo", []AuthConfig{rex}, true},
		{"quay.io/foo/bar", []AuthConfig{rex}, true},
	}

	for i, tt := range tests {
		match, ok := dk.Lookup(tt.image)
		if tt.ok != ok {
			t.Errorf("case %d: expected ok=%t, got %t", i, tt.ok, ok)
		}

		if !reflect.DeepEqual(tt.match, match) {
			t.Errorf("case %d: expected match=%#v, got %#v", i, tt.match, match)
		}
	}
}
