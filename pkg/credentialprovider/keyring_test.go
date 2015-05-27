/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"testing"
)

func TestDockerKeyringFromBytes(t *testing.T) {
	url := "hello.kubernetes.io"
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

	creds, ok := keyring.Lookup(url + "/foo/bar")
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

func TestKeyringMiss(t *testing.T) {
	url := "hello.kubernetes.io"
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

func TestKeyringMissWithDockerHubCredentials(t *testing.T) {
	url := defaultRegistryHost
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
	url := defaultRegistryHost
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
	url := defaultRegistryHost
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
	url := defaultRegistryHost
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
				t.Errorf("Expected '%s' to be %s, got %s", imageName, expected, got)
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
func (d *testProvider) Provide() DockerConfig {
	d.Count += 1
	return DockerConfig{}
}

func TestLazyKeyring(t *testing.T) {
	provider := &testProvider{
		Count: 0,
	}
	lazy := &lazyDockerKeyring{
		Providers: []DockerConfigProvider{
			provider,
		},
	}

	if provider.Count != 0 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	lazy.Lookup("foo")
	if provider.Count != 1 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	lazy.Lookup("foo")
	if provider.Count != 2 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	lazy.Lookup("foo")
	if provider.Count != 3 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
}
