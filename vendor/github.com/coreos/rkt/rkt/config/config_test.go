// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package config

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const tstprefix = "config-test"

// tmpConfigFile is based on ioutil.Tempfile. The differences are that
// this function is simpler (no reseeding and whatnot) and, most
// importantly, it returns a file with ".json" extension.
func tmpConfigFile(prefix string) (*os.File, error) {
	dir := os.TempDir()
	idx := 0
	tries := 10000
	for i := 0; i < tries; i++ {
		name := filepath.Join(dir, fmt.Sprintf("%s%d.json", prefix, idx))
		f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
		if os.IsExist(err) {
			idx++
			continue
		}
		return f, err
	}
	return nil, fmt.Errorf("Failed to get tmpfile after %d tries", tries)
}

func TestAuthConfigFormat(t *testing.T) {
	tests := []struct {
		contents string
		expected map[string]http.Header
		fail     bool
	}{
		{"bogus contents", nil, true},
		{`{"bogus": {"foo": "bar"}}`, nil, true},
		{`{"rktKind": "foo"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "foo"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": "foo"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": []}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"]}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "foo"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic", "credentials": {}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic", "credentials": {"user": ""}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic", "credentials": {"user": "bar"}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic", "credentials": {"user": "bar", "password": ""}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "basic", "credentials": {"user": "bar", "password": "baz"}}`, map[string]http.Header{"coreos.com": {"Authorization": []string{"Basic YmFyOmJheg=="}}}, false},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "oauth"}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "oauth", "credentials": {}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "oauth", "credentials": {"token": ""}}`, nil, true},
		{`{"rktKind": "auth", "rktVersion": "v1", "domains": ["coreos.com"], "type": "oauth", "credentials": {"token": "sometoken"}}`, map[string]http.Header{"coreos.com": {"Authorization": []string{"Bearer sometoken"}}}, false},
	}
	for _, tt := range tests {
		cfg, err := getConfigFromContents(tt.contents, "auth")
		if vErr := verifyFailure(tt.fail, tt.contents, err); vErr != nil {
			t.Errorf("%v", vErr)
		} else if !tt.fail {
			result := make(map[string]http.Header)
			for k, v := range cfg.AuthPerHost {
				result[k] = v.GetHeader()
			}
			if !reflect.DeepEqual(result, tt.expected) {
				t.Error("Got unexpected results\nResult:\n", result, "\n\nExpected:\n", tt.expected)
			}
		}

		if _, err := json.Marshal(cfg); err != nil {
			t.Errorf("error marshaling config %v", err)
		}
	}
}

func TestDockerAuthConfigFormat(t *testing.T) {
	tests := []struct {
		contents string
		expected map[string]BasicCredentials
		fail     bool
	}{
		{"bogus contents", nil, true},
		{`{"bogus": {"foo": "bar"}}`, nil, true},
		{`{"rktKind": "foo"}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "foo"}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1"}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": "foo"}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": []}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"]}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"], "credentials": {}}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"], "credentials": {"user": ""}}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"], "credentials": {"user": "bar"}}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"], "credentials": {"user": "bar", "password": ""}}`, nil, true},
		{`{"rktKind": "dockerAuth", "rktVersion": "v1", "registries": ["coreos.com"], "credentials": {"user": "bar", "password": "baz"}}`, map[string]BasicCredentials{"coreos.com": {User: "bar", Password: "baz"}}, false},
	}
	for _, tt := range tests {
		cfg, err := getConfigFromContents(tt.contents, "dockerAuth")
		if vErr := verifyFailure(tt.fail, tt.contents, err); vErr != nil {
			t.Errorf("%v", vErr)
		} else if !tt.fail {
			result := cfg.DockerCredentialsPerRegistry
			if !reflect.DeepEqual(result, tt.expected) {
				t.Error("Got unexpected results\nResult:\n", result, "\n\nExpected:\n", tt.expected)
			}
		}

		if _, err := json.Marshal(cfg); err != nil {
			t.Errorf("error marshaling config %v", err)
		}
	}
}

func TestPathsConfigFormat(t *testing.T) {
	tests := []struct {
		contents string
		expected ConfigurablePaths
		fail     bool
	}{
		{"bogus contents", ConfigurablePaths{}, true},
		{`{"bogus": {"foo": "bar"}}`, ConfigurablePaths{}, true},
		{`{"rktKind": "foo"}`, ConfigurablePaths{}, true},
		{`{"rktKind": "paths", "rktVersion": "foo"}`, ConfigurablePaths{}, true},
		{`{"rktKind": "paths", "rktVersion": "v1"}`, ConfigurablePaths{}, false},
		{`{"rktKind": "paths", "rktVersion": "v1", "data": "/dir1"}`, ConfigurablePaths{DataDir: "/dir1"}, false},
		{`{"rktKind": "paths", "rktVersion": "v1", "data": "/dir1", "stage1-images": "/dir2"}`, ConfigurablePaths{DataDir: "/dir1", Stage1ImagesDir: "/dir2"}, false},
	}
	for _, tt := range tests {
		cfg, err := getConfigFromContents(tt.contents, "paths")
		if vErr := verifyFailure(tt.fail, tt.contents, err); vErr != nil {
			t.Errorf("%v", vErr)
		} else if !tt.fail {
			result := cfg.Paths
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Got unexpected results\nResult:\n%#v\n\nExpected:\n%#v", result, tt.expected)
			}
		}
	}
}

func TestStage1ConfigFormat(t *testing.T) {
	tests := []struct {
		contents string
		expected Stage1Data
		fail     bool
	}{
		{"bogus contents", Stage1Data{}, true},
		{`{"bogus": {"foo": "bar"}}`, Stage1Data{}, true},
		{`{"rktKind": "foo"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "foo"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "v1"}`, Stage1Data{}, false},
		{`{"rktKind": "stage1", "rktVersion": "v1", "name": "example.com/stage1"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "v1", "version": "1.2.3"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "v1", "name": "example.com/stage1", "version": "1.2.3"}`, Stage1Data{Name: "example.com/stage1", Version: "1.2.3"}, false},
		{`{"rktKind": "stage1", "rktVersion": "v1", "location": "/image.aci"}`, Stage1Data{Location: "/image.aci"}, false},
		{`{"rktKind": "stage1", "rktVersion": "v1", "name": "example.com/stage1", "location": "/image.aci"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "v1", "version": "1.2.3", "location": "/image.aci"}`, Stage1Data{}, true},
		{`{"rktKind": "stage1", "rktVersion": "v1", "name": "example.com/stage1", "version": "1.2.3", "location": "/image.aci"}`, Stage1Data{Name: "example.com/stage1", Version: "1.2.3", Location: "/image.aci"}, false},
	}
	for _, tt := range tests {
		cfg, err := getConfigFromContents(tt.contents, "stage1")
		if vErr := verifyFailure(tt.fail, tt.contents, err); vErr != nil {
			t.Errorf("%v", vErr)
		} else if !tt.fail {
			result := cfg.Stage1
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Got unexpected results\nResult:\n%#v\n\nExpected:\n%#v", result, tt.expected)
			}
		}
	}
}

func verifyFailure(shouldFail bool, contents string, err error) error {
	var vErr error = nil
	if err != nil {
		if !shouldFail {
			vErr = fmt.Errorf("Expected test to succeed, failed unexpectedly (contents: `%s`): %v", contents, err)
		}
	} else if shouldFail {
		vErr = fmt.Errorf("Expected test to fail, succeeded unexpectedly (contents: `%s`)", contents)
	}
	return vErr
}

func getConfigFromContents(contents, kind string) (*Config, error) {
	f, err := tmpConfigFile(tstprefix)
	if err != nil {
		panic(fmt.Sprintf("Failed to create tmp config file: %v", err))
	}
	// First remove the file, then close it (last deferred item is
	// executed first).
	defer f.Close()
	defer os.Remove(f.Name())
	if _, err := f.Write([]byte(contents)); err != nil {
		panic(fmt.Sprintf("Writing config to file failed: %v", err))
	}
	fi, err := f.Stat()
	if err != nil {
		panic(fmt.Sprintf("Stating a tmp config file failed: %v", err))
	}
	cfg := newConfig()
	return cfg, readFile(cfg, fi, f.Name(), []string{kind})
}

func TestConfigLoading(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		panic(fmt.Sprintf("Failed to create temporary directory: %v", err))
	}
	defer os.RemoveAll(dir)
	systemAuth := filepath.Join("system", "auth.d")
	systemIgnored := filepath.Join(systemAuth, "ignoreddir")
	localAuth := filepath.Join("local", "auth.d")
	localIgnored := filepath.Join(localAuth, "ignoreddir")
	dirs := []string{
		"system",
		systemAuth,
		systemIgnored,
		"local",
		localAuth,
		localIgnored,
	}
	for _, d := range dirs {
		cd := filepath.Join(dir, d)
		if err := os.Mkdir(cd, 0700); err != nil {
			panic(fmt.Sprintf("Failed to create configuration directory %q: %v", cd, err))
		}
	}
	files := []struct {
		path   string
		domain string
		user   string
		pass   string
	}{
		{filepath.Join(dir, systemAuth, "endocode.json"), "endocode.com", "system_user1", "system_password1"},
		{filepath.Join(dir, systemAuth, "coreos.json"), "coreos.com", "system_user2", "system_password2"},
		{filepath.Join(dir, systemAuth, "ignoredfile"), "example1.com", "ignored_user1", "ignored_password1"},
		{filepath.Join(dir, systemIgnored, "ignoredfile"), "example2.com", "ignored_user2", "ignored_password2"},
		{filepath.Join(dir, systemIgnored, "ignoredanyway.json"), "example3.com", "ignored_user3", "ignored_password3"},
		{filepath.Join(dir, localAuth, "endocode.json"), "endocode.com", "local_user1", "local_password1"},
		{filepath.Join(dir, localAuth, "tectonic.json"), "tectonic.com", "local_user2", "local_password2"},
		{filepath.Join(dir, localAuth, "ignoredfile"), "example4.com", "ignored_user4", "ignored_password4"},
		{filepath.Join(dir, localIgnored, "ignoredfile"), "example5.com", "ignored_user5", "ignored_password5"},
		{filepath.Join(dir, localIgnored, "ignoredanyway.json"), "example6.com", "ignored_user6", "ignored_password6"},
	}
	for _, f := range files {
		if err := writeBasicConfig(f.path, f.domain, f.user, f.pass); err != nil {
			panic(fmt.Sprintf("Failed to write configuration file: %v", err))
		}
	}
	cfg, err := GetConfigFrom(filepath.Join(dir, "system"), filepath.Join(dir, "local"))
	if err != nil {
		panic(fmt.Sprintf("Failed to get configuration: %v", err))
	}
	result := make(map[string]http.Header)
	for d, h := range cfg.AuthPerHost {
		result[d] = h.GetHeader()
	}
	expected := map[string]http.Header{
		"endocode.com": {
			// local_user1:local_password1
			authHeader: []string{"Basic bG9jYWxfdXNlcjE6bG9jYWxfcGFzc3dvcmQx"},
		},
		"coreos.com": {
			// system_user2:system_password2
			authHeader: []string{"Basic c3lzdGVtX3VzZXIyOnN5c3RlbV9wYXNzd29yZDI="},
		},
		"tectonic.com": {
			// local_user2:local_password2
			authHeader: []string{"Basic bG9jYWxfdXNlcjI6bG9jYWxfcGFzc3dvcmQy"},
		},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Error("Got unexpected results\nResult:\n", result, "\n\nExpected:\n", expected)
	}
}

func writeBasicConfig(path, domain, user, pass string) error {
	type basicv1creds struct {
		User     string `json:"user"`
		Password string `json:"password"`
	}
	type basicv1 struct {
		RktVersion  string       `json:"rktVersion"`
		RktKind     string       `json:"rktKind"`
		Domains     []string     `json:"domains"`
		Type        string       `json:"type"`
		Credentials basicv1creds `json:"credentials"`
	}
	config := &basicv1{
		RktVersion: "v1",
		RktKind:    "auth",
		Domains:    []string{domain},
		Type:       "basic",
		Credentials: basicv1creds{
			User:     user,
			Password: pass,
		},
	}
	raw, err := json.Marshal(config)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, raw, 0600)
}
