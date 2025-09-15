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
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestReadDockerConfigFile(t *testing.T) {
	configJSONFileName := "config.json"
	var fileInfo *os.File

	//test dockerconfig json
	inputDockerconfigJSONFile := "{ \"auths\": { \"http://foo.example.com\":{\"auth\":\"Zm9vOmJhcgo=\",\"email\":\"foo@example.com\"}}}"

	preferredPath, err := os.MkdirTemp("", "test_foo_bar_dockerconfigjson_")
	if err != nil {
		t.Fatalf("Creating tmp dir fail: %v", err)
		return
	}
	defer os.RemoveAll(preferredPath)
	absDockerConfigFileLocation, err := filepath.Abs(filepath.Join(preferredPath, configJSONFileName))
	if err != nil {
		t.Fatalf("While trying to canonicalize %s: %v", preferredPath, err)
	}

	if _, err := os.Stat(absDockerConfigFileLocation); os.IsNotExist(err) {
		//create test cfg file
		fileInfo, err = os.OpenFile(absDockerConfigFileLocation, os.O_CREATE|os.O_RDWR, 0664)
		if err != nil {
			t.Fatalf("While trying to create file %s: %v", absDockerConfigFileLocation, err)
		}
		defer fileInfo.Close()
	}

	fileInfo.WriteString(inputDockerconfigJSONFile)

	orgPreferredPath := GetPreferredDockercfgPath()
	SetPreferredDockercfgPath(preferredPath)
	defer SetPreferredDockercfgPath(orgPreferredPath)
	if _, err := ReadDockerConfigFile(); err != nil {
		t.Errorf("Getting docker config file fail : %v preferredPath : %q", err, preferredPath)
	}
}
func TestDockerConfigJsonJSONDecode(t *testing.T) {
	// Fake values for testing.
	input := []byte(`{"auths": {"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}, "http://bar.example.com":{"username": "bar", "password": "baz", "email": "bar@example.com"}}}`)

	expect := DockerConfigJSON{
		Auths: DockerConfig(map[string]DockerConfigEntry{
			"http://foo.example.com": {
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
			"http://bar.example.com": {
				Username: "bar",
				Password: "baz",
				Email:    "bar@example.com",
			},
		}),
	}

	var output DockerConfigJSON
	err := json.Unmarshal(input, &output)
	if err != nil {
		t.Errorf("Received unexpected error: %v", err)
	}

	if !reflect.DeepEqual(expect, output) {
		t.Errorf("Received unexpected output. Expected %#v, got %#v", expect, output)
	}
}

func TestDockerConfigJSONDecode(t *testing.T) {
	// Fake values for testing.
	input := []byte(`{"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}, "http://bar.example.com":{"username": "bar", "password": "baz", "email": "bar@example.com"}}`)

	expect := DockerConfig(map[string]DockerConfigEntry{
		"http://foo.example.com": {
			Username: "foo",
			Password: "bar",
			Email:    "foo@example.com",
		},
		"http://bar.example.com": {
			Username: "bar",
			Password: "baz",
			Email:    "bar@example.com",
		},
	})

	var output DockerConfig
	err := json.Unmarshal(input, &output)
	if err != nil {
		t.Errorf("Received unexpected error: %v", err)
	}

	if !reflect.DeepEqual(expect, output) {
		t.Errorf("Received unexpected output. Expected %#v, got %#v", expect, output)
	}
}

func TestDockerConfigEntryJSONDecode(t *testing.T) {
	tests := []struct {
		input  []byte
		expect DockerConfigEntry
		fail   bool
	}{
		// simple case, just decode the fields
		{
			// Fake values for testing.
			input: []byte(`{"username": "foo", "password": "bar", "email": "foo@example.com"}`),
			expect: DockerConfigEntry{
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// auth field decodes to username & password
		{
			input: []byte(`{"auth": "Zm9vOmJhcg==", "email": "foo@example.com"}`),
			expect: DockerConfigEntry{
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// auth field overrides username & password
		{
			// Fake values for testing.
			input: []byte(`{"username": "foo", "password": "bar", "auth": "cGluZzpwb25n", "email": "foo@example.com"}`),
			expect: DockerConfigEntry{
				Username: "ping",
				Password: "pong",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// poorly-formatted auth causes failure
		{
			input: []byte(`{"auth": "pants", "email": "foo@example.com"}`),
			expect: DockerConfigEntry{
				Username: "",
				Password: "",
				Email:    "foo@example.com",
			},
			fail: true,
		},

		// invalid JSON causes failure
		{
			input: []byte(`{"email": false}`),
			expect: DockerConfigEntry{
				Username: "",
				Password: "",
				Email:    "",
			},
			fail: true,
		},
	}

	for i, tt := range tests {
		var output DockerConfigEntry
		err := json.Unmarshal(tt.input, &output)
		if (err != nil) != tt.fail {
			t.Errorf("case %d: expected fail=%t, got err=%v", i, tt.fail, err)
		}

		if !reflect.DeepEqual(tt.expect, output) {
			t.Errorf("case %d: expected output %#v, got %#v", i, tt.expect, output)
		}
	}
}

func TestDecodeDockerConfigFieldAuth(t *testing.T) {
	tests := []struct {
		input    string
		username string
		password string
		fail     bool
	}{
		// auth field decodes to username & password
		{
			input:    "Zm9vOmJhcg==",
			username: "foo",
			password: "bar",
		},

		// some test as before but with field not well padded
		{
			input:    "Zm9vOmJhcg",
			username: "foo",
			password: "bar",
		},

		// some test as before but with new line characters
		{
			input:    "Zm9vOm\nJhcg==\n",
			username: "foo",
			password: "bar",
		},

		// standard encoding (with padding)
		{
			input:    base64.StdEncoding.EncodeToString([]byte("foo:bar")),
			username: "foo",
			password: "bar",
		},

		// raw encoding (without padding)
		{
			input:    base64.RawStdEncoding.EncodeToString([]byte("foo:bar")),
			username: "foo",
			password: "bar",
		},

		// the input is encoded with encodeDockerConfigFieldAuth (standard encoding)
		{
			input:    encodeDockerConfigFieldAuth("foo", "bar"),
			username: "foo",
			password: "bar",
		},

		// good base64 data, but no colon separating username & password
		{
			input: "cGFudHM=",
			fail:  true,
		},

		// only new line characters are ignored
		{
			input: "Zm9vOmJhcg== ",
			fail:  true,
		},

		// bad base64 data
		{
			input: "pants",
			fail:  true,
		},
	}

	for i, tt := range tests {
		username, password, err := decodeDockerConfigFieldAuth(tt.input)
		if (err != nil) != tt.fail {
			t.Errorf("case %d: expected fail=%t, got err=%v", i, tt.fail, err)
		}

		if tt.username != username {
			t.Errorf("case %d: expected username %q, got %q", i, tt.username, username)
		}

		if tt.password != password {
			t.Errorf("case %d: expected password %q, got %q", i, tt.password, password)
		}
	}
}

func TestDockerConfigEntryJSONCompatibleEncode(t *testing.T) {
	tests := []struct {
		input  DockerConfigEntry
		expect []byte
	}{
		// simple case, just decode the fields
		{
			// Fake values for testing.
			expect: []byte(`{"username":"foo","password":"bar","email":"foo@example.com","auth":"Zm9vOmJhcg=="}`),
			input: DockerConfigEntry{
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
		},
	}

	for i, tt := range tests {
		actual, err := json.Marshal(tt.input)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}

		if string(tt.expect) != string(actual) {
			t.Errorf("case %d: expected %v, got %v", i, string(tt.expect), string(actual))
		}
	}
}

func TestReadDockerConfigFileFromBytes(t *testing.T) {
	testCases := []struct {
		id               string
		input            []byte
		expectedCfg      DockerConfig
		errorExpected    bool
		expectedErrorMsg string
	}{
		{
			id:    "valid input, no error expected",
			input: []byte(`{"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}}`),
			expectedCfg: DockerConfig(map[string]DockerConfigEntry{
				"http://foo.example.com": {
					Username: "foo",
					Password: "bar",
					Email:    "foo@example.com",
				},
			}),
		},
		{
			id:               "invalid input, error expected",
			input:            []byte(`{"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"`),
			errorExpected:    true,
			expectedErrorMsg: "error occurred while trying to unmarshal json",
		},
	}

	for _, tc := range testCases {
		cfg, err := ReadDockerConfigFileFromBytes(tc.input)
		if err != nil && !tc.errorExpected {
			t.Fatalf("Error was not expected: %v", err)
		}
		if err != nil && tc.errorExpected {
			if !reflect.DeepEqual(err.Error(), tc.expectedErrorMsg) {
				t.Fatalf("Expected error message: `%s` got `%s`", tc.expectedErrorMsg, err.Error())
			}
		} else {
			if !reflect.DeepEqual(cfg, tc.expectedCfg) {
				t.Fatalf("expected: %v got %v", tc.expectedCfg, cfg)
			}
		}
	}
}

func TestReadDockerConfigJSONFileFromBytes(t *testing.T) {
	testCases := []struct {
		id               string
		input            []byte
		expectedCfg      DockerConfig
		errorExpected    bool
		expectedErrorMsg string
	}{
		{
			id:    "valid input, no error expected",
			input: []byte(`{"auths": {"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}, "http://bar.example.com":{"username": "bar", "password": "baz", "email": "bar@example.com"}}}`),
			expectedCfg: DockerConfig(map[string]DockerConfigEntry{
				"http://foo.example.com": {
					Username: "foo",
					Password: "bar",
					Email:    "foo@example.com",
				},
				"http://bar.example.com": {
					Username: "bar",
					Password: "baz",
					Email:    "bar@example.com",
				},
			}),
		},
		{
			id:               "invalid input, error expected",
			input:            []byte(`{"auths": {"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}, "http://bar.example.com":{"username": "bar", "password": "baz", "email": "bar@example.com"`),
			errorExpected:    true,
			expectedErrorMsg: "error occurred while trying to unmarshal json",
		},
	}

	for _, tc := range testCases {
		cfg, err := readDockerConfigJSONFileFromBytes(tc.input)
		if err != nil && !tc.errorExpected {
			t.Fatalf("Error was not expected: %v", err)
		}
		if err != nil && tc.errorExpected {
			if !reflect.DeepEqual(err.Error(), tc.expectedErrorMsg) {
				t.Fatalf("Expected error message: `%s` got `%s`", tc.expectedErrorMsg, err.Error())
			}
		} else {
			if !reflect.DeepEqual(cfg, tc.expectedCfg) {
				t.Fatalf("expected: %v got %v", tc.expectedCfg, cfg)
			}
		}
	}
}
