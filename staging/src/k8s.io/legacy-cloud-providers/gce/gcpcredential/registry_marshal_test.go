/*
Copyright 2021 The Kubernetes Authors.

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

package gcpcredential

import (
	"encoding/base64"
	"encoding/json"
	"k8s.io/cloud-provider/credentialconfig"
	"reflect"
	"testing"
)

// Code copied (and edited to replace DockerConfig* with RegistryConfig*) from:
// pkg/credentialprovider/config_test.go.

func TestRegistryConfigEntryJSONDecode(t *testing.T) {
	tests := []struct {
		input  []byte
		expect RegistryConfigEntry
		fail   bool
	}{
		// simple case, just decode the fields
		{
			// Fake values for testing.
			input: []byte(`{"username": "foo", "password": "bar", "email": "foo@example.com"}`),
			expect: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "foo",
					Password: "bar",
					Email:    "foo@example.com",
				},
			},
			fail: false,
		},

		// auth field decodes to username & password
		{
			input: []byte(`{"auth": "Zm9vOmJhcg==", "email": "foo@example.com"}`),
			expect: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "foo",
					Password: "bar",
					Email:    "foo@example.com",
				},
			},
			fail: false,
		},

		// auth field overrides username & password
		{
			// Fake values for testing.
			input: []byte(`{"username": "foo", "password": "bar", "auth": "cGluZzpwb25n", "email": "foo@example.com"}`),
			expect: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "ping",
					Password: "pong",
					Email:    "foo@example.com",
				},
			},
			fail: false,
		},

		// poorly-formatted auth causes failure
		{
			input: []byte(`{"auth": "pants", "email": "foo@example.com"}`),
			expect: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "",
					Password: "",
					Email:    "foo@example.com",
				},
			},
			fail: true,
		},

		// invalid JSON causes failure
		{
			input: []byte(`{"email": false}`),
			expect: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "",
					Password: "",
					Email:    "",
				},
			},
			fail: true,
		},
	}

	for i, tt := range tests {
		var output RegistryConfigEntry
		err := json.Unmarshal(tt.input, &output)
		if (err != nil) != tt.fail {
			t.Errorf("case %d: expected fail=%t, got err=%v", i, tt.fail, err)
		}

		if !reflect.DeepEqual(tt.expect, output) {
			t.Errorf("case %d: expected output %#v, got %#v", i, tt.expect, output)
		}
	}
}

func TestDecodeRegistryConfigFieldAuth(t *testing.T) {
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

		// the input is encoded with encodeRegistryConfigFieldAuth (standard encoding)
		{
			input:    encodeRegistryConfigFieldAuth("foo", "bar"),
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
		username, password, err := decodeRegistryConfigFieldAuth(tt.input)
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

func TestRegistryConfigEntryJSONCompatibleEncode(t *testing.T) {
	tests := []struct {
		input  RegistryConfigEntry
		expect []byte
	}{
		// simple case, just decode the fields
		{
			// Fake values for testing.
			expect: []byte(`{"username":"foo","password":"bar","email":"foo@example.com","auth":"Zm9vOmJhcg=="}`),
			input: RegistryConfigEntry{
				credentialconfig.RegistryConfigEntry{
					Username: "foo",
					Password: "bar",
					Email:    "foo@example.com",
				},
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
