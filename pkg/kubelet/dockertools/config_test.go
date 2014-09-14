/*
Copyright 2014 Google Inc. All rights reserved.

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

package dockertools

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/fsouza/go-dockerclient"
)

func TestDockerConfigJSONDecode(t *testing.T) {
	input := []byte(`{"http://foo.example.com":{"username": "foo", "password": "bar", "email": "foo@example.com"}, "http://bar.example.com":{"username": "bar", "password": "baz", "email": "bar@example.com"}}`)

	expect := dockerConfig(map[string]dockerConfigEntry{
		"http://foo.example.com": dockerConfigEntry{
			Username: "foo",
			Password: "bar",
			Email:    "foo@example.com",
		},
		"http://bar.example.com": dockerConfigEntry{
			Username: "bar",
			Password: "baz",
			Email:    "bar@example.com",
		},
	})

	var output dockerConfig
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
		expect dockerConfigEntry
		fail   bool
	}{
		// simple case, just decode the fields
		{
			input: []byte(`{"username": "foo", "password": "bar", "email": "foo@example.com"}`),
			expect: dockerConfigEntry{
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// auth field decodes to username & password
		{
			input: []byte(`{"auth": "Zm9vOmJhcg==", "email": "foo@example.com"}`),
			expect: dockerConfigEntry{
				Username: "foo",
				Password: "bar",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// auth field overrides username & password
		{
			input: []byte(`{"username": "foo", "password": "bar", "auth": "cGluZzpwb25n", "email": "foo@example.com"}`),
			expect: dockerConfigEntry{
				Username: "ping",
				Password: "pong",
				Email:    "foo@example.com",
			},
			fail: false,
		},

		// poorly-formatted auth causes failure
		{
			input: []byte(`{"auth": "pants", "email": "foo@example.com"}`),
			expect: dockerConfigEntry{
				Username: "",
				Password: "",
				Email:    "foo@example.com",
			},
			fail: true,
		},

		// invalid JSON causes failure
		{
			input: []byte(`{"email": false}`),
			expect: dockerConfigEntry{
				Username: "",
				Password: "",
				Email:    "",
			},
			fail: true,
		},
	}

	for i, tt := range tests {
		var output dockerConfigEntry
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

		// good base64 data, but no colon separating username & password
		{
			input: "cGFudHM=",
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

func TestDockerKeyringFromConfig(t *testing.T) {
	cfg := dockerConfig(map[string]dockerConfigEntry{
		"http://foo.example.com": dockerConfigEntry{
			Username: "foo",
			Password: "bar",
			Email:    "foo@example.com",
		},
		"https://bar.example.com": dockerConfigEntry{
			Username: "bar",
			Password: "baz",
			Email:    "bar@example.com",
		},
	})

	dk := newDockerKeyring()
	cfg.addToKeyring(dk)

	expect := newDockerKeyring()
	expect.add("foo.example.com",
		docker.AuthConfiguration{
			Username: "foo",
			Password: "bar",
			Email:    "foo@example.com",
		},
	)
	expect.add("bar.example.com",
		docker.AuthConfiguration{
			Username: "bar",
			Password: "baz",
			Email:    "bar@example.com",
		},
	)

	if !reflect.DeepEqual(expect, dk) {
		t.Errorf("Received unexpected output. Expected %#v, got %#v", expect, dk)
	}
}
