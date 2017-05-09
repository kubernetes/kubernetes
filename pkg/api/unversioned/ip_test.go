/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"encoding/json"
	"net"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"
)

type IPHolder struct {
	I IP `json:"ip,omitempty"`
}

func TestIPMarshalYAML(t *testing.T) {
	cases := []struct {
		input  IP
		result string
	}{}

	for _, c := range cases {
		input := IPHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: %q: %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: %q: expected %q, got %q", input, c.result, string(result))
		}
	}
}

func TestIPUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input       string
		result      IP
		expectError bool
	}{
		{
			input:  "ip: 192.168.0.1\n",
			result: IP{net.ParseIP("192.168.0.1")},
		},
		{
			input:  "ip: 2001:4860:0:2001::68\n",
			result: IP{net.ParseIP("2001:4860:0:2001::68")},
		},
		{
			input:       "ip: bingbangblah\n",
			expectError: true,
		},
	}

	for _, c := range cases {
		var result IPHolder
		err := yaml.Unmarshal([]byte(c.input), &result)
		if err != nil && !c.expectError {
			t.Errorf("Failed to unmarshal input %q: %v", c.input, err)
		}
		if err == nil && c.expectError {
			t.Errorf("Expected error unmarshaling yaml")
		}
		if !reflect.DeepEqual(result.I, c.result) {
			t.Errorf("Failed to unmarshal input %q: expected %q, got %q", c.input, c.result, result)
		}
	}
}

func TestIPMarshalJSON(t *testing.T) {
	cases := []struct {
		input  IP
		result string
	}{
		{
			input:  IP{net.ParseIP("192.168.0.1")},
			result: `{"ip":"192.168.0.1"}`,
		},
		{
			input:  IP{net.ParseIP("2001:4860:0:2001::68")},
			result: `{"ip":"2001:4860:0:2001::68"}`,
		},
	}

	for _, c := range cases {
		input := IPHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: %q: %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: %q: expected %q, got %q", input, c.result, string(result))
		}
	}
}

func TestIPUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input       string
		result      IP
		expectError bool
	}{
		{
			input:  `{"ip":"192.168.0.1"}`,
			result: IP{net.ParseIP("192.168.0.1")},
		},
		{
			input:  `{"ip":"2001:4860:0:2001::68"}`,
			result: IP{net.ParseIP("2001:4860:0:2001::68")},
		},
		{
			input:       `{"ip":"bingbangblah"}`,
			expectError: true,
		},
	}

	for _, c := range cases {
		var result IPHolder
		err := json.Unmarshal([]byte(c.input), &result)
		if err != nil && !c.expectError {
			t.Errorf("Failed to unmarshal input %q: %v", c.input, err)
		}
		if err == nil && c.expectError {
			t.Errorf("Expected error unmarshaling json")
		}

		if !reflect.DeepEqual(result.I, c.result) {
			t.Errorf("Failed to unmarshal input %q: expected %q, got %q", c.input, c.result, result)
		}
	}
}
