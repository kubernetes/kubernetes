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

package v1

import (
	"encoding/json"
	"testing"
	"time"

	"sigs.k8s.io/yaml"
)

type DurationHolder struct {
	D Duration `json:"d"`
}

func TestDurationMarshalYAML(t *testing.T) {
	cases := []struct {
		input  Duration
		result string
	}{
		{Duration{5 * time.Second}, "d: 5s\n"},
		{Duration{2 * time.Minute}, "d: 2m0s\n"},
		{Duration{time.Hour + 3*time.Millisecond}, "d: 1h0m0.003s\n"},
	}

	for _, c := range cases {
		input := DurationHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: %q: %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: %q: expected %q, got %q", input, c.result, string(result))
		}
	}
}

func TestDurationUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input  string
		result Duration
	}{
		{"d: 0s\n", Duration{}},
		{"d: 5s\n", Duration{5 * time.Second}},
		{"d: 2m0s\n", Duration{2 * time.Minute}},
		{"d: 1h0m0.003s\n", Duration{time.Hour + 3*time.Millisecond}},

		// Units with zero values can optionally be dropped
		{"d: 2m\n", Duration{2 * time.Minute}},
		{"d: 1h0.003s\n", Duration{time.Hour + 3*time.Millisecond}},
	}

	for _, c := range cases {
		var result DurationHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input %q: %v", c.input, err)
		}
		if result.D != c.result {
			t.Errorf("Failed to unmarshal input %q: expected %q, got %q", c.input, c.result, result)
		}
	}
}

func TestDurationMarshalJSON(t *testing.T) {
	cases := []struct {
		input  Duration
		result string
	}{
		{Duration{5 * time.Second}, `{"d":"5s"}`},
		{Duration{2 * time.Minute}, `{"d":"2m0s"}`},
		{Duration{time.Hour + 3*time.Millisecond}, `{"d":"1h0m0.003s"}`},
	}

	for _, c := range cases {
		input := DurationHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: %q: %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: %q: expected %q, got %q", input, c.result, string(result))
		}
	}
}

func TestDurationUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result Duration
	}{
		{`{"d":"0s"}`, Duration{}},
		{`{"d":"5s"}`, Duration{5 * time.Second}},
		{`{"d":"2m0s"}`, Duration{2 * time.Minute}},
		{`{"d":"1h0m0.003s"}`, Duration{time.Hour + 3*time.Millisecond}},

		// Units with zero values can optionally be dropped
		{`{"d":"2m"}`, Duration{2 * time.Minute}},
		{`{"d":"1h0.003s"}`, Duration{time.Hour + 3*time.Millisecond}},
	}

	for _, c := range cases {
		var result DurationHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input %q: %v", c.input, err)
		}
		if result.D != c.result {
			t.Errorf("Failed to unmarshal input %q: expected %q, got %q", c.input, c.result, result)
		}
	}
}

func TestDurationMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input Duration
	}{
		{Duration{}},
		{Duration{5 * time.Second}},
		{Duration{2 * time.Minute}},
		{Duration{time.Hour + 3*time.Millisecond}},
	}

	for i, c := range cases {
		input := DurationHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("%d-1: Failed to marshal input: '%v': %v", i, input, err)
		}

		var result DurationHolder
		if err := yaml.Unmarshal(jsonMarshalled, &result); err != nil {
			t.Errorf("%d-2: Failed to unmarshal '%+v': %v", i, string(jsonMarshalled), err)
		}

		if input.D != result.D {
			t.Errorf("%d-4: Failed to marshal input '%#v': got %#v", i, input, result)
		}
	}
}
