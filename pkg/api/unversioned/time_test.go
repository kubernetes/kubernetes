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

package unversioned

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ghodss/yaml"
)

type TimeHolder struct {
	T Time `json:"t"`
}

func TestTimeMarshalYAML(t *testing.T) {
	cases := []struct {
		input  Time
		result string
	}{
		{Time{}, "t: null\n"},
		{Date(1998, time.May, 5, 1, 5, 5, 50, time.FixedZone("test", -4*60*60)), "t: 1998-05-05T05:05:05Z\n"},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "t: 1998-05-05T05:05:05Z\n"},
	}

	for _, c := range cases {
		input := TimeHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestTimeUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input  string
		result Time
	}{
		{"t: null\n", Time{}},
		{"t: 1998-05-05T05:05:05Z\n", Time{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}

	for _, c := range cases {
		var result TimeHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.T != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestTimeMarshalJSON(t *testing.T) {
	cases := []struct {
		input  Time
		result string
	}{
		{Time{}, "{\"t\":null}"},
		{Date(1998, time.May, 5, 5, 5, 5, 50, time.UTC), "{\"t\":\"1998-05-05T05:05:05Z\"}"},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "{\"t\":\"1998-05-05T05:05:05Z\"}"},
	}

	for _, c := range cases {
		input := TimeHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestTimeUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result Time
	}{
		{"{\"t\":null}", Time{}},
		{"{\"t\":\"1998-05-05T05:05:05Z\"}", Time{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}

	for _, c := range cases {
		var result TimeHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.T != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestTimeMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input Time
	}{
		{Time{}},
		{Date(1998, time.May, 5, 5, 5, 5, 50, time.Local).Rfc3339Copy()},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.Local).Rfc3339Copy()},
	}

	for i, c := range cases {
		input := TimeHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("%d-1: Failed to marshal input: '%v': %v", i, input, err)
		}

		var result TimeHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("%d-2: Failed to unmarshal '%+v': %v", i, string(jsonMarshalled), err)
		}

		iN, iO := input.T.Zone()
		oN, oO := result.T.Zone()
		if iN != oN || iO != oO {
			t.Errorf("%d-3: Time zones differ before and after serialization %s:%d %s:%d", i, iN, iO, oN, oO)
		}

		if input.T.UnixNano() != result.T.UnixNano() {
			t.Errorf("%d-4: Failed to marshal input '%#v': got %#v", i, input, result)
		}
	}
}
