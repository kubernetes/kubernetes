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

package v1

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/ghodss/yaml"
)

type MicroTimeHolder struct {
	T MicroTime `json:"t"`
}

func TestMicroTimeMarshalYAML(t *testing.T) {
	cases := []struct {
		input  MicroTime
		result string
	}{
		{MicroTime{}, "t: null\n"},
		{DateMicro(1998, time.May, 5, 1, 5, 5, 50, time.FixedZone("test", -4*60*60)), "t: 1998-05-05T05:05:05.000000Z\n"},
		{DateMicro(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "t: 1998-05-05T05:05:05.000000Z\n"},
	}

	for _, c := range cases {
		input := MicroTimeHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestMicroTimeUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input  string
		result MicroTime
	}{
		{"t: null\n", MicroTime{}},
		{"t: 1998-05-05T05:05:05.000000Z\n", MicroTime{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}

	for _, c := range cases {
		var result MicroTimeHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.T != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestMicroTimeMarshalJSON(t *testing.T) {
	cases := []struct {
		input  MicroTime
		result string
	}{
		{MicroTime{}, "{\"t\":null}"},
		{DateMicro(1998, time.May, 5, 5, 5, 5, 50, time.UTC), "{\"t\":\"1998-05-05T05:05:05.000000Z\"}"},
		{DateMicro(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "{\"t\":\"1998-05-05T05:05:05.000000Z\"}"},
	}

	for _, c := range cases {
		input := MicroTimeHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestMicroTimeUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result MicroTime
	}{
		{"{\"t\":null}", MicroTime{}},
		{"{\"t\":\"1998-05-05T05:05:05.000000Z\"}", MicroTime{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}

	for _, c := range cases {
		var result MicroTimeHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.T != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestMicroTimeProto(t *testing.T) {
	cases := []struct {
		input MicroTime
	}{
		{MicroTime{}},
		{DateMicro(1998, time.May, 5, 1, 5, 5, 50, time.Local)},
		{DateMicro(1998, time.May, 5, 5, 5, 5, 0, time.Local)},
	}

	for _, c := range cases {
		input := c.input
		data, err := input.Marshal()
		if err != nil {
			t.Fatalf("Failed to marshal input: '%v': %v", input, err)
		}
		time := MicroTime{}
		if err := time.Unmarshal(data); err != nil {
			t.Fatalf("Failed to unmarshal output: '%v': %v", input, err)
		}
		if !reflect.DeepEqual(input, time) {
			t.Errorf("Marshal->Unmarshal is not idempotent: '%v' vs '%v'", input, time)
		}
	}
}
