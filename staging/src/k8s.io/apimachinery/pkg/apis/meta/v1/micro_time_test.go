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

	"sigs.k8s.io/yaml"
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
		{DateMicro(1998, time.May, 5, 1, 5, 5, 50, time.FixedZone("test", -4*60*60)), "t: \"1998-05-05T05:05:05.000000Z\"\n"},
		{DateMicro(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "t: \"1998-05-05T05:05:05.000000Z\"\n"},
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
		{DateMicro(1998, time.May, 5, 1, 5, 5, 1000, time.Local)},
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

func TestMicroTimeEqual(t *testing.T) {
	t1 := NewMicroTime(time.Now())
	cases := []struct {
		name   string
		x      *MicroTime
		y      *MicroTime
		result bool
	}{
		{"nil =? nil", nil, nil, true},
		{"!nil =? !nil", &t1, &t1, true},
		{"nil =? !nil", nil, &t1, false},
		{"!nil =? nil", &t1, nil, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.Equal(c.y)
			if result != c.result {
				t.Errorf("Failed equality test for '%v', '%v': expected %+v, got %+v", c.x, c.y, c.result, result)
			}
		})
	}
}

func TestMicroTimeEqualTime(t *testing.T) {
	t1 := NewMicroTime(time.Now())
	t2 := NewTime(t1.Time)
	cases := []struct {
		name   string
		x      *MicroTime
		y      *Time
		result bool
	}{
		{"nil =? nil", nil, nil, true},
		{"!nil =? !nil", &t1, &t2, true},
		{"nil =? !nil", nil, &t2, false},
		{"!nil =? nil", &t1, nil, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.EqualTime(c.y)
			if result != c.result {
				t.Errorf("Failed equality test for '%v', '%v': expected %+v, got %+v", c.x, c.y, c.result, result)
			}
		})
	}
}

func TestMicroTimeBefore(t *testing.T) {
	t1 := NewMicroTime(time.Now())
	cases := []struct {
		name string
		x    *MicroTime
		y    *MicroTime
	}{
		{"nil <? nil", nil, nil},
		{"!nil <? !nil", &t1, &t1},
		{"nil <? !nil", nil, &t1},
		{"!nil <? nil", &t1, nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.Before(c.y)
			if result {
				t.Errorf("Failed before test for '%v', '%v': expected false, got %+v", c.x, c.y, result)
			}
		})
	}
}
func TestMicroTimeBeforeTime(t *testing.T) {
	t1 := NewMicroTime(time.Now())
	t2 := NewTime(t1.Time)
	cases := []struct {
		name string
		x    *MicroTime
		y    *Time
	}{
		{"nil <? nil", nil, nil},
		{"!nil <? !nil", &t1, &t2},
		{"nil <? !nil", nil, &t2},
		{"!nil <? nil", &t1, nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.BeforeTime(c.y)
			if result {
				t.Errorf("Failed before test for '%v', '%v': expected false, got %+v", c.x, c.y, result)
			}
		})
	}
}

func TestMicroTimeIsZero(t *testing.T) {
	t1 := NewMicroTime(time.Now())
	cases := []struct {
		name   string
		x      *MicroTime
		result bool
	}{
		{"nil =? 0", nil, true},
		{"!nil =? 0", &t1, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.IsZero()
			if result != c.result {
				t.Errorf("Failed equality test for '%v': expected %+v, got %+v", c.x, c.result, result)
			}
		})
	}
}

func TestMicroTimeUnmarshalJSONAndProtoEqual(t *testing.T) {
	cases := []struct {
		name   string
		input  MicroTime
		result bool
	}{
		{"nanosecond level precision", UnixMicro(123, 123123123), true},
		{"microsecond level precision", UnixMicro(123, 123123000), true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			jsonData, err := c.input.MarshalJSON()
			if err != nil {
				t.Fatalf("Failed to marshal input to JSON: '%v': %v", c.input, err)
			}

			protoData, err := c.input.Marshal()
			if err != nil {
				t.Fatalf("Failed to marshal input to proto: '%v': %v", c.input, err)
			}

			var tJSON, tProto MicroTime
			if err = tJSON.UnmarshalJSON(jsonData); err != nil {
				t.Fatalf("Failed to unmarshal JSON: '%v': %v", jsonData, err)
			}
			if err = tProto.Unmarshal(protoData); err != nil {
				t.Fatalf("Failed to unmarshal proto: '%v': %v", protoData, err)
			}

			result := tJSON.Equal(&tProto)
			if result != c.result {
				t.Errorf("Failed equality test for '%v': expected %+v, got %+v", c.input, c.result, result)
			}
		})
	}
}

func TestMicroTimeProtoUnmarshalRaw(t *testing.T) {
	cases := []struct {
		name     string
		input    []byte
		expected MicroTime
	}{
		// input is generated by Timestamp{123, 123123123}.Marshal()
		{"nanosecond level precision", []byte{8, 123, 16, 179, 235, 218, 58}, UnixMicro(123, 123123000)},
		// input is generated by Timestamp{123, 123123000}.Marshal()
		{"microsecond level precision", []byte{8, 123, 16, 184, 234, 218, 58}, UnixMicro(123, 123123000)},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var actual MicroTime
			if err := actual.Unmarshal(c.input); err != nil {
				t.Fatalf("Failed to unmarshal proto: '%v': %v", c.input, err)
			}

			if !actual.Equal(&c.expected) {
				t.Errorf("Failed unmarshal from nanosecond-precise raw for '%v': expected %+v, got %+v", c.input, c.expected, actual)
			}
		})
	}

}
