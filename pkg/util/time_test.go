package util

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"gopkg.in/v1/yaml"
)

type TimeHolder struct {
	T Time `json:"t" yaml:"t"`
}

func TestTimeMarshalYAML(t *testing.T) {
	cases := []struct {
		input  Time
		result string
	}{
		{Time{}, "t: \"null\"\n"},
		{Date(1998, time.May, 5, 5, 5, 5, 50, time.UTC), "t: 1998-05-05T05:05:05Z\n"},
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
		{"t: \"null\"\n", Time{}},
		{"t: 1998-05-05T05:05:05Z\n", Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC)},
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
		{"{\"t\":\"1998-05-05T05:05:05Z\"}", Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC)},
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
		{Date(1998, time.May, 5, 5, 5, 5, 50, time.UTC).Rfc3339Copy()},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Rfc3339Copy()},
	}

	for _, c := range cases {
		input := TimeHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("1: Failed to marshal input: '%v': %v", input, err)
		}

		var result TimeHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("2: Failed to unmarshall '%+v': %v", string(jsonMarshalled), err)
		}

		if !reflect.DeepEqual(input, result) {
			t.Errorf("3: Failed to marshal input '%+v': got %+v", input, result)
		}
	}
}
