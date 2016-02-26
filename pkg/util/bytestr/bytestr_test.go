/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package bytestr

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"
)

type StringOrByteSliceHolder struct {
	SorB StringOrByteSlice `json:"val"`
}

func TestStringOrByteSliceUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result StringOrByteSlice
	}{
		{
			input:  `{"val":null}`,
			result: StringOrByteSlice([]byte(nil)),
		},
		{
			input:  `{"val":""}`,
			result: StringOrByteSlice([]byte(``)),
		},
		{
			input:  `{"val":"string:"}`,
			result: StringOrByteSlice([]byte(``)),
		},
		{
			input:  `{"val":"QUJD"}`,
			result: StringOrByteSlice([]byte(`ABC`)),
		},
		{
			input:  `{"val":"string:ABC"}`,
			result: StringOrByteSlice([]byte(`ABC`)),
		},
		{
			input:  `{"val":"ScOxdMOrcm7DonRpw7Ruw6BsaXrDpnRpw7hu"}`,
			result: StringOrByteSlice([]byte(`Iñtërnâtiônàlizætiøn`)),
		},
		{
			input:  `{"val":"string:Iñtërnâtiônàlizætiøn"}`,
			result: StringOrByteSlice([]byte(`Iñtërnâtiônàlizætiøn`)),
		},
		{
			input:  `{"val":"5Lit5paH"}`,
			result: StringOrByteSlice([]byte(`中文`)),
		},
		{
			input:  `{"val":"string:中文"}`,
			result: StringOrByteSlice([]byte(`中文`)),
		},
	}

	for i, c := range cases {
		var result StringOrByteSliceHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("%d: Failed to unmarshal input '%v': %v", i, c.input, err)
		}
		if !reflect.DeepEqual(result.SorB, c.result) {
			t.Errorf("%d: Failed to unmarshal input '%v': expected %+v, got %+v", i, c.input, c.result, result)
		}
	}
}

func TestStringOrByteSliceMarshalJSON(t *testing.T) {
	cases := []struct {
		input  StringOrByteSlice
		result string
	}{
		{
			input:  StringOrByteSlice([]byte(nil)),
			result: `{"val":null}`,
		},
		{
			input:  StringOrByteSlice([]byte(``)),
			result: `{"val":""}`,
		},
		{
			input:  StringOrByteSlice([]byte(`ABC`)),
			result: `{"val":"QUJD"}`,
		},
		{
			input:  StringOrByteSlice([]byte(`Iñtërnâtiônàlizætiøn`)),
			result: `{"val":"ScOxdMOrcm7DonRpw7Ruw6BsaXrDpnRpw7hu"}`,
		},
		{
			input:  StringOrByteSlice([]byte(`中文`)),
			result: `{"val":"5Lit5paH"}`,
		},
	}

	for _, c := range cases {
		input := StringOrByteSliceHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %s, got %s", input, c.result, string(result))
		}
	}
}

func TestStringOrByteSliceMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input StringOrByteSlice
	}{
		{StringOrByteSlice([]byte(nil))},
		{StringOrByteSlice([]byte(``))},
		{StringOrByteSlice([]byte(`ABC`))},
		{StringOrByteSlice([]byte(`Iñtërnâtiônàlizætiøn`))},
		{StringOrByteSlice([]byte(`中文`))},
	}

	for _, c := range cases {
		input := StringOrByteSliceHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("1: Failed to marshal input: '%v': %v", input, err)
		}

		var result StringOrByteSliceHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("2: Failed to unmarshal '%+v': %v", string(jsonMarshalled), err)
		}

		if !reflect.DeepEqual(input, result) {
			t.Errorf("3: Failed to round-trip '%+v', got '%v' -> '%+v'", input, string(jsonMarshalled), result)
		}
	}
}
