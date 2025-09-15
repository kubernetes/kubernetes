/*
Copyright 2015 The Kubernetes Authors.

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
	gojson "encoding/json"
	"reflect"
	"testing"

	utiljson "k8s.io/apimachinery/pkg/util/json"
)

type GroupVersionHolder struct {
	GV GroupVersion `json:"val"`
}

func TestGroupVersionUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  []byte
		expect GroupVersion
	}{
		{[]byte(`{"val": "v1"}`), GroupVersion{"", "v1"}},
		{[]byte(`{"val": "apps/v1"}`), GroupVersion{"apps", "v1"}},
	}

	for _, c := range cases {
		var result GroupVersionHolder
		// test golang lib's JSON codec
		if err := gojson.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("JSON codec failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.GV, c.expect) {
			t.Errorf("JSON codec failed to unmarshal input '%s': expected %+v, got %+v", c.input, c.expect, result.GV)
		}
		// test the utiljson codec
		if err := utiljson.Unmarshal(c.input, &result); err != nil {
			t.Errorf("util/json codec failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.GV, c.expect) {
			t.Errorf("util/json codec failed to unmarshal input '%s': expected %+v, got %+v", c.input, c.expect, result.GV)
		}
	}
}

func TestGroupVersionMarshalJSON(t *testing.T) {
	cases := []struct {
		input  GroupVersion
		expect []byte
	}{
		{GroupVersion{"", "v1"}, []byte(`{"val":"v1"}`)},
		{GroupVersion{"apps", "v1"}, []byte(`{"val":"apps/v1"}`)},
	}

	for _, c := range cases {
		input := GroupVersionHolder{c.input}
		result, err := gojson.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if !reflect.DeepEqual(result, c.expect) {
			t.Errorf("Failed to marshal input '%+v': expected: %s, got: %s", input, c.expect, result)
		}
	}
}
