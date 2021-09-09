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
	"bytes"
	gojson "encoding/json"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

func TestVerbsMarshalJSON(t *testing.T) {
	cases := []struct {
		input  APIResource
		result string
	}{
		{APIResource{}, `{"name":"","singularName":"","namespaced":false,"kind":"","verbs":null}`},
		{APIResource{Verbs: Verbs([]string{})}, `{"name":"","singularName":"","namespaced":false,"kind":"","verbs":[]}`},
		{APIResource{Verbs: Verbs([]string{"delete"})}, `{"name":"","singularName":"","namespaced":false,"kind":"","verbs":["delete"]}`},
	}

	for i, c := range cases {
		result, err := gojson.Marshal(&c.input)
		if err != nil {
			t.Errorf("[%d] Failed to marshal input: '%v': %v", i, c.input, err)
		}
		if string(result) != c.result {
			t.Errorf("[%d] Failed to marshal input: '%v': expected '%v', got '%v'", i, c.input, c.result, string(result))
		}
	}
}

func TestVerbsJsonIterUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result APIResource
	}{
		{`{}`, APIResource{}},
		{`{"verbs":null}`, APIResource{}},
		{`{"verbs":[]}`, APIResource{Verbs: Verbs([]string{})}},
		{`{"verbs":["delete"]}`, APIResource{Verbs: Verbs([]string{"delete"})}},
	}

	iter := json.CaseSensitiveJSONIterator()
	for i, c := range cases {
		var result APIResource
		if err := iter.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("[%d] Failed to unmarshal input '%v': %v", i, c.input, err)
		}
		if !reflect.DeepEqual(result, c.result) {
			t.Errorf("[%d] Failed to unmarshal input '%v': expected %+v, got %+v", i, c.input, c.result, result)
		}
	}
}

// TestMarshalJSONWithOmit tests that we don't have regressions regarding nil and empty slices with "omit"
func TestMarshalJSONWithOmit(t *testing.T) {
	cases := []struct {
		input  LabelSelector
		result string
	}{
		{LabelSelector{}, `{}`},
		{LabelSelector{MatchExpressions: []LabelSelectorRequirement{}}, `{}`},
		{LabelSelector{MatchExpressions: []LabelSelectorRequirement{{}}}, `{"matchExpressions":[{"key":"","operator":""}]}`},
	}

	for i, c := range cases {
		result, err := gojson.Marshal(&c.input)
		if err != nil {
			t.Errorf("[%d] Failed to marshal input: '%v': %v", i, c.input, err)
		}
		if string(result) != c.result {
			t.Errorf("[%d] Failed to marshal input: '%v': expected '%v', got '%v'", i, c.input, c.result, string(result))
		}
	}
}

func TestVerbsUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result APIResource
	}{
		{`{}`, APIResource{}},
		{`{"verbs":null}`, APIResource{}},
		{`{"verbs":[]}`, APIResource{Verbs: Verbs([]string{})}},
		{`{"verbs":["delete"]}`, APIResource{Verbs: Verbs([]string{"delete"})}},
	}

	for i, c := range cases {
		var result APIResource
		if err := gojson.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("[%d] Failed to unmarshal input '%v': %v", i, c.input, err)
		}
		if !reflect.DeepEqual(result, c.result) {
			t.Errorf("[%d] Failed to unmarshal input '%v': expected %+v, got %+v", i, c.input, c.result, result)
		}
	}
}

func TestVerbsProto(t *testing.T) {
	cases := []APIResource{
		{},
		{Verbs: Verbs([]string{})},
		{Verbs: Verbs([]string{"delete"})},
	}

	for _, input := range cases {
		data, err := input.Marshal()
		if err != nil {
			t.Fatalf("Failed to marshal input: '%v': %v", input, err)
		}
		resource := APIResource{}
		if err := resource.Unmarshal(data); err != nil {
			t.Fatalf("Failed to unmarshal output: '%v': %v", input, err)
		}
		if !reflect.DeepEqual(input, resource) {
			t.Errorf("Marshal->Unmarshal is not idempotent: '%v' vs '%v'", input, resource)
		}
	}
}

func TestCreationTimestampMarshalJSON(t *testing.T) {
	testCases := []struct {
		name string
		objectMeta ObjectMeta
		expected []byte
	}{
		{
			name: "ObjectMeta has CreationTimestamp set.",
			objectMeta: ObjectMeta{
				Name: "namespace",
				CreationTimestamp: NewTime(time.Date(2021, 9, 6, 20, 9, 23, 0, time.UTC)),
			},
			expected: []byte(`{"name":"namespace","creationTimestamp":"2021-09-06T20:09:23Z"}`),
		},{
			name: "ObjectMeta has CreationTimestamp with zero value.",
			objectMeta: ObjectMeta{
				Name: "namespace",
			},
			expected: []byte(`{"name":"namespace"}`),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := gojson.Marshal(tc.objectMeta)
			if err != nil {
				t.Fatalf(err.Error())
			}

			if !bytes.Equal(data, tc.expected) {
				t.Fatalf("Actual output does not match expected.\nActual:  %v\nExpected: %v\n", string(data), string(tc.expected))
			}
		})
	}
}