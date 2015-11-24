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
	"reflect"
	"testing"

	"github.com/ghodss/yaml"
	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

type FieldSelectorWrapper struct {
	Holder FieldSelector `json:"holder"`
}

func parseFieldSelector(t *testing.T, selector string) fields.Selector {
	result, err := fields.ParseSelector(selector)
	if err != nil {
		t.Fatalf("Unexpected error: %#v", err)
	}
	return result
}

func TestFieldSelectorMarshal(t *testing.T) {
	cases := []struct {
		selector fields.Selector
		result   string
	}{
		{parseFieldSelector(t, ""), "{\"holder\":\"\"}"},
		{parseFieldSelector(t, "foo=bar"), "{\"holder\":\"foo=bar\"}"},
		{parseFieldSelector(t, "foo=bar,bar=foo"), "{\"holder\":\"bar=foo,foo=bar\"}"},
		{parseFieldSelector(t, "foo=bar,bar!=foo"), "{\"holder\":\"bar!=foo,foo=bar\"}"},
	}

	for _, c := range cases {
		wrapper := FieldSelectorWrapper{FieldSelector{c.selector}}
		marshalled, err := json.Marshal(&wrapper)
		if err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if string(marshalled) != c.result {
			t.Errorf("Expected: %s, got: %s", c.result, string(marshalled))
		}

		var result FieldSelectorWrapper
		if err := json.Unmarshal(marshalled, &result); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(result, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, result)
		}

		var yamlResult FieldSelectorWrapper
		if err := yaml.Unmarshal(marshalled, &yamlResult); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(yamlResult, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, yamlResult)
		}

		var codecResult FieldSelectorWrapper
		if err := codec.NewDecoderBytes(marshalled, new(codec.JsonHandle)).Decode(&codecResult); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(codecResult, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, codecResult)
		}
	}
}

type LabelSelectorWrapper struct {
	Holder LabelSelector `json:"holder"`
}

func TestSelectorMarshal(t *testing.T) {
	cases := []struct {
		input  labels.Set
		result string
	}{
		{labels.Set(map[string]string{}), "{\"holder\":\"\"}"},
		{labels.Set(map[string]string{"foo": "bar"}), "{\"holder\":\"foo=bar\"}"},
		{labels.Set(map[string]string{"foo": "bar", "bar": "foo"}), "{\"holder\":\"bar=foo,foo=bar\"}"},
	}

	for _, c := range cases {
		selector := c.input.AsSelector()
		wrapper := LabelSelectorWrapper{LabelSelector{selector}}
		marshalled, err := json.Marshal(&wrapper)
		if err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if string(marshalled) != c.result {
			t.Errorf("Expected: %s, got: %s", c.result, string(marshalled))
		}

		var result LabelSelectorWrapper
		if err := json.Unmarshal(marshalled, &result); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(result, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, result)
		}

		var yamlResult LabelSelectorWrapper
		if err := yaml.Unmarshal(marshalled, &yamlResult); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(yamlResult, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, yamlResult)
		}

		var codecResult LabelSelectorWrapper
		if err := codec.NewDecoderBytes(marshalled, new(codec.JsonHandle)).Decode(&codecResult); err != nil {
			t.Errorf("Unexpected error: %#v", err)
		}
		if !reflect.DeepEqual(codecResult, wrapper) {
			t.Errorf("Expected: %#v, got: %#v", wrapper, codecResult)
		}
	}
}
