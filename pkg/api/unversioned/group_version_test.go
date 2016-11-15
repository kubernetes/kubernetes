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

package unversioned

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/ugorji/go/codec"
)

func TestGroupVersionParse(t *testing.T) {
	tests := []struct {
		input string
		out   GroupVersion
		err   func(error) bool
	}{
		{input: "v1", out: GroupVersion{Version: "v1"}},
		{input: "v2", out: GroupVersion{Version: "v2"}},
		{input: "/v1", out: GroupVersion{Version: "v1"}},
		{input: "v1/", out: GroupVersion{Group: "v1"}},
		{input: "/v1/", err: func(err error) bool { return err.Error() == "unexpected GroupVersion string: /v1/" }},
		{input: "v1/a", out: GroupVersion{Group: "v1", Version: "a"}},
	}
	for i, test := range tests {
		out, err := ParseGroupVersion(test.input)
		if test.err == nil && err != nil || err == nil && test.err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if test.err != nil && !test.err(err) {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		if out != test.out {
			t.Errorf("%d: unexpected output: %#v", i, out)
		}
	}
}

func TestGroupResourceParse(t *testing.T) {
	tests := []struct {
		input string
		out   GroupResource
	}{
		{input: "v1", out: GroupResource{Resource: "v1"}},
		{input: ".v1", out: GroupResource{Group: "v1"}},
		{input: "v1.", out: GroupResource{Resource: "v1"}},
		{input: "v1.a", out: GroupResource{Group: "a", Resource: "v1"}},
		{input: "b.v1.a", out: GroupResource{Group: "v1.a", Resource: "b"}},
	}
	for i, test := range tests {
		out := ParseGroupResource(test.input)
		if out != test.out {
			t.Errorf("%d: unexpected output: %#v", i, out)
		}
	}
}

func TestParseResourceArg(t *testing.T) {
	tests := []struct {
		input string
		gvr   *GroupVersionResource
		gr    GroupResource
	}{
		{input: "v1", gr: GroupResource{Resource: "v1"}},
		{input: ".v1", gr: GroupResource{Group: "v1"}},
		{input: "v1.", gr: GroupResource{Resource: "v1"}},
		{input: "v1.a", gr: GroupResource{Group: "a", Resource: "v1"}},
		{input: "b.v1.a", gvr: &GroupVersionResource{Group: "a", Version: "v1", Resource: "b"}, gr: GroupResource{Group: "v1.a", Resource: "b"}},
	}
	for i, test := range tests {
		gvr, gr := ParseResourceArg(test.input)
		if (gvr != nil && test.gvr == nil) || (gvr == nil && test.gvr != nil) || (test.gvr != nil && *gvr != *test.gvr) {
			t.Errorf("%d: unexpected output: %#v", i, gvr)
		}
		if gr != test.gr {
			t.Errorf("%d: unexpected output: %#v", i, gr)
		}
	}
}

type GroupVersionHolder struct {
	GV GroupVersion `json:"val"`
}

func TestGroupVersionUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  []byte
		expect GroupVersion
	}{
		{[]byte(`{"val": "v1"}`), GroupVersion{"", "v1"}},
		{[]byte(`{"val": "extensions/v1beta1"}`), GroupVersion{"extensions", "v1beta1"}},
	}

	for _, c := range cases {
		var result GroupVersionHolder
		// test golang lib's JSON codec
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("JSON codec failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.GV, c.expect) {
			t.Errorf("JSON codec failed to unmarshal input '%s': expected %+v, got %+v", c.input, c.expect, result.GV)
		}
		// test the Ugorji codec
		if err := codec.NewDecoderBytes(c.input, new(codec.JsonHandle)).Decode(&result); err != nil {
			t.Errorf("Ugorji codec failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.GV, c.expect) {
			t.Errorf("Ugorji codec failed to unmarshal input '%s': expected %+v, got %+v", c.input, c.expect, result.GV)
		}
	}
}

func TestGroupVersionMarshalJSON(t *testing.T) {
	cases := []struct {
		input  GroupVersion
		expect []byte
	}{
		{GroupVersion{"", "v1"}, []byte(`{"val":"v1"}`)},
		{GroupVersion{"extensions", "v1beta1"}, []byte(`{"val":"extensions/v1beta1"}`)},
	}

	for _, c := range cases {
		input := GroupVersionHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if !reflect.DeepEqual(result, c.expect) {
			t.Errorf("Failed to marshal input '%+v': expected: %s, got: %s", input, c.expect, result)
		}
	}
}

func TestKindForGroupVersionKinds(t *testing.T) {
	gvks := GroupVersions{
		GroupVersion{Group: "batch", Version: "v1"},
		GroupVersion{Group: "batch", Version: "v2alpha1"},
		GroupVersion{Group: "policy", Version: "v1beta1"},
	}
	cases := []struct {
		input  []GroupVersionKind
		target GroupVersionKind
		ok     bool
	}{
		{
			input:  []GroupVersionKind{{Group: "batch", Version: "v2alpha1", Kind: "ScheduledJob"}},
			target: GroupVersionKind{Group: "batch", Version: "v2alpha1", Kind: "ScheduledJob"},
			ok:     true,
		},
		{
			input:  []GroupVersionKind{{Group: "batch", Version: "v3alpha1", Kind: "CronJob"}},
			target: GroupVersionKind{Group: "batch", Version: "v1", Kind: "CronJob"},
			ok:     true,
		},
		{
			input:  []GroupVersionKind{{Group: "policy", Version: "v1beta1", Kind: "PodDisruptionBudget"}},
			target: GroupVersionKind{Group: "policy", Version: "v1beta1", Kind: "PodDisruptionBudget"},
			ok:     true,
		},
		{
			input:  []GroupVersionKind{{Group: "apps", Version: "v1alpha1", Kind: "StatefulSet"}},
			target: GroupVersionKind{},
			ok:     false,
		},
	}

	for i, c := range cases {
		target, ok := gvks.KindForGroupVersionKinds(c.input)
		if c.target != target {
			t.Errorf("%d: unexpected target: %v, expected %v", i, target, c.target)
		}
		if c.ok != ok {
			t.Errorf("%d: unexpected ok: %v, expected %v", i, ok, c.ok)
		}
	}
}
