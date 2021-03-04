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

package schema

import (
	"testing"
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

func TestParseKindArg(t *testing.T) {
	tests := []struct {
		input string
		gvk   *GroupVersionKind
		gk    GroupKind
	}{
		{input: "Pod", gk: GroupKind{Kind: "Pod"}},
		{input: ".apps", gk: GroupKind{Group: "apps"}},
		{input: "Pod.", gk: GroupKind{Kind: "Pod"}},
		{input: "StatefulSet.apps", gk: GroupKind{Group: "apps", Kind: "StatefulSet"}},
		{input: "StatefulSet.v1.apps", gvk: &GroupVersionKind{Group: "apps", Version: "v1", Kind: "StatefulSet"}, gk: GroupKind{Group: "v1.apps", Kind: "StatefulSet"}},
	}
	for i, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			gvk, gk := ParseKindArg(test.input)
			if (gvk != nil && test.gvk == nil) || (gvk == nil && test.gvk != nil) || (test.gvk != nil && *gvk != *test.gvk) {
				t.Errorf("%d: expected output: %#v, got: %#v", i, test.gvk, gvk)
			}
			if gk != test.gk {
				t.Errorf("%d: expected output: %#v, got: %#v", i, test.gk, gk)
			}
		})
	}
}

func TestParseGroupKind(t *testing.T) {
	tests := []struct {
		input string
		out   GroupKind
	}{
		{input: "Pod", out: GroupKind{Kind: "Pod"}},
		{input: ".StatefulSet", out: GroupKind{Group: "StatefulSet"}},
		{input: "StatefulSet.apps", out: GroupKind{Group: "apps", Kind: "StatefulSet"}},
	}
	for i, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			out := ParseGroupKind(test.input)
			if out != test.out {
				t.Errorf("%d: expected output: %#v, got: %#v", i, test.out, out)
			}
		})
	}
}
