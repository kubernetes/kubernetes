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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
)

func TestLabelSelectorAsSelector(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	matchExpressions := []LabelSelectorRequirement{{
		Key:      "baz",
		Operator: LabelSelectorOpIn,
		Values:   []string{"qux", "norf"},
	}}
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        *LabelSelector
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: &LabelSelector{}, out: labels.Everything()},
		{
			in:  &LabelSelector{MatchLabels: matchLabels},
			out: mustParse("foo=bar"),
		},
		{
			in:  &LabelSelector{MatchExpressions: matchExpressions},
			out: mustParse("baz in (norf,qux)"),
		},
		{
			in:  &LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions},
			out: mustParse("baz in (norf,qux),foo=bar"),
		},
		{
			in: &LabelSelector{
				MatchExpressions: []LabelSelectorRequirement{{
					Key:      "baz",
					Operator: LabelSelectorOpExists,
					Values:   []string{"qux", "norf"},
				}},
			},
			expectErr: true,
		},
	}

	for i, tc := range tc {
		inCopy := tc.in.DeepCopy()
		out, err := LabelSelectorAsSelector(tc.in)
		// after calling LabelSelectorAsSelector, tc.in shouldn't be modified
		if !reflect.DeepEqual(inCopy, tc.in) {
			t.Errorf("[%v]expected:\n\t%#v\nbut got:\n\t%#v", i, inCopy, tc.in)
		}
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		// fmt.Sprint() over String() as nil.String() will panic
		if fmt.Sprint(out) != fmt.Sprint(tc.out) {
			t.Errorf("[%v]expected:\n\t%s\nbut got:\n\t%s", i, fmt.Sprint(tc.out), fmt.Sprint(out))
		}
	}
}

func BenchmarkLabelSelectorAsSelector(b *testing.B) {
	selector := &LabelSelector{
		MatchLabels: map[string]string{
			"foo": "foo",
			"bar": "bar",
		},
		MatchExpressions: []LabelSelectorRequirement{{
			Key:      "baz",
			Operator: LabelSelectorOpExists,
		}},
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := LabelSelectorAsSelector(selector)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestLabelSelectorAsMap(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	matchExpressions := func(operator LabelSelectorOperator, values []string) []LabelSelectorRequirement {
		return []LabelSelectorRequirement{{
			Key:      "baz",
			Operator: operator,
			Values:   values,
		}}
	}

	tests := []struct {
		in        *LabelSelector
		out       map[string]string
		errString string
	}{
		{in: nil, out: nil},
		{
			in:  &LabelSelector{MatchLabels: matchLabels},
			out: map[string]string{"foo": "bar"},
		},
		{
			in:  &LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(LabelSelectorOpIn, []string{"norf"})},
			out: map[string]string{"foo": "bar", "baz": "norf"},
		},
		{
			in:  &LabelSelector{MatchExpressions: matchExpressions(LabelSelectorOpIn, []string{"norf"})},
			out: map[string]string{"baz": "norf"},
		},
		{
			in:        &LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(LabelSelectorOpIn, []string{"norf", "qux"})},
			out:       map[string]string{"foo": "bar"},
			errString: "without a single value cannot be converted",
		},
		{
			in:        &LabelSelector{MatchExpressions: matchExpressions(LabelSelectorOpNotIn, []string{"norf", "qux"})},
			out:       map[string]string{},
			errString: "cannot be converted",
		},
		{
			in:        &LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(LabelSelectorOpExists, []string{})},
			out:       map[string]string{"foo": "bar"},
			errString: "cannot be converted",
		},
		{
			in:        &LabelSelector{MatchExpressions: matchExpressions(LabelSelectorOpDoesNotExist, []string{})},
			out:       map[string]string{},
			errString: "cannot be converted",
		},
	}

	for i, tc := range tests {
		out, err := LabelSelectorAsMap(tc.in)
		if err == nil && len(tc.errString) > 0 {
			t.Errorf("[%v]expected error but got none.", i)
			continue
		}
		if err != nil && len(tc.errString) == 0 {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
			continue
		}
		if err != nil && len(tc.errString) > 0 && !strings.Contains(err.Error(), tc.errString) {
			t.Errorf("[%v]expected error with %q but got: %v", i, tc.errString, err)
			continue
		}
		if !reflect.DeepEqual(out, tc.out) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, tc.out, out)
		}
	}
}

func TestResetObjectMetaForStatus(t *testing.T) {
	meta := &ObjectMeta{}
	existingMeta := &ObjectMeta{}

	// fuzz the existingMeta to set every field, no nils
	f := fuzz.New().NilChance(0).NumElements(1, 1).MaxDepth(10)
	f.Fuzz(existingMeta)
	ResetObjectMetaForStatus(meta, existingMeta)

	// not all fields are stomped during the reset.  These fields should not have been set. False
	// set them all to their zero values.  Before you add anything to this list, consider whether or not
	// you're enforcing immutability (those are fine) and whether /status should be able to update
	// these values (these are usually not fine).

	// generateName doesn't do anything after create
	existingMeta.SetGenerateName("")
	// resourceVersion is enforced in validation and used during the storage update
	existingMeta.SetResourceVersion("")
	// fields made immutable in validation
	existingMeta.SetUID(types.UID(""))
	existingMeta.SetName("")
	existingMeta.SetNamespace("")
	existingMeta.SetCreationTimestamp(Time{})
	existingMeta.SetDeletionTimestamp(nil)
	existingMeta.SetDeletionGracePeriodSeconds(nil)
	existingMeta.SetManagedFields(nil)

	if !reflect.DeepEqual(meta, existingMeta) {
		t.Error(cmp.Diff(meta, existingMeta))
	}
}

func TestSetMetaDataLabel(t *testing.T) {
	tests := []struct {
		obj   *ObjectMeta
		label string
		value string
		want  map[string]string
	}{
		{
			obj:   &ObjectMeta{},
			label: "foo",
			value: "bar",
			want:  map[string]string{"foo": "bar"},
		},
		{
			obj:   &ObjectMeta{Labels: map[string]string{"foo": "bar"}},
			label: "foo",
			value: "baz",
			want:  map[string]string{"foo": "baz"},
		},
		{
			obj:   &ObjectMeta{Labels: map[string]string{"foo": "bar"}},
			label: "version",
			value: "1.0.0",
			want:  map[string]string{"foo": "bar", "version": "1.0.0"},
		},
	}

	for _, tc := range tests {
		SetMetaDataLabel(tc.obj, tc.label, tc.value)
		if !reflect.DeepEqual(tc.obj.Labels, tc.want) {
			t.Errorf("got %v, want %v", tc.obj.Labels, tc.want)
		}
	}
}

func TestFieldsV1MarshalJSON(t *testing.T) {
	for _, tc := range []struct {
		Name     string
		FieldsV1 FieldsV1
		Want     []byte
		Error    string
	}{
		{
			Name:     "nil encodes as json null",
			FieldsV1: FieldsV1{},
			Want:     []byte(`null`),
		},
		{
			Name:     "empty invalid json is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{}},
			Want:     []byte{},
		},
		{
			Name:     "cbor null is transcoded to json null",
			FieldsV1: FieldsV1{Raw: []byte{0xf6}}, // null
			Want:     []byte(`null`),
		},
		{
			Name:     "valid non-map cbor and valid non-object json is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{0x30}},
			Want:     []byte{0x30}, // Valid CBOR encoding of -17 and JSON encoding of 0!
		},
		{
			Name:     "self-described cbor map is transcoded to json map",
			FieldsV1: FieldsV1{Raw: []byte{0xd9, 0xd9, 0xf7, 0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'}}, // 55799({"foo":"bar"})
			Want:     []byte(`{"foo":"bar"}`),
		},
		{
			Name:     "json object is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte(" \t\r\n{\"foo\":\"bar\"}")},
			Want:     []byte(" \t\r\n{\"foo\":\"bar\"}"),
		},
		{
			Name:     "invalid json is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte(`{{`)},
			Want:     []byte(`{{`),
		},
		{
			Name:     "invalid cbor fails to transcode to json",
			FieldsV1: FieldsV1{Raw: []byte{0xa1}},
			Error:    "metav1.FieldsV1 cbor invalid: unexpected EOF",
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := tc.FieldsV1.MarshalJSON()
			if err != nil {
				if tc.Error == "" {
					t.Fatalf("unexpected error: %v", err)
				}
				if msg := err.Error(); msg != tc.Error {
					t.Fatalf("expected error %q, got %q", tc.Error, msg)
				}
			} else if tc.Error != "" {
				t.Fatalf("expected error %q, got nil", tc.Error)
			}
			if diff := cmp.Diff(tc.Want, got); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}

func TestFieldsV1MarshalCBOR(t *testing.T) {
	for _, tc := range []struct {
		Name     string
		FieldsV1 FieldsV1
		Want     []byte
		Error    string
	}{
		{
			Name:     "nil encodes as cbor null",
			FieldsV1: FieldsV1{},
			Want:     []byte{0xf6}, // null
		},
		{
			Name:     "empty invalid cbor is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{}},
			Want:     []byte{},
		},
		{
			Name:     "json null is transcoded to cbor null",
			FieldsV1: FieldsV1{Raw: []byte(`null`)},
			Want:     []byte{0xf6}, // null
		},
		{
			Name:     "valid non-map cbor and valid non-object json is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{0x30}},
			Want:     []byte{0x30}, // Valid CBOR encoding of -17 and JSON encoding of 0!
		},
		{
			Name:     "json object is transcoded to cbor map",
			FieldsV1: FieldsV1{Raw: []byte(" \t\r\n{\"foo\":\"bar\"}")},
			Want:     []byte{0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'},
		},
		{
			Name:     "self-described cbor map is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{0xd9, 0xd9, 0xf7, 0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'}}, // 55799({"foo":"bar"})
			Want:     []byte{0xd9, 0xd9, 0xf7, 0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'},                // 55799({"foo":"bar"})
		},
		{
			Name:     "invalid json fails to transcode to cbor",
			FieldsV1: FieldsV1{Raw: []byte(`{{`)},
			Error:    "metav1.FieldsV1 json invalid: invalid character '{' looking for beginning of object key string",
		},
		{
			Name:     "invalid cbor is returned as-is",
			FieldsV1: FieldsV1{Raw: []byte{0xa1}},
			Want:     []byte{0xa1},
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := tc.FieldsV1.MarshalCBOR()
			if err != nil {
				if tc.Error == "" {
					t.Fatalf("unexpected error: %v", err)
				}
				if msg := err.Error(); msg != tc.Error {
					t.Fatalf("expected error %q, got %q", tc.Error, msg)
				}
			} else if tc.Error != "" {
				t.Fatalf("expected error %q, got nil", tc.Error)
			}

			if diff := cmp.Diff(tc.Want, got); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}

func TestFieldsV1UnmarshalJSON(t *testing.T) {
	for _, tc := range []struct {
		Name  string
		JSON  []byte
		Into  *FieldsV1
		Want  *FieldsV1
		Error string
	}{
		{
			Name:  "nil receiver returns error",
			Into:  nil,
			Error: "metav1.FieldsV1: UnmarshalJSON on nil pointer",
		},
		{
			Name: "json null does not modify receiver", // conventional for json.Unmarshaler
			JSON: []byte(`null`),
			Into: &FieldsV1{Raw: []byte(`unmodified`)},
			Want: &FieldsV1{Raw: []byte(`unmodified`)},
		},
		{
			Name: "valid input is copied verbatim",
			JSON: []byte("{\"foo\":\"bar\"} \t\r\n"),
			Into: &FieldsV1{},
			Want: &FieldsV1{Raw: []byte("{\"foo\":\"bar\"} \t\r\n")},
		},
		{
			Name: "invalid input is copied verbatim",
			JSON: []byte("{{"),
			Into: &FieldsV1{},
			Want: &FieldsV1{Raw: []byte("{{")},
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			got := tc.Into.DeepCopy()
			err := got.UnmarshalJSON(tc.JSON)
			if err != nil {
				if tc.Error == "" {
					t.Fatalf("unexpected error: %v", err)
				}
				if msg := err.Error(); msg != tc.Error {
					t.Fatalf("expected error %q, got %q", tc.Error, msg)
				}
			} else if tc.Error != "" {
				t.Fatalf("expected error %q, got nil", tc.Error)
			}

			if diff := cmp.Diff(tc.Want, got); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}

func TestFieldsV1UnmarshalCBOR(t *testing.T) {
	for _, tc := range []struct {
		Name  string
		CBOR  []byte
		Into  *FieldsV1
		Want  *FieldsV1
		Error string
	}{
		{
			Name:  "nil receiver returns error",
			Into:  nil,
			Want:  nil,
			Error: "metav1.FieldsV1: UnmarshalCBOR on nil pointer",
		},
		{
			Name: "cbor null does not modify receiver",
			CBOR: []byte{0xf6},
			Into: &FieldsV1{Raw: []byte(`unmodified`)},
			Want: &FieldsV1{Raw: []byte(`unmodified`)},
		},
		{
			Name: "valid input is copied verbatim",
			CBOR: []byte{0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'},
			Into: &FieldsV1{},
			Want: &FieldsV1{Raw: []byte{0xa1, 0x43, 'f', 'o', 'o', 0x43, 'b', 'a', 'r'}},
		},
		{
			Name: "invalid input is copied verbatim",
			CBOR: []byte{0xff}, // UnmarshalCBOR should never be called with malformed input, testing anyway.
			Into: &FieldsV1{},
			Want: &FieldsV1{Raw: []byte{0xff}},
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			got := tc.Into.DeepCopy()
			err := got.UnmarshalCBOR(tc.CBOR)
			if err != nil {
				if tc.Error == "" {
					t.Fatalf("unexpected error: %v", err)
				}
				if msg := err.Error(); msg != tc.Error {
					t.Fatalf("expected error %q, got %q", tc.Error, msg)
				}
			} else if tc.Error != "" {
				t.Fatalf("expected error %q, got nil", tc.Error)
			}

			if diff := cmp.Diff(tc.Want, got); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}
