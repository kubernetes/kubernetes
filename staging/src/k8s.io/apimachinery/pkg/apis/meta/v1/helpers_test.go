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

	"k8s.io/apimachinery/pkg/labels"
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
