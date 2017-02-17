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

package selectors

import (
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

func TestLabelSelectorAsSelector(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	matchExpressions := []metav1.LabelSelectorRequirement{{
		Key:      "baz",
		Operator: metav1.LabelSelectorOpIn,
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
		in        *metav1.LabelSelector
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: &metav1.LabelSelector{}, out: labels.Everything()},
		{
			in:  &metav1.LabelSelector{MatchLabels: matchLabels},
			out: mustParse("foo=bar"),
		},
		{
			in:  &metav1.LabelSelector{MatchExpressions: matchExpressions},
			out: mustParse("baz in (norf,qux)"),
		},
		{
			in:  &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions},
			out: mustParse("baz in (norf,qux),foo=bar"),
		},
		{
			in: &metav1.LabelSelector{
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "baz",
					Operator: metav1.LabelSelectorOpExists,
					Values:   []string{"qux", "norf"},
				}},
			},
			expectErr: true,
		},
	}

	for i, tc := range tc {
		out, err := LabelSelectorAsSelector(tc.in)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		if !reflect.DeepEqual(out, tc.out) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, tc.out, out)
		}
	}
}

func TestLabelSelectorAsMap(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	matchExpressions := func(operator metav1.LabelSelectorOperator, values []string) []metav1.LabelSelectorRequirement {
		return []metav1.LabelSelectorRequirement{{
			Key:      "baz",
			Operator: operator,
			Values:   values,
		}}
	}

	tests := []struct {
		in        *metav1.LabelSelector
		out       map[string]string
		errString string
	}{
		{in: nil, out: nil},
		{
			in:  &metav1.LabelSelector{MatchLabels: matchLabels},
			out: map[string]string{"foo": "bar"},
		},
		{
			in:  &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(metav1.LabelSelectorOpIn, []string{"norf"})},
			out: map[string]string{"foo": "bar", "baz": "norf"},
		},
		{
			in:  &metav1.LabelSelector{MatchExpressions: matchExpressions(metav1.LabelSelectorOpIn, []string{"norf"})},
			out: map[string]string{"baz": "norf"},
		},
		{
			in:        &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(metav1.LabelSelectorOpIn, []string{"norf", "qux"})},
			out:       map[string]string{"foo": "bar"},
			errString: "without a single value cannot be converted",
		},
		{
			in:        &metav1.LabelSelector{MatchExpressions: matchExpressions(metav1.LabelSelectorOpNotIn, []string{"norf", "qux"})},
			out:       map[string]string{},
			errString: "cannot be converted",
		},
		{
			in:        &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions(metav1.LabelSelectorOpExists, []string{})},
			out:       map[string]string{"foo": "bar"},
			errString: "cannot be converted",
		},
		{
			in:        &metav1.LabelSelector{MatchExpressions: matchExpressions(metav1.LabelSelectorOpDoesNotExist, []string{})},
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
