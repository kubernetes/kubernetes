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
	"reflect"
	"testing"
)

func TestCloneSelectorAndAddLabel(t *testing.T) {
	labels := map[string]string{
		"foo1": "bar1",
		"foo2": "bar2",
		"foo3": "bar3",
	}
	expressions := []LabelSelectorRequirement{
		{
			Key:      "key1",
			Operator: LabelSelectorOperator("in"),
			Values:   []string{"v1", "v2"},
		},
		{
			Key:      "key2",
			Operator: LabelSelectorOperator("in"),
			Values:   []string{"v3", "v4"},
		},
	}

	cases := []struct {
		selector   LabelSelector
		labelKey   string
		labelValue string
		want       LabelSelector
	}{
		{
			selector: LabelSelector{
				MatchLabels:      labels,
				MatchExpressions: expressions,
			},
			want: LabelSelector{
				MatchLabels:      labels,
				MatchExpressions: expressions,
			},
		},
		{
			selector: LabelSelector{
				MatchLabels:      labels,
				MatchExpressions: expressions,
			},
			labelKey:   "foo4",
			labelValue: "89",
			want: LabelSelector{
				MatchLabels: map[string]string{
					"foo1": "bar1",
					"foo2": "bar2",
					"foo3": "bar3",
					"foo4": "89",
				},
				MatchExpressions: expressions,
			},
		},
		{
			selector:   LabelSelector{},
			labelKey:   "foo4",
			labelValue: "12",
			want: LabelSelector{
				MatchLabels: map[string]string{
					"foo4": "12",
				},
			},
		},
	}

	for _, tc := range cases {
		got := CloneSelectorAndAddLabel(&tc.selector, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, &tc.want) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
	}
}

// Test if the CloneSelectorAndAddLabel really has cloned the input selector.
func TestIfCloneInCloneSelectorAndAddLabel(t *testing.T) {
	selector := &LabelSelector{
		MatchLabels: map[string]string{
			"foo1": "bar1",
			"foo2": "bar2",
			"foo3": "bar3",
		},
		MatchExpressions: []LabelSelectorRequirement{
			{
				Key:      "key1",
				Operator: LabelSelectorOperator("in"),
				Values:   []string{"v1", "v2"},
			},
			{
				Key:      "key2",
				Operator: LabelSelectorOperator("in"),
				Values:   []string{"v3", "v4"},
			},
		},
	}

	got := CloneSelectorAndAddLabel(selector, "new-key", "new-value")
	selector.MatchLabels["foo2"] = "bar2-new"
	selector.MatchExpressions[1].Values = []string{"v3-new", "v4-new"}
	if reflect.DeepEqual(got, selector) {
		t.Errorf("The CloneSelectorAndAddLabel don't clone selector, got %v", got)
	}
}

func TestAddLabelToSelector(t *testing.T) {
	labels := map[string]string{
		"foo1": "bar1",
		"foo2": "bar2",
		"foo3": "bar3",
	}

	cases := []struct {
		labels     map[string]string
		labelKey   string
		labelValue string
		want       map[string]string
	}{
		{
			labels: labels,
			want:   labels,
		},
		{
			labels:     labels,
			labelKey:   "foo4",
			labelValue: "89",
			want: map[string]string{
				"foo1": "bar1",
				"foo2": "bar2",
				"foo3": "bar3",
				"foo4": "89",
			},
		},
		{
			labels:     nil,
			labelKey:   "foo4",
			labelValue: "12",
			want: map[string]string{
				"foo4": "12",
			},
		},
	}

	for _, tc := range cases {
		ls_in := LabelSelector{MatchLabels: tc.labels}
		ls_out := LabelSelector{MatchLabels: tc.want}

		got := AddLabelToSelector(&ls_in, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, &ls_out) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
	}
}
