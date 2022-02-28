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

package labels

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCloneAndAddLabel(t *testing.T) {
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
			labelValue: "42",
			want: map[string]string{
				"foo1": "bar1",
				"foo2": "bar2",
				"foo3": "bar3",
				"foo4": "42",
			},
		},
	}

	for _, tc := range cases {
		got := CloneAndAddLabel(tc.labels, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("[Add] got %v, want %v", got, tc.want)
		}
		// now test the inverse.
		got_rm := CloneAndRemoveLabel(got, tc.labelKey)
		if !reflect.DeepEqual(got_rm, tc.labels) {
			t.Errorf("[RM] got %v, want %v", got_rm, tc.labels)
		}
	}
}

func TestAddLabel(t *testing.T) {
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
			labelValue: "food",
			want: map[string]string{
				"foo1": "bar1",
				"foo2": "bar2",
				"foo3": "bar3",
				"foo4": "food",
			},
		},
		{
			labels:     nil,
			labelKey:   "foo4",
			labelValue: "food",
			want: map[string]string{
				"foo4": "food",
			},
		},
	}

	for _, tc := range cases {
		got := AddLabel(tc.labels, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
	}
}

func TestCloneSelectorAndAddLabel(t *testing.T) {
	labels := map[string]string{
		"foo1": "bar1",
		"foo2": "bar2",
		"foo3": "bar3",
	}

	cases := []struct {
		name             string
		labels           map[string]string
		matchExpressions []metav1.LabelSelectorRequirement
		labelKey         string
		labelValue       string
		want             metav1.LabelSelector
	}{
		{
			name:   "don't add a label when labelKey is empty",
			labels: labels,
			want: metav1.LabelSelector{
				MatchLabels: labels,
			},
		},
		{
			name:       "add a label",
			labels:     labels,
			labelKey:   "foo4",
			labelValue: "89",
			want: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo1": "bar1",
					"foo2": "bar2",
					"foo3": "bar3",
					"foo4": "89",
				},
			},
		},
		{
			name:       "add a label when selector MatchLabels is nil",
			labels:     nil,
			labelKey:   "foo4",
			labelValue: "12",
			want: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo4": "12",
				},
				MatchExpressions: nil,
			},
		},
		{
			name:             "add a label when selector MatchLabels is not nil",
			labels:           labels,
			matchExpressions: nil,
			labelKey:         "foo4",
			labelValue:       "89",
			want: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo1": "bar1",
					"foo2": "bar2",
					"foo3": "bar3",
					"foo4": "89",
				},
				MatchExpressions: nil,
			},
		},
		{
			name:   "selector matchExpressions is not nil and matchExpressions.Values is nil",
			labels: labels,
			matchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "foo1",
				Operator: metav1.LabelSelectorOpExists,
			}},
			labelKey:   "foo4",
			labelValue: "89",
			want: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo1": "bar1",
					"foo2": "bar2",
					"foo3": "bar3",
					"foo4": "89",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "foo1",
					Operator: metav1.LabelSelectorOpExists,
					Values:   nil,
				}},
			},
		},
		{
			name:   "selector matchExpressions.Values is not nil",
			labels: labels,
			matchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "foo1",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"bar", "bar1"},
			}},
			labelKey:   "foo4",
			labelValue: "89",
			want: metav1.LabelSelector{
				MatchLabels: map[string]string{
					"foo1": "bar1",
					"foo2": "bar2",
					"foo3": "bar3",
					"foo4": "89",
				},
				MatchExpressions: []metav1.LabelSelectorRequirement{{
					Key:      "foo1",
					Operator: metav1.LabelSelectorOpIn,
					Values:   []string{"bar", "bar1"},
				}},
			},
		},
	}

	for _, tc := range cases {
		ls_in := metav1.LabelSelector{MatchLabels: tc.labels, MatchExpressions: tc.matchExpressions}
		ls_out := metav1.LabelSelector{MatchLabels: tc.want.MatchLabels, MatchExpressions: tc.matchExpressions}

		got := CloneSelectorAndAddLabel(&ls_in, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, &ls_out) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
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
		ls_in := metav1.LabelSelector{MatchLabels: tc.labels}
		ls_out := metav1.LabelSelector{MatchLabels: tc.want}

		got := AddLabelToSelector(&ls_in, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, &ls_out) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
	}
}

func TestSelectorHasLabel(t *testing.T) {
	labels := map[string]string{
		"foo1": "bar1",
		"foo2": "bar2",
	}

	cases := []struct {
		name         string
		labels       map[string]string
		labelKey     string
		wantHasLabel bool
	}{
		{
			name:         "the given label key is empty",
			labels:       labels,
			labelKey:     "",
			wantHasLabel: false,
		},
		{
			name:         "the given selector.MatchLabels contains the given label key",
			labels:       labels,
			labelKey:     "foo1",
			wantHasLabel: true,
		},
		{
			name:         "the given selector.MatchLabels is nil",
			labels:       nil,
			labelKey:     "foo2",
			wantHasLabel: false,
		},
		{
			name:         "the given selector.MatchLabels does not contains the given label key",
			labels:       labels,
			labelKey:     "foo3",
			wantHasLabel: false,
		},
	}

	for _, tc := range cases {
		lsInput := metav1.LabelSelector{MatchLabels: tc.labels}

		got := SelectorHasLabel(&lsInput, tc.labelKey)
		if !reflect.DeepEqual(got, tc.wantHasLabel) {
			t.Errorf("got %v, want %v", got, tc.wantHasLabel)
		}
	}
}
