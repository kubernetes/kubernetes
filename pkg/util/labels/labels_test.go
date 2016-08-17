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

	"k8s.io/kubernetes/pkg/api/unversioned"
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
		labelValue uint32
		want       map[string]string
	}{
		{
			labels: labels,
			want:   labels,
		},
		{
			labels:     labels,
			labelKey:   "foo4",
			labelValue: uint32(42),
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
		labels     map[string]string
		labelKey   string
		labelValue uint32
		want       map[string]string
	}{
		{
			labels: labels,
			want:   labels,
		},
		{
			labels:     labels,
			labelKey:   "foo4",
			labelValue: 89,
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
			labelValue: 12,
			want: map[string]string{
				"foo4": "12",
			},
		},
	}

	for _, tc := range cases {
		ls_in := unversioned.LabelSelector{MatchLabels: tc.labels}
		ls_out := unversioned.LabelSelector{MatchLabels: tc.want}

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
		ls_in := unversioned.LabelSelector{MatchLabels: tc.labels}
		ls_out := unversioned.LabelSelector{MatchLabels: tc.want}

		got := AddLabelToSelector(&ls_in, tc.labelKey, tc.labelValue)
		if !reflect.DeepEqual(got, &ls_out) {
			t.Errorf("got %v, want %v", got, tc.want)
		}
	}
}
