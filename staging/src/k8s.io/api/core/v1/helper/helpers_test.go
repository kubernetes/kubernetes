/*
Copyright The Kubernetes Authors.

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

package helper

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
)

func TestTopologySelectorRequirementsAsSelector(t *testing.T) {
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        []v1.TopologySelectorLabelRequirement
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: []v1.TopologySelectorLabelRequirement{}, out: labels.Nothing()},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "foo",
				Values: []string{"bar", "baz"},
			}},
			out: mustParse("foo in (baz,bar)"),
		},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "foo",
				Values: []string{},
			}},
			expectErr: true,
		},
		{
			in: []v1.TopologySelectorLabelRequirement{
				{
					Key:    "foo",
					Values: []string{"bar", "baz"},
				},
				{
					Key:    "invalid",
					Values: []string{},
				},
			},
			expectErr: true,
		},
		{
			in: []v1.TopologySelectorLabelRequirement{{
				Key:    "/invalidkey",
				Values: []string{"bar", "baz"},
			}},
			expectErr: true,
		},
	}

	for i, tc := range tc {
		out, err := TopologySelectorRequirementsAsSelector(tc.in)
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

func TestMatchTopologySelectorTerms(t *testing.T) {
	type args struct {
		topologySelectorTerms []v1.TopologySelectorTerm
		labels                labels.Set
	}

	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "nil term list",
			args: args{
				topologySelectorTerms: nil,
				labels:                nil,
			},
			want: true,
		},
		{
			name: "nil term",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{},
					},
				},
				labels: nil,
			},
			want: false,
		},
		{
			name: "label matches MatchLabelExpressions terms",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val"},
			},
			want: true,
		},
		{
			name: "label does not match MatchLabelExpressions terms",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val-failed"},
			},
			want: false,
		},
		{
			name: "multi-values in one requirement, one matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val1", "label_1_val2"},
						}},
					},
				},
				labels: map[string]string{"label_1": "label_1_val2"},
			},
			want: true,
		},
		{
			name: "multi-terms was set, one matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_1",
							Values: []string{"label_1_val"},
						}},
					},
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{{
							Key:    "label_2",
							Values: []string{"label_2_val"},
						}},
					},
				},
				labels: map[string]string{
					"label_2": "label_2_val",
				},
			},
			want: true,
		},
		{
			name: "multi-requirement in one term, fully matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
							{
								Key:    "label_1",
								Values: []string{"label_1_val"},
							},
							{
								Key:    "label_2",
								Values: []string{"label_2_val"},
							},
						},
					},
				},
				labels: map[string]string{
					"label_1": "label_1_val",
					"label_2": "label_2_val",
				},
			},
			want: true,
		},
		{
			name: "multi-requirement in one term, partial matched",
			args: args{
				topologySelectorTerms: []v1.TopologySelectorTerm{
					{
						MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
							{
								Key:    "label_1",
								Values: []string{"label_1_val"},
							},
							{
								Key:    "label_2",
								Values: []string{"label_2_val"},
							},
						},
					},
				},
				labels: map[string]string{
					"label_1": "label_1_val-failed",
					"label_2": "label_2_val",
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchTopologySelectorTerms(tt.args.topologySelectorTerms, tt.args.labels); got != tt.want {
				t.Errorf("MatchTopologySelectorTermsORed() = %v, want %v", got, tt.want)
			}
		})
	}
}
