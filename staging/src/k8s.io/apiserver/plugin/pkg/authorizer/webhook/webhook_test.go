/*
Copyright 2024 The Kubernetes Authors.

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

package webhook

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func mustLabelRequirement(selector string) labels.Requirements {
	ret, err := labels.Parse(selector)
	if err != nil {
		panic(err)
	}
	requirements, _ := ret.Requirements()
	return requirements
}

func Test_resourceAttributesFrom(t *testing.T) {
	type args struct {
		attr authorizer.Attributes
	}
	tests := []struct {
		name string
		args args
		want *authorizationv1.ResourceAttributes
	}{
		{
			name: "field selector: ignore error",
			args: args{
				attr: authorizer.AttributesRecord{
					FieldSelectorRequirements: fields.Requirements{
						fields.OneTermEqualSelector("foo", "bar").Requirements()[0],
					},
					FieldSelectorParsingErr: errors.New("failed"),
				},
			},
			want: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{{Key: "foo", Operator: "In", Values: []string{"bar"}}},
				},
			},
		},
		{
			name: "label selector: ignore error",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("foo in (bar,baz)"),
					LabelSelectorParsingErr:   errors.New("failed"),
				},
			},
			want: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{{Key: "foo", Operator: "In", Values: []string{"bar", "baz"}}},
				},
			},
		},
		{
			name: "field selector: equals, double equals, in",
			args: args{
				attr: authorizer.AttributesRecord{
					FieldSelectorRequirements: fields.Requirements{
						{Operator: selection.Equals, Field: "foo", Value: "bar"},
						{Operator: selection.DoubleEquals, Field: "one", Value: "two"},
						{Operator: selection.In, Field: "apple", Value: "banana"},
					},
					FieldSelectorParsingErr: nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "foo",
							Operator: "In",
							Values:   []string{"bar"},
						},
						{
							Key:      "one",
							Operator: "In",
							Values:   []string{"two"},
						},
						{
							Key:      "apple",
							Operator: "In",
							Values:   []string{"banana"},
						},
					},
				},
			},
		},
		{
			name: "field selector: not equals, not in",
			args: args{
				attr: authorizer.AttributesRecord{
					FieldSelectorRequirements: fields.Requirements{
						{Operator: selection.NotEquals, Field: "foo", Value: "bar"},
						{Operator: selection.NotIn, Field: "apple", Value: "banana"},
					},
					FieldSelectorParsingErr: nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "foo",
							Operator: "NotIn",
							Values:   []string{"bar"},
						},
						{
							Key:      "apple",
							Operator: "NotIn",
							Values:   []string{"banana"},
						},
					},
				},
			},
		},
		{
			name: "field selector: unknown operator skipped",
			args: args{
				attr: authorizer.AttributesRecord{
					FieldSelectorRequirements: fields.Requirements{
						{Operator: selection.NotEquals, Field: "foo", Value: "bar"},
						{Operator: selection.Operator("bad"), Field: "apple", Value: "banana"},
					},
					FieldSelectorParsingErr: nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "foo",
							Operator: "NotIn",
							Values:   []string{"bar"},
						},
					},
				},
			},
		},
		{
			name: "field selector: no requirements has no fieldselector",
			args: args{
				attr: authorizer.AttributesRecord{
					FieldSelectorRequirements: fields.Requirements{
						{Operator: selection.Operator("bad"), Field: "apple", Value: "banana"},
					},
					FieldSelectorParsingErr: nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{},
		},
		{
			name: "label selector: in, equals, double equals",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("foo in (bar,baz), one=two, apple==banana"),
					LabelSelectorParsingErr:   nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "apple",
							Operator: "In",
							Values:   []string{"banana"},
						},
						{
							Key:      "foo",
							Operator: "In",
							Values:   []string{"bar", "baz"},
						},
						{
							Key:      "one",
							Operator: "In",
							Values:   []string{"two"},
						},
					},
				},
			},
		},
		{
			name: "label selector: not in, not equals",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("foo notin (bar,baz), one!=two"),
					LabelSelectorParsingErr:   nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: "NotIn",
							Values:   []string{"bar", "baz"},
						},
						{
							Key:      "one",
							Operator: "NotIn",
							Values:   []string{"two"},
						},
					},
				},
			},
		},
		{
			name: "label selector: exists, not exists",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("foo, !one"),
					LabelSelectorParsingErr:   nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: "Exists",
						},
						{
							Key:      "one",
							Operator: "DoesNotExist",
						},
					},
				},
			},
		},
		{
			name: "label selector: unknown operator skipped",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("foo != bar, apple > 1"),
					LabelSelectorParsingErr:   nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: "NotIn",
							Values:   []string{"bar"},
						},
					},
				},
			},
		},
		{
			name: "label selector: no requirements has no labelselector",
			args: args{
				attr: authorizer.AttributesRecord{
					LabelSelectorRequirements: mustLabelRequirement("apple > 1"),
					LabelSelectorParsingErr:   nil,
				},
			},
			want: &authorizationv1.ResourceAttributes{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resourceAttributesFrom(tt.args.attr); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("resourceAttributesFrom() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_shouldCache(t *testing.T) {
	smallSpec := authorizationv1.SubjectAccessReviewSpec{
		User: "alice",
		ResourceAttributes: &authorizationv1.ResourceAttributes{
			Verb: "list", Resource: "pods",
		},
	}
	smallKey, err := json.Marshal(smallSpec)
	if err != nil {
		t.Fatalf("marshal small spec: %v", err)
	}

	tests := []struct {
		name string
		key  []byte
		want bool
	}{
		{"empty", nil, true},
		{"small legitimate", smallKey, true},
		{"just under limit", bytes.Repeat([]byte("a"), maxCacheKeySize-1), true},
		{"at limit", bytes.Repeat([]byte("a"), maxCacheKeySize), false},
		{"over limit", bytes.Repeat([]byte("a"), maxCacheKeySize+1), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldCache(tc.key); got != tc.want {
				t.Fatalf("shouldCache(key len=%d) = %v, want %v", len(tc.key), got, tc.want)
			}
		})
	}
}

// Test_shouldCache_rejectsLargeFieldSelectorFromAttributes reproduces the cache-poisoning
// DoS path: a requester-controlled fieldSelector with thousands of terms produces a SAR
// spec whose JSON key far exceeds maxCacheKeySize, yet the scalar-field check
// used to pass because selector requirements were not counted. The fix measures the key.
func Test_shouldCache_rejectsLargeFieldSelectorFromAttributes(t *testing.T) {
	reqs := make(fields.Requirements, 5000)
	for i := range reqs {
		reqs[i] = fields.Requirement{
			Field:    "spec.nodeName",
			Operator: selection.Equals,
			Value:    fmt.Sprintf("node-%04d", i),
		}
	}
	attr := authorizer.AttributesRecord{
		Verb:                      "list",
		Resource:                  "pods",
		FieldSelectorRequirements: reqs,
	}

	spec := authorizationv1.SubjectAccessReviewSpec{
		User:               "alice",
		ResourceAttributes: resourceAttributesFrom(attr),
	}
	key, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if len(key) <= maxCacheKeySize {
		t.Fatalf("test premise broken: key is only %d bytes, need > %d to exercise the gate", len(key), maxCacheKeySize)
	}
	if shouldCache(key) {
		t.Fatalf("shouldCache returned true for %d-byte fieldSelector key; cache would be poisoned", len(key))
	}
}
