/*
Copyright 2023 The Kubernetes Authors.

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

package equality_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestSemanticDeepEqual(t *testing.T) {
	for _, tc := range []struct {
		a, b     interface{}
		expected bool
	}{
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "j",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a"},
			},
			expected: false,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a", "b"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"b", "a"},
			},
			expected: true,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a", "a"},
			},
			expected: true,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpNotIn,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpNotIn,
				Values:   []string{"a", "a"},
			},
			expected: true,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpNotIn,
				Values:   []string{"a", "a"},
			},
			expected: false,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpExists,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpExists,
				Values:   []string{"a", "a"},
			},
			expected: false,
		},
		{
			a: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpDoesNotExist,
				Values:   []string{"a"},
			},
			b: metav1.LabelSelectorRequirement{
				Key:      "k",
				Operator: metav1.LabelSelectorOpDoesNotExist,
				Values:   []string{"a", "a"},
			},
			expected: false,
		},
	} {
		if actual := equality.Semantic.DeepEqual(tc.a, tc.b); actual != tc.expected {
			t.Errorf("Semantic.DeepEqual(%#v, %#v) did not return %t", tc.a, tc.b, tc.expected)
		}
	}
}
