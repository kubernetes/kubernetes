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

package create

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPopulateResourceListV1(t *testing.T) {

	tests := map[string]struct {
		hard      string
		expected  corev1.ResourceList
		expectErr bool
	}{
		"single resource": {
			hard: "cpu=1",
			expected: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
		},
		"single resource with no name": {
			hard:      "=1",
			expectErr: true,
		},
		"double resource with no name in one": {
			hard:      "cpu=1,=2",
			expectErr: true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := populateResourceListV1(tc.hard)
			if tc.expectErr {
				if err == nil {
					t.Errorf("unexpected error:\n%#v\n", err)
					return
				}
			}

			if !apiequality.Semantic.DeepEqual(got, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, got)
			}
		})
	}
}

func TestCreateQuota(t *testing.T) {
	hards := []string{"cpu=1", "cpu=1,pods=42"}
	var resourceQuotaSpecLists []corev1.ResourceList
	for _, hard := range hards {
		resourceQuotaSpecList, err := populateResourceListV1(hard)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		resourceQuotaSpecLists = append(resourceQuotaSpecLists, resourceQuotaSpecList)
	}

	tests := map[string]struct {
		options  *QuotaOpts
		expected *corev1.ResourceQuota
	}{
		"single resource": {
			options: &QuotaOpts{
				Name:   "my-quota",
				Hard:   hards[0],
				Scopes: "",
			},
			expected: &corev1.ResourceQuota{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ResourceQuota",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-quota",
				},
				Spec: corev1.ResourceQuotaSpec{
					Hard: resourceQuotaSpecLists[0],
				},
			},
		},
		"single resource with a scope": {
			options: &QuotaOpts{
				Name:   "my-quota",
				Hard:   hards[0],
				Scopes: "BestEffort",
			},
			expected: &corev1.ResourceQuota{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ResourceQuota",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-quota",
				},
				Spec: corev1.ResourceQuotaSpec{
					Hard:   resourceQuotaSpecLists[0],
					Scopes: []corev1.ResourceQuotaScope{"BestEffort"},
				},
			},
		},
		"multiple resources": {
			options: &QuotaOpts{
				Name:   "my-quota",
				Hard:   hards[1],
				Scopes: "BestEffort",
			},
			expected: &corev1.ResourceQuota{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ResourceQuota",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-quota",
				},
				Spec: corev1.ResourceQuotaSpec{
					Hard:   resourceQuotaSpecLists[1],
					Scopes: []corev1.ResourceQuotaScope{"BestEffort"},
				},
			},
		},
		"single resource with multiple scopes": {
			options: &QuotaOpts{
				Name:   "my-quota",
				Hard:   hards[0],
				Scopes: "BestEffort,NotTerminating",
			},
			expected: &corev1.ResourceQuota{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ResourceQuota",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "my-quota",
				},
				Spec: corev1.ResourceQuotaSpec{
					Hard:   resourceQuotaSpecLists[0],
					Scopes: []corev1.ResourceQuotaScope{"BestEffort", "NotTerminating"},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			resourceQuota, err := tc.options.createQuota()
			if err != nil {
				t.Errorf("unexpected error:\n%#v\n", err)
				return
			}
			if !apiequality.Semantic.DeepEqual(resourceQuota, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, resourceQuota)
			}
		})
	}
}
