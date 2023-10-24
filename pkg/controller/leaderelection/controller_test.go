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

package leaderelection

import (
	"testing"

	v1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCompare(t *testing.T) {
	cases := []struct {
		name           string
		lhs            *v1.Lease
		rhs            *v1.Lease
		expectedResult int
	}{
		{
			name: "identical versions",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: 0,
		},
		{
			name: "no lhs version",
			lhs:  &v1.Lease{},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "no rhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs:            &v1.Lease{},
			expectedResult: 1,
		},
		{
			name: "invalid lhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "xyz",
						BinaryVersionAnnotationName:        "xyz",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "invalid rhs version",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.21",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "xyz",
						BinaryVersionAnnotationName:        "xyz",
					},
				},
			},
			expectedResult: 1,
		},
		{
			name: "lhs less than rhs",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			expectedResult: -1,
		},
		{
			name: "rhs less than lhs",
			lhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.20",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			rhs: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.20",
					},
				},
			},
			expectedResult: 1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := compare(tc.lhs, tc.rhs)
			if result != tc.expectedResult {
				t.Errorf("Expected comparison result of %d but got %d", tc.expectedResult, result)
			}
		})
	}
}

func TestPickLeader(t *testing.T) {
	cases := []struct {
		name               string
		candidates         []*v1.Lease
		expectedLeaderName string
		expectNoLeader     bool
	}{
		{
			name: "same compatibility version, newer binary version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-1",
		},
		{
			name: "same binary version, newer compatibility version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.20",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-2",
		},
		{
			name: "one candidate",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
			},
			expectedLeaderName: "component-identity-1",
		},
		{
			name:           "no candidates",
			candidates:     []*v1.Lease{},
			expectNoLeader: true,
		},
		// TODO: Add test cases where candidates have invalid version numbers
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			leader := pickLeader(tc.candidates)
			if tc.expectNoLeader == true {
				if leader != nil {
					t.Errorf("Expected no leader but got %s", leader.Name)
				}
			} else {
				if leader == nil {
					t.Errorf("Expected leader %s, but got nil leader response", tc.expectedLeaderName)
				} else if leader.Name != tc.expectedLeaderName {
					t.Errorf("Expected leader to be %s but got %s", tc.expectedLeaderName, leader.Name)
				}
			}
		})
	}
}

func TestShouldReelect(t *testing.T) {
	cases := []struct {
		name          string
		candidates    []*v1.Lease
		currentLeader *v1.Lease
		expectResult  bool
	}{
		{
			name: "candidate with newer binary version",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.20",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: true,
		},
		{
			name: "no newer candidates",
			candidates: []*v1.Lease{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
						Annotations: map[string]string{
							CompatibilityVersionAnnotationName: "1.19",
							BinaryVersionAnnotationName:        "1.19",
						},
					},
				},
			},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: false,
		},
		{
			name:       "no candidates",
			candidates: []*v1.Lease{},
			currentLeader: &v1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
					Annotations: map[string]string{
						CompatibilityVersionAnnotationName: "1.19",
						BinaryVersionAnnotationName:        "1.19",
					},
				},
			},
			expectResult: false,
		},
		// TODO: Add test cases where candidates have invalid version numbers
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := shouldReelect(tc.candidates, tc.currentLeader)
			if tc.expectResult != result {
				t.Errorf("Expected %t but got %t", tc.expectResult, result)
			}
		})
	}
}
