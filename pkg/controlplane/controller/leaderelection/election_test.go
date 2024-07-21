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

package leaderelection

import (
	"testing"
	"time"

	"github.com/blang/semver/v4"
	v1 "k8s.io/api/coordination/v1"
	v1alpha1 "k8s.io/api/coordination/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPickBestLeaderOldestEmulationVersion(t *testing.T) {
	tests := []struct {
		name       string
		candidates []*v1alpha1.LeaseCandidate
		want       *v1alpha1.LeaseCandidate
	}{
		{
			name:       "empty",
			candidates: []*v1alpha1.LeaseCandidate{},
			want:       nil,
		},
		{
			name: "single candidate",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
			},
			want: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
		},
		{
			name: "multiple candidates, different emulation versions",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate2",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.2.0",
						BinaryVersion:    "0.2.0",
					},
				},
			},
			want: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "v1",
					BinaryVersion:    "v1",
				},
			},
		},
		{
			name: "multiple candidates, same emulation versions, different binary versions",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate2",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.2.0",
					},
				},
			},
			want: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
		},
		{
			name: "multiple candidates, same emulation versions, same binary versions, different creation timestamps",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate2",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
			},
			want: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pickBestLeaderOldestEmulationVersion(tt.candidates)
			if got != nil && tt.want != nil {
				if got.Name != tt.want.Name || got.Namespace != tt.want.Namespace {
					t.Errorf("pickBestLeaderOldestEmulationVersion() = %v, want %v", got, tt.want)
				}
			} else if got != tt.want {
				t.Errorf("pickBestLeaderOldestEmulationVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValidLeaseCandidateForOldestEmulationVersion(t *testing.T) {
	tests := []struct {
		name      string
		candidate *v1alpha1.LeaseCandidate
		want      bool
	}{
		{
			name: "valid emulation and binary versions",
			candidate: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
			want: true,
		},
		{
			name: "invalid emulation version",
			candidate: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "invalid",
					BinaryVersion:    "0.1.0",
				},
			},
			want: false,
		},
		{
			name: "invalid binary version",
			candidate: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "invalid",
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validLeaseCandidateForOldestEmulationVersion(tt.candidate)
			if got != tt.want {
				t.Errorf("validLeaseCandidateForOldestEmulationVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetEmulationVersion(t *testing.T) {
	tests := []struct {
		name      string
		candidate *v1alpha1.LeaseCandidate
		want      semver.Version
	}{
		{
			name: "valid emulation version",
			candidate: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
				},
			},
			want: semver.MustParse("0.1.0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getEmulationVersion(tt.candidate)
			if got.FinalizeVersion() != tt.want.FinalizeVersion() {
				t.Errorf("getEmulationVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetBinaryVersion(t *testing.T) {
	tests := []struct {
		name      string
		candidate *v1alpha1.LeaseCandidate
		want      semver.Version
	}{
		{
			name: "valid binary version",
			candidate: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					BinaryVersion: "0.3.0",
				},
			},
			want: semver.MustParse("0.3.0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBinaryVersion(tt.candidate)
			if got.FinalizeVersion() != tt.want.FinalizeVersion() {
				t.Errorf("getBinaryVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCompare(t *testing.T) {
	nowTime := time.Now()
	cases := []struct {
		name           string
		lhs            *v1alpha1.LeaseCandidate
		rhs            *v1alpha1.LeaseCandidate
		expectedResult int
	}{
		{
			name: "identical versions earlier timestamp",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.Time{Time: nowTime.Add(time.Duration(1))},
				},
			},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.Time{Time: nowTime},
				},
			},
			expectedResult: 1,
		},
		{
			name: "no lhs version",
			lhs:  &v1alpha1.LeaseCandidate{},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "no rhs version",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			rhs:            &v1alpha1.LeaseCandidate{},
			expectedResult: 1,
		},
		{
			name: "invalid lhs version",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "xyz",
					BinaryVersion:    "xyz",
				},
			},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "invalid rhs version",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "xyz",
					BinaryVersion:    "xyz",
				},
			},
			expectedResult: 1,
		},
		{
			name: "lhs less than rhs",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.20.0",
				},
			},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.20.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "rhs less than lhs",
			lhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.20.0",
				},
			},
			rhs: &v1alpha1.LeaseCandidate{
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.20.0",
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

func TestShouldReelect(t *testing.T) {
	cases := []struct {
		name          string
		candidates    []*v1alpha1.LeaseCandidate
		currentLeader *v1alpha1.LeaseCandidate
		expectResult  bool
	}{
		{
			name: "candidate with newer binary version",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.20.0",
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			currentLeader: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion:    "1.19.0",
					BinaryVersion:       "1.19.0",
					PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
				},
			},
			expectResult: false,
		},
		{
			name: "no newer candidates",
			candidates: []*v1alpha1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
					},
					Spec: v1alpha1.LeaseCandidateSpec{
						EmulationVersion:    "1.19.0",
						BinaryVersion:       "1.19.0",
						PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
					},
				},
			},
			currentLeader: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion:    "1.19.0",
					BinaryVersion:       "1.19.0",
					PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
				},
			},
			expectResult: false,
		},
		{
			name:       "no candidates",
			candidates: []*v1alpha1.LeaseCandidate{},
			currentLeader: &v1alpha1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1alpha1.LeaseCandidateSpec{
					EmulationVersion:    "1.19.0",
					BinaryVersion:       "1.19.0",
					PreferredStrategies: []v1.CoordinatedLeaseStrategy{v1.OldestEmulationVersion},
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
