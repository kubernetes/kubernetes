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
	v1beta1 "k8s.io/api/coordination/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPickBestLeaderOldestEmulationVersion(t *testing.T) {
	tests := []struct {
		name       string
		candidates []*v1beta1.LeaseCandidate
		want       *v1beta1.LeaseCandidate
	}{
		{
			name:       "empty",
			candidates: []*v1beta1.LeaseCandidate{},
			want:       nil,
		},
		{
			name: "single candidate",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now()},
					},
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
			},
			want: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
		},
		{
			name: "multiple candidates, different emulation versions",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1beta1.LeaseCandidateSpec{
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
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "0.2.0",
						BinaryVersion:    "0.2.0",
					},
				},
			},
			want: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "v1",
					BinaryVersion:    "v1",
				},
			},
		},
		{
			name: "multiple candidates, same emulation versions, different binary versions",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1beta1.LeaseCandidateSpec{
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
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.2.0",
					},
				},
			},
			want: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
		},
		{
			name: "multiple candidates, same emulation versions, same binary versions, different creation timestamps",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "candidate1",
						Namespace:         "default",
						CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)},
					},
					Spec: v1beta1.LeaseCandidateSpec{
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
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "0.1.0",
						BinaryVersion:    "0.1.0",
					},
				},
			},
			want: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "candidate1",
					Namespace: "default",
				},
				Spec: v1beta1.LeaseCandidateSpec{
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
		candidate *v1beta1.LeaseCandidate
		want      bool
	}{
		{
			name: "valid emulation and binary versions",
			candidate: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
					BinaryVersion:    "0.1.0",
				},
			},
			want: true,
		},
		{
			name: "invalid emulation version",
			candidate: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "invalid",
					BinaryVersion:    "0.1.0",
				},
			},
			want: false,
		},
		{
			name: "invalid binary version",
			candidate: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
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
		candidate *v1beta1.LeaseCandidate
		want      semver.Version
	}{
		{
			name: "valid emulation version",
			candidate: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "0.1.0",
				},
			},
			want: semver.MustParse("0.1.0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getEmulationVersionOrZero(tt.candidate)
			if got.FinalizeVersion() != tt.want.FinalizeVersion() {
				t.Errorf("getEmulationVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetBinaryVersion(t *testing.T) {
	tests := []struct {
		name      string
		candidate *v1beta1.LeaseCandidate
		want      semver.Version
	}{
		{
			name: "valid binary version",
			candidate: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					BinaryVersion: "0.3.0",
				},
			},
			want: semver.MustParse("0.3.0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getBinaryVersionOrZero(tt.candidate)
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
		lhs            *v1beta1.LeaseCandidate
		rhs            *v1beta1.LeaseCandidate
		expectedResult int
	}{
		{
			name: "identical versions earlier timestamp",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.Time{Time: nowTime.Add(time.Duration(1))},
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
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
			lhs:  &v1beta1.LeaseCandidate{},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "no rhs version",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			rhs:            &v1beta1.LeaseCandidate{},
			expectedResult: 1,
		},
		{
			name: "invalid lhs version",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "xyz",
					BinaryVersion:    "xyz",
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "invalid rhs version",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.21.0",
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "xyz",
					BinaryVersion:    "xyz",
				},
			},
			expectedResult: 1,
		},
		{
			name: "lhs less than rhs",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.20.0",
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.20.0",
				},
			},
			expectedResult: -1,
		},
		{
			name: "rhs less than lhs",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.20.0",
					BinaryVersion:    "1.20.0",
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.20.0",
				},
			},
			expectedResult: 1,
		},
		{
			name: "lhs less than rhs, lexographical order check",
			lhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.2.0",
					BinaryVersion:    "1.20.0",
				},
			},
			rhs: &v1beta1.LeaseCandidate{
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.20.0",
				},
			},
			expectedResult: -1,
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
		candidates    []*v1beta1.LeaseCandidate
		currentLeader *v1beta1.LeaseCandidate
		expectResult  bool
	}{
		{
			name: "candidate with newer binary version",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						Strategy:         v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.20.0",
						Strategy:         v1.OldestEmulationVersion,
					},
				},
			},
			currentLeader: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.19.0",
					Strategy:         v1.OldestEmulationVersion,
				},
			},
			expectResult: false,
		},
		{
			name: "no newer candidates",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-1",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						Strategy:         v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "component-identity-2",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						EmulationVersion: "1.19.0",
						BinaryVersion:    "1.19.0",
						Strategy:         v1.OldestEmulationVersion,
					},
				},
			},
			currentLeader: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.19.0",
					Strategy:         v1.OldestEmulationVersion,
				},
			},
			expectResult: false,
		},
		{
			name:       "no candidates",
			candidates: []*v1beta1.LeaseCandidate{},
			currentLeader: &v1beta1.LeaseCandidate{
				ObjectMeta: metav1.ObjectMeta{
					Name: "component-identity-1",
				},
				Spec: v1beta1.LeaseCandidateSpec{
					EmulationVersion: "1.19.0",
					BinaryVersion:    "1.19.0",
					Strategy:         v1.OldestEmulationVersion,
				},
			},
			expectResult: false,
		},
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

func TestPickBestStrategy(t *testing.T) {
	tests := []struct {
		name         string
		candidates   []*v1beta1.LeaseCandidate
		wantStrategy v1.CoordinatedLeaseStrategy
		wantError    bool
	}{
		{
			name: "single candidate, single preferred strategy",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName: "component-A",
						Strategy:  v1.OldestEmulationVersion,
					},
				},
			},
			wantStrategy: v1.OldestEmulationVersion,
			wantError:    false,
		},
		{
			name: "multiple candidates, different preferred strategies should fail",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName: "component-A",
						Strategy:  v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName: "component-A",
						Strategy:  v1.CoordinatedLeaseStrategy("foo.com/bar"),
					},
				},
			},
			wantError: true,
		},
		{
			name: "multiple candidates, different preferred strategy different binary version should resolve",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.32.0",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.CoordinatedLeaseStrategy("foo.com/bar"),
					},
				},
			},
			wantStrategy: v1.OldestEmulationVersion,
			wantError:    false,
		},
		{
			name: "multiple candidates, different preferred strategy different binary version should resolve, order agnostic",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.CoordinatedLeaseStrategy("foo.com/bar"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.32.0",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
			},
			wantStrategy: v1.OldestEmulationVersion,
			wantError:    false,
		},
		{
			name: "multiple candidates, different preferred strategy different binary version string comparison check",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.1.10",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.1.2",
						Strategy:      v1.CoordinatedLeaseStrategy("foo.com/bar"),
					},
				},
			},
			wantStrategy: v1.OldestEmulationVersion,
			wantError:    false,
		},

		{
			name: "multiple candidates, same preferred strategy",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
			},
			wantStrategy: v1.OldestEmulationVersion,
			wantError:    false,
		},
		{
			name: "multiple candidates, conflicting preferred strategy",
			candidates: []*v1beta1.LeaseCandidate{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate1",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.OldestEmulationVersion,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "candidate2",
						Namespace: "default",
					},
					Spec: v1beta1.LeaseCandidateSpec{
						LeaseName:     "component-A",
						BinaryVersion: "1.31.0",
						Strategy:      v1.CoordinatedLeaseStrategy("foo.com/bar"),
					},
				},
			},
			wantStrategy: "",
			wantError:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotStrategy, err := pickBestStrategy(tc.candidates)
			gotError := err != nil
			if gotError != tc.wantError {
				t.Errorf("pickBestStrategy() error = %v,:%v want %v", gotError, err, tc.wantError)
			}
			if !gotError && gotStrategy != tc.wantStrategy {
				t.Errorf("pickBestStrategy() = %v, want %v", gotStrategy, tc.wantStrategy)
			}
		})
	}
}

func shouldReelect(candidates []*v1beta1.LeaseCandidate, currentLeader *v1beta1.LeaseCandidate) bool {
	pickedLeader := pickBestLeaderOldestEmulationVersion(candidates)
	if pickedLeader == nil {
		return false
	}
	return compare(currentLeader, pickedLeader) > 0
}
