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

package e2enode

import (
	"reflect"
	"testing"
)

func TestMemqosCheckControlState(t *testing.T) {
	tests := []struct {
		name     string
		expected map[string]string
		observed map[string]string
		want     memqosConformanceResult
	}{
		{
			name: "fulfilled",
			expected: map[string]string{
				"memory.min":  "0",
				"memory.low":  "134217728",
				"memory.high": "255012864",
				"memory.max":  "268435456",
			},
			observed: map[string]string{
				"memory.min":  "0",
				"memory.low":  "134217728",
				"memory.high": "255012864",
				"memory.max":  "268435456",
			},
			want: memqosConformanceResult{Verdict: memqosConformanceFulfilled},
		},
		{
			name: "drifted",
			expected: map[string]string{
				"memory.high": "255012864",
			},
			observed: map[string]string{
				"memory.high": "max",
			},
			want: memqosConformanceResult{
				Verdict: memqosConformanceDrifted,
				Mismatches: []memqosControlMismatch{{
					Control:  "memory.high",
					Expected: "255012864",
					Observed: "max",
				}},
			},
		},
		{
			name: "missing observed control",
			expected: map[string]string{
				"memory.high": "255012864",
			},
			observed: map[string]string{},
			want:     memqosConformanceResult{Verdict: memqosConformanceInsufficientEvidence},
		},
		{
			name:     "missing expected state",
			expected: map[string]string{},
			observed: map[string]string{"memory.high": "max"},
			want:     memqosConformanceResult{Verdict: memqosConformanceInsufficientEvidence},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := memqosCheckControlState(tt.expected, tt.observed)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("memqosCheckControlState() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
