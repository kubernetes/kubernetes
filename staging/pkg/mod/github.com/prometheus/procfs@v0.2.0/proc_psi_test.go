// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"strings"
	"testing"
)

func TestPSIStats(t *testing.T) {
	t.Run("fake", func(*testing.T) {
		stats, err := getProcFixtures(t).PSIStatsForResource("fake")
		if err == nil {
			t.Fatal("fake resource does not have PSI statistics")
		}

		if stats.Some != nil || stats.Full != nil {
			t.Error("a fake resource cannot have PSILine entries")
		}
	})

	t.Run("cpu", func(t *testing.T) {
		stats, err := getProcFixtures(t).PSIStatsForResource("cpu")
		if err != nil {
			t.Fatal(err)
		}

		if stats.Full != nil {
			t.Fatal("cpu resource cannot have 'full' stats")
		}

		if stats.Some == nil {
			t.Fatal("cpu resource should not have nil 'some' stats")
		}

		testCases := []struct {
			name string
			got  float64
			want float64
		}{
			{"Avg10", stats.Some.Avg10, 0.1},
			{"Avg60", stats.Some.Avg60, 2.0},
			{"Avg300", stats.Some.Avg300, 3.85},
			{"Total", float64(stats.Some.Total), 15.0},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				if tc.got != tc.want {
					t.Errorf("got: %f, want: %f", tc.got, tc.want)
				}
			})
		}
	})

	res := []string{"memory", "io"}

	for _, resource := range res {
		t.Run(resource, func(t *testing.T) {
			stats, err := getProcFixtures(t).PSIStatsForResource(resource)
			if err != nil {
				t.Fatal(err)
			}

			if stats.Full == nil {
				t.Fatalf("%s resource must not have nil 'full' stats", resource)
			}

			if stats.Some == nil {
				t.Fatalf("%s resource must not have nil 'some' stats", resource)
			}

			testCases := []struct {
				name string
				got  float64
				want float64
			}{
				{"some/Avg10", stats.Some.Avg10, 0.1},
				{"some/Avg60", stats.Some.Avg60, 2.0},
				{"some/Avg300", stats.Some.Avg300, 3.85},
				{"some/Total", float64(stats.Some.Total), 15.0},
				{"full/Avg10", stats.Full.Avg10, 0.2},
				{"full/Avg60", stats.Full.Avg60, 3.0},
				{"full/Avg300", stats.Full.Avg300, 4.95},
				{"full/Total", float64(stats.Full.Total), 25.0},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					if tc.got != tc.want {
						t.Errorf("got: %f, want: %f", tc.got, tc.want)
					}
				})
			}
		})
	}
}

// TestParsePSIStats tests the edge cases that we won't run into when running TestPSIStats
func TestParsePSIStats(t *testing.T) {
	t.Run("unknown measurement type", func(t *testing.T) {
		raw := "nonesense haha test=fake"
		_, err := parsePSIStats("fake", strings.NewReader(raw))
		if err != nil {
			t.Error("unknown measurement type must be ignored")
		}
	})

	t.Run("malformed measurement", func(t *testing.T) {
		t.Run("some", func(t *testing.T) {
			raw := `some avg10=0.10 avg60=2.00 avg300=3.85 total=oops
full avg10=0.20 avg60=3.00 avg300=teddy total=25`
			stats, err := parsePSIStats("fake", strings.NewReader(raw))
			if err == nil {
				t.Error("a malformed line must result in a parse error")
			}

			if stats.Some != nil || stats.Full != nil {
				t.Error("a parse error must result in a nil PSILine")
			}
		})
		t.Run("full", func(t *testing.T) {
			raw := `some avg10=0.10 avg60=2.00 avg300=3.85 total=1
full avg10=0.20 avg60=3.00 avg300=test total=25`
			stats, err := parsePSIStats("fake", strings.NewReader(raw))
			t.Log(err)
			t.Log(stats)
			if err == nil {
				t.Error("a malformed line must result in a parse error")
			}

			if stats.Some != nil || stats.Full != nil {
				t.Error("a parse error must result in a nil PSILine")
			}
		})

	})
}
