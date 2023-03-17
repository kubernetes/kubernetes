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

package benchmark

import (
	"strings"
	"testing"
)

func TestLabelFilter(t *testing.T) {
	empty := ""
	performance := "performance"
	fastPerformance := "performance,fast"
	notFastPerformance := "+performance,-fast"
	notFast := "-fast"

	testcases := map[string]map[string]bool{
		empty: {
			empty:           true,
			performance:     true,
			fastPerformance: true,
		},
		performance: {
			empty:           false,
			performance:     true,
			fastPerformance: true,
		},
		fastPerformance: {
			empty:           false,
			performance:     false,
			fastPerformance: true,
		},
		notFast: {
			empty:           true,
			performance:     true,
			fastPerformance: false,
		},
		notFastPerformance: {
			empty:           false,
			performance:     true,
			fastPerformance: false,
		},
	}

	for labelFilter, labelResults := range testcases {
		t.Run(labelFilter, func(t *testing.T) {
			for labels, expected := range labelResults {
				t.Run(labels, func(t *testing.T) {
					actual := enabled(labelFilter, strings.Split(labels, ",")...)
					if actual != expected {
						t.Errorf("expected enabled to be %v, got %v", expected, actual)
					}
				})
			}
		})
	}
}
