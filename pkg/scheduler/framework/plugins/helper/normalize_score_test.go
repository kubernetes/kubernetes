/*
Copyright 2019 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestDefaultNormalizeScore(t *testing.T) {
	tests := []struct {
		reverse        bool
		scores         []int64
		expectedScores []int64
	}{
		{
			scores:         []int64{1, 2, 3, 4},
			expectedScores: []int64{25, 50, 75, 100},
		},
		{
			reverse:        true,
			scores:         []int64{1, 2, 3, 4},
			expectedScores: []int64{75, 50, 25, 0},
		},
		{
			scores:         []int64{1000, 10, 20, 30},
			expectedScores: []int64{100, 1, 2, 3},
		},
		{
			reverse:        true,
			scores:         []int64{1000, 10, 20, 30},
			expectedScores: []int64{0, 99, 98, 97},
		},
		{
			scores:         []int64{1, 1, 1, 1},
			expectedScores: []int64{100, 100, 100, 100},
		},
		{
			scores:         []int64{1000, 1, 1, 1},
			expectedScores: []int64{100, 0, 0, 0},
		},
		{
			reverse:        true,
			scores:         []int64{0, 1, 1, 1},
			expectedScores: []int64{100, 0, 0, 0},
		},
	}

	for i, test := range tests {
		scores := framework.NodeScoreList{}
		for _, score := range test.scores {
			scores = append(scores, framework.NodeScore{Score: score})
		}

		expectedScores := framework.NodeScoreList{}
		for _, score := range test.expectedScores {
			expectedScores = append(expectedScores, framework.NodeScore{Score: score})
		}

		DefaultNormalizeScore(framework.MaxNodeScore, test.reverse, scores)
		if !reflect.DeepEqual(scores, expectedScores) {
			t.Errorf("test %d, expected %v, got %v", i, expectedScores, scores)
		}
	}
}
