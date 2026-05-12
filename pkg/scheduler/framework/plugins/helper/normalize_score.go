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
	fwk "k8s.io/kube-scheduler/framework"
)

// DefaultNormalizeScore generates a Normalize Score function that can normalize the
// scores from [0, max(scores)] to [0, maxPriority]. If reverse is set to true, it
// reverses the scores by subtracting it from maxPriority.
// Note: The input scores are always assumed to be non-negative integers.
func DefaultNormalizeScore(maxPriority int64, reverse bool, scores fwk.NodeScoreList) *fwk.Status {
	var maxCount int64
	for i := range scores {
		if scores[i].Score > maxCount {
			maxCount = scores[i].Score
		}
	}

	if maxCount == 0 {
		if reverse {
			for i := range scores {
				scores[i].Score = maxPriority
			}
		}
		return nil
	}

	for i := range scores {
		score := scores[i].Score

		score = maxPriority * score / maxCount
		if reverse {
			score = maxPriority - score
		}

		scores[i].Score = score
	}
	return nil
}
