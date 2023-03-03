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

package metrics

// RatioedGaugeVecPhasedElementPair extracts a pair of elements that differ in handling phase
func RatioedGaugeVecPhasedElementPair(vec RatioedGaugeVec, initialWaitingDenominator, initialExecutingDenominator float64, labelValues []string) RatioedGaugePair {
	return RatioedGaugePair{
		RequestsWaiting:   vec.NewForLabelValuesSafe(0, initialWaitingDenominator, append([]string{LabelValueWaiting}, labelValues...)),
		RequestsExecuting: vec.NewForLabelValuesSafe(0, initialExecutingDenominator, append([]string{LabelValueExecuting}, labelValues...)),
	}
}
