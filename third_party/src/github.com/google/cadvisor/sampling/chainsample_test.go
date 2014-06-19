// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sampling

import "testing"

func TestChainSampler(t *testing.T) {
	numSamples := 10
	windowSize := 10 * numSamples
	numObservations := 10 * windowSize
	numSampleRounds := 10 * numObservations

	s := NewChainSampler(numSamples, windowSize)
	hist := make(map[int]int, numSamples)
	for i := 0; i < numSampleRounds; i++ {
		sampleStream(hist, numObservations, s)
	}
	ratio := histStddev(hist) / histMean(hist)
	if ratio > 1.05 {
		// XXX(dengnan): better sampler?
		t.Errorf("std dev: %v; mean: %v. Either we have a really bad PRNG, or a bad implementation", histStddev(hist), histMean(hist))
	}
	if len(hist) > windowSize {
		t.Errorf("sampled %v data. larger than window size %v", len(hist), windowSize)
	}
	for seqNum, freq := range hist {
		if seqNum < numObservations-windowSize && freq > 0 {
			t.Errorf("observation with seqnum %v is sampled %v times", seqNum, freq)
		}
	}
}
