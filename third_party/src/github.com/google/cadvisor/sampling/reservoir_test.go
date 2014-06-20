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

import (
	"math"
	"testing"
)

func sampleStream(hist map[int]int, n int, s Sampler) {
	s.Reset()
	for i := 0; i < n; i++ {
		s.Update(i)
	}
	s.Map(func(d interface{}) {
		j := d.(int)
		if _, ok := hist[j]; !ok {
			hist[j] = 0
		}
		hist[j]++
	})
}

func histMean(hist map[int]int) float64 {
	total := 0
	for _, v := range hist {
		total += v
	}
	return float64(total) / float64(len(hist))
}

func histStddev(hist map[int]int) float64 {
	mean := histMean(hist)
	var totalDiff float64
	for _, v := range hist {
		diff := float64(v) - mean
		sq := diff * diff
		totalDiff += sq
	}
	return math.Sqrt(totalDiff / float64(len(hist)))
}

// XXX(dengnan): This test may take more than 10 seconds.
func TestReservoirSampler(t *testing.T) {
	reservoirSize := 10
	numSamples := 10 * reservoirSize
	numSampleRounds := 100 * numSamples

	s := NewReservoirSampler(reservoirSize)
	hist := make(map[int]int, numSamples)
	for i := 0; i < numSampleRounds; i++ {
		sampleStream(hist, numSamples, s)
	}
	ratio := histStddev(hist) / histMean(hist)
	if ratio > 0.05 {
		t.Errorf("std dev: %v; mean: %v. Either we have a really bad PRNG, or a bad implementation", histStddev(hist), histMean(hist))
	}
}
