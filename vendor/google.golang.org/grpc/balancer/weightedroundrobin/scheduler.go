/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package weightedroundrobin

import (
	"math"
)

type scheduler interface {
	nextIndex() int
}

// newScheduler uses scWeights to create a new scheduler for selecting endpoints
// in a picker.  It will return a round robin implementation if at least
// len(scWeights)-1 are zero or there is only a single endpoint, otherwise it
// will return an Earliest Deadline First (EDF) scheduler implementation that
// selects the endpoints according to their weights.
func (p *picker) newScheduler(recordMetrics bool) scheduler {
	epWeights := p.endpointWeights(recordMetrics)
	n := len(epWeights)
	if n == 0 {
		return nil
	}
	if n == 1 {
		if recordMetrics {
			rrFallbackMetric.Record(p.metricsRecorder, 1, p.target, p.locality)
		}
		return &rrScheduler{numSCs: 1, inc: p.inc}
	}
	sum := float64(0)
	numZero := 0
	max := float64(0)
	for _, w := range epWeights {
		sum += w
		if w > max {
			max = w
		}
		if w == 0 {
			numZero++
		}
	}

	if numZero >= n-1 {
		if recordMetrics {
			rrFallbackMetric.Record(p.metricsRecorder, 1, p.target, p.locality)
		}
		return &rrScheduler{numSCs: uint32(n), inc: p.inc}
	}
	unscaledMean := sum / float64(n-numZero)
	scalingFactor := maxWeight / max
	mean := uint16(math.Round(scalingFactor * unscaledMean))

	weights := make([]uint16, n)
	allEqual := true
	for i, w := range epWeights {
		if w == 0 {
			// Backends with weight = 0 use the mean.
			weights[i] = mean
		} else {
			scaledWeight := uint16(math.Round(scalingFactor * w))
			weights[i] = scaledWeight
			if scaledWeight != mean {
				allEqual = false
			}
		}
	}

	if allEqual {
		return &rrScheduler{numSCs: uint32(n), inc: p.inc}
	}

	logger.Infof("using edf scheduler with weights: %v", weights)
	return &edfScheduler{weights: weights, inc: p.inc}
}

const maxWeight = math.MaxUint16

// edfScheduler implements EDF using the same algorithm as grpc-c++ here:
//
// https://github.com/grpc/grpc/blob/master/src/core/ext/filters/client_channel/lb_policy/weighted_round_robin/static_stride_scheduler.cc
type edfScheduler struct {
	inc     func() uint32
	weights []uint16
}

// Returns the index in s.weights for the picker to choose.
func (s *edfScheduler) nextIndex() int {
	const offset = maxWeight / 2

	for {
		idx := uint64(s.inc())

		// The sequence number (idx) is split in two: the lower %n gives the
		// index of the backend, and the rest gives the number of times we've
		// iterated through all backends. `generation` is used to
		// deterministically decide whether we pick or skip the backend on this
		// iteration, in proportion to the backend's weight.

		backendIndex := idx % uint64(len(s.weights))
		generation := idx / uint64(len(s.weights))
		weight := uint64(s.weights[backendIndex])

		// We pick a backend `weight` times per `maxWeight` generations. The
		// multiply and modulus ~evenly spread out the picks for a given
		// backend between different generations. The offset by `backendIndex`
		// helps to reduce the chance of multiple consecutive non-picks: if we
		// have two consecutive backends with an equal, say, 80% weight of the
		// max, with no offset we would see 1/5 generations that skipped both.
		// TODO(b/190488683): add test for offset efficacy.
		mod := uint64(weight*generation+backendIndex*offset) % maxWeight

		if mod < maxWeight-weight {
			continue
		}
		return int(backendIndex)
	}
}

// A simple RR scheduler to use for fallback when fewer than two backends have
// non-zero weights, or all backends have the same weight, or when only one
// subconn exists.
type rrScheduler struct {
	inc    func() uint32
	numSCs uint32
}

func (s *rrScheduler) nextIndex() int {
	idx := s.inc()
	return int(idx % s.numSCs)
}
