// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package trace

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type SamplingPolicy interface {
	// Sample determines whether to sample the next request.  If so, it also
	// returns a string and rate describing the reason the request was chosen.
	Sample() (sample bool, policy string, rate float64)
}

type sampler struct {
	fraction float64
	*rate.Limiter
	*rand.Rand
	sync.Mutex
}

func (s *sampler) Sample() (sample bool, reason string, rate float64) {
	s.Lock()
	x := s.Float64()
	s.Unlock()
	return s.sample(time.Now(), x)
}

// sample contains the a deterministic, time-independent logic of Sample.
func (s *sampler) sample(now time.Time, x float64) (bool, string, float64) {
	if x >= s.fraction || !s.AllowN(now, 1) {
		return false, "", 0.0
	}
	if s.fraction < 1.0 {
		return true, "fraction", s.fraction
	}
	return true, "qps", float64(s.Limit())
}

// NewLimitedSampler returns a sampling policy that traces a given fraction of
// requests, and enforces a limit on the number of traces per second.
// Returns a nil SamplingPolicy if either fraction or maxqps is zero.
func NewLimitedSampler(fraction, maxqps float64) (SamplingPolicy, error) {
	if !(fraction >= 0) {
		return nil, fmt.Errorf("invalid fraction %f", fraction)
	}
	if !(maxqps >= 0) {
		return nil, fmt.Errorf("invalid maxqps %f", maxqps)
	}
	if fraction == 0 || maxqps == 0 {
		return nil, nil
	}
	// Set a limit on the number of accumulated "tokens", to limit bursts of
	// traced requests.  Use one more than a second's worth of tokens, or 100,
	// whichever is smaller.
	// See https://godoc.org/golang.org/x/time/rate#NewLimiter.
	maxTokens := 100
	if maxqps < 99.0 {
		maxTokens = 1 + int(maxqps)
	}
	var seed int64
	if err := binary.Read(crand.Reader, binary.LittleEndian, &seed); err != nil {
		seed = time.Now().UnixNano()
	}
	s := sampler{
		fraction: fraction,
		Limiter:  rate.NewLimiter(rate.Limit(maxqps), maxTokens),
		Rand:     rand.New(rand.NewSource(seed)),
	}
	return &s, nil
}
