/*
 * Copyright 2024 gRPC authors.
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

// Package internal contains code internal to the pickfirst package.
package internal

import (
	rand "math/rand/v2"
	"time"
)

var (
	// RandShuffle pseudo-randomizes the order of addresses.
	RandShuffle = rand.Shuffle
	// RandFloat64 returns, as a float64, a pseudo-random number in [0.0,1.0).
	RandFloat64 = rand.Float64
	// TimeAfterFunc allows mocking the timer for testing connection delay
	// related functionality.
	TimeAfterFunc = func(d time.Duration, f func()) func() {
		timer := time.AfterFunc(d, f)
		return func() { timer.Stop() }
	}
)
