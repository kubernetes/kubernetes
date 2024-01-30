/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package halfconn

import "errors"

// counter is a 64-bit counter.
type counter struct {
	val           uint64
	hasOverflowed bool
}

// newCounter creates a new counter with the initial value set to val.
func newCounter(val uint64) counter {
	return counter{val: val}
}

// value returns the current value of the counter.
func (c *counter) value() (uint64, error) {
	if c.hasOverflowed {
		return 0, errors.New("counter has overflowed")
	}
	return c.val, nil
}

// increment increments the counter and checks for overflow.
func (c *counter) increment() {
	// If the counter is already invalid due to overflow, there is no need to
	// increase it. We check for the hasOverflowed flag in the call to value().
	if c.hasOverflowed {
		return
	}
	c.val++
	if c.val == 0 {
		c.hasOverflowed = true
	}
}

// reset sets the counter value to zero and sets the hasOverflowed flag to
// false.
func (c *counter) reset() {
	c.val = 0
	c.hasOverflowed = false
}
