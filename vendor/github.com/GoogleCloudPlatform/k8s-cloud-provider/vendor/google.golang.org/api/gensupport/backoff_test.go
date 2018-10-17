// Copyright 2017 Google Inc. All Rights Reserved.
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

package gensupport

import (
	"testing"
	"time"
)

func TestBackoff(t *testing.T) {
	eb := &ExponentialBackoff{Base: time.Millisecond, Max: time.Second}

	var total time.Duration
	for n, max := 0, 2*time.Millisecond; ; n, max = n+1, max*2 {
		if n > 100 {
			// There's less than 1 in 10^28 of taking longer than 100 iterations,
			// so this is just to check we don't have an infinite loop.
			t.Fatalf("Failed to timeout after 100 iterations.")
		}
		pause, retry := eb.Pause()
		if !retry {
			break
		}

		if 0 > pause || pause >= max {
			t.Errorf("Iteration %d: pause = %v; want in range [0, %v)", n, pause, max)
		}
		total += pause
	}

	if total < time.Second {
		t.Errorf("Total time = %v; want > %v", total, time.Second)
	}
}
