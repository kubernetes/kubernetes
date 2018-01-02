// Copyright 2016 CoreOS, Inc.
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

package daemon

import (
	"os"
	"strconv"
	"testing"
	"time"
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func TestSdWatchdogEnabled(t *testing.T) {
	mypid := strconv.Itoa(os.Getpid())
	tests := []struct {
		usec     string // empty => unset
		pid      string // empty => unset
		unsetEnv bool   // arbitrarily set across testcases

		werr   bool
		wdelay time.Duration
	}{
		// Success cases
		{"100", mypid, true, false, 100 * time.Microsecond},
		{"50", mypid, true, false, 50 * time.Microsecond},
		{"1", mypid, false, false, 1 * time.Microsecond},
		{"1", "", true, false, 1 * time.Microsecond},

		// No-op cases
		{"", mypid, true, false, 0}, // WATCHDOG_USEC not set
		{"1", "0", false, false, 0}, // WATCHDOG_PID doesn't match
		{"", "", true, false, 0},    // Both not set

		// Failure cases
		{"-1", mypid, true, true, 0},                // Negative USEC
		{"string", "1", false, true, 0},             // Non-integer USEC value
		{"1", "string", true, true, 0},              // Non-integer PID value
		{"stringa", "stringb", false, true, 0},      // E v e r y t h i n g
		{"-10239", "-eleventythree", true, true, 0}, // i s   w r o n g
	}

	for i, tt := range tests {
		if tt.usec != "" {
			must(os.Setenv("WATCHDOG_USEC", tt.usec))
		} else {
			must(os.Unsetenv("WATCHDOG_USEC"))
		}
		if tt.pid != "" {
			must(os.Setenv("WATCHDOG_PID", tt.pid))
		} else {
			must(os.Unsetenv("WATCHDOG_PID"))
		}

		delay, err := SdWatchdogEnabled(tt.unsetEnv)

		if tt.werr && err == nil {
			t.Errorf("#%d: want non-nil err, got nil", i)
		} else if !tt.werr && err != nil {
			t.Errorf("#%d: want nil err, got %v", i, err)
		}
		if tt.wdelay != delay {
			t.Errorf("#%d: want delay=%d, got %d", i, tt.wdelay, delay)
		}
		if tt.unsetEnv && (os.Getenv("WATCHDOG_PID") != "" || os.Getenv("WATCHDOG_USEC") != "") {
			t.Errorf("#%d: environment variables not cleaned up", i)
		}
	}
}
