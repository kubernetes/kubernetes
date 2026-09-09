// Copyright 2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build unix

package daemon

import (
	"strconv"

	"golang.org/x/sys/unix"
)

// SdNotifyMonotonicUsec returns a MONOTONIC_USEC=... assignment for the current time
// with a trailing newline included. This is typically used with [SdNotifyReloading].
//
// If the monotonic clock is not available on the system, the empty string is returned.
func SdNotifyMonotonicUsec() string {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts); err != nil {
		// Monotonic clock is not available on this system.
		return ""
	}
	return "MONOTONIC_USEC=" + strconv.FormatInt(ts.Nano()/1000, 10) + "\n"
}
