// Copyright The OpenTelemetry Authors
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

package internal // import "go.opentelemetry.io/otel/sdk/internal"

import (
	"fmt"
	"time"

	opentelemetry "go.opentelemetry.io/otel/sdk"
)

// UserAgent is the user agent to be added to the outgoing
// requests from the exporters.
var UserAgent = fmt.Sprintf("opentelemetry-go/%s", opentelemetry.Version())

// MonotonicEndTime returns the end time at present
// but offset from start, monotonically.
//
// The monotonic clock is used in subtractions hence
// the duration since start added back to start gives
// end as a monotonic time.
// See https://golang.org/pkg/time/#hdr-Monotonic_Clocks
func MonotonicEndTime(start time.Time) time.Time {
	return start.Add(time.Since(start))
}
