// Copyright 2018, OpenCensus Authors
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

package metricdata

import (
	"time"
)

// Exemplars keys.
const (
	AttachmentKeySpanContext = "SpanContext"
)

// Exemplar is an example data point associated with each bucket of a
// distribution type aggregation.
//
// Their purpose is to provide an example of the kind of thing
// (request, RPC, trace span, etc.) that resulted in that measurement.
type Exemplar struct {
	Value       float64     // the value that was recorded
	Timestamp   time.Time   // the time the value was recorded
	Attachments Attachments // attachments (if any)
}

// Attachments is a map of extra values associated with a recorded data point.
type Attachments map[string]interface{}
