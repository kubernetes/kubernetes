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

package internal

import (
	"go.opencensus.io/tag"
)

// DefaultRecorder will be called for each Record call.
var DefaultRecorder func(tags *tag.Map, measurement interface{}, attachments map[string]interface{})

// MeasurementRecorder will be called for each Record call. This is the same as DefaultRecorder but
// avoids interface{} conversion.
// This will be a func(tags *tag.Map, measurement []Measurement, attachments map[string]interface{}) type,
// but is interface{} here to avoid import loops
var MeasurementRecorder interface{}

// SubscriptionReporter reports when a view subscribed with a measure.
var SubscriptionReporter func(measure string)
