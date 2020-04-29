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

package metric

import "go.opentelemetry.io/otel/api/core"

// Int64ObserverResult is an interface for reporting integral
// observations.
type Int64ObserverResult interface {
	Observe(value int64, labels ...core.KeyValue)
}

// Float64ObserverResult is an interface for reporting floating point
// observations.
type Float64ObserverResult interface {
	Observe(value float64, labels ...core.KeyValue)
}

// Int64ObserverCallback is a type of callback that integral
// observers run.
type Int64ObserverCallback func(result Int64ObserverResult)

// Float64ObserverCallback is a type of callback that floating point
// observers run.
type Float64ObserverCallback func(result Float64ObserverResult)

// Int64Observer is a metric that captures a set of int64 values at a
// point in time.
type Int64Observer struct {
	asyncInstrument
}

// Float64Observer is a metric that captures a set of float64 values
// at a point in time.
type Float64Observer struct {
	asyncInstrument
}
