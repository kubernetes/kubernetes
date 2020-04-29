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

// metric package provides an API for reporting diagnostic
// measurements using four basic kinds of instruments.
//
// The three basic kinds are:
//
// - counters
// - measures
// - observers
//
// All instruments report either float64 or int64 values.
//
// The primary object that handles metrics is Meter. Meter can be
// obtained from Provider. The implementations of the Meter and
// Provider are provided by SDK. Normally, the Meter is used directly
// only for the instrument creation and batch recording.
//
// Counters are instruments that are reporting a quantity or a sum. An
// example could be bank account balance or bytes downloaded. Counters
// can be created with either NewFloat64Counter or
// NewInt64Counter. Counters expect non-negative values by default to
// be reported. This can be changed with the WithMonotonic option
// (passing false as a parameter) passed to the Meter.New*Counter
// function - this allows reporting negative values. To report the new
// value, use an Add function.
//
// Measures are instruments that are reporting values that are
// recorded separately to figure out some statistical properties from
// those values (like average). An example could be temperature over
// time or lines of code in the project over time. Measures can be
// created with either NewFloat64Measure or NewInt64Measure. Measures
// by default take only non-negative values. This can be changed with
// the WithAbsolute option (passing false as a parameter) passed to
// the New*Measure function - this allows reporting negative values
// too. To report a new value, use the Record function.
//
// Observers are instruments that are reporting a current state of a
// set of values. An example could be voltage or
// temperature. Observers can be created with either
// RegisterFloat64Observer or RegisterInt64Observer. Observers by
// default have no limitations about reported values - they can be
// less or greater than the last reported value. This can be changed
// with the WithMonotonic option passed to the Register*Observer
// function - this permits the reported values only to go
// up. Reporting of the new values happens asynchronously, with the
// use of a callback passed to the Register*Observer function. The
// callback can report multiple values. There is no unregister function.
//
// Counters and measures support creating bound instruments for a
// potentially more efficient reporting. The bound instruments have
// the same function names as the instruments (so a Counter bound
// instrument has Add, and a Measure bound instrument has Record).
// Bound Instruments can be created with the Bind function of the
// respective instrument. When done with the bound instrument, call
// Unbind on it.
package metric // import "go.opentelemetry.io/otel/api/metric"
