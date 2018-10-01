// Copyright 2017, OpenCensus Authors
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
//

package stats

// Int64Measure is a measure for int64 values.
type Int64Measure struct {
	*measureDescriptor
}

// M creates a new int64 measurement.
// Use Record to record measurements.
func (m *Int64Measure) M(v int64) Measurement {
	return Measurement{m: m.measureDescriptor, v: float64(v)}
}

// Int64 creates a new measure for int64 values.
//
// See the documentation for interface Measure for more guidance on the
// parameters of this function.
func Int64(name, description, unit string) *Int64Measure {
	mi := registerMeasureHandle(name, description, unit)
	return &Int64Measure{mi}
}
