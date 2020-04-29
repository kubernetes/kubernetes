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

// MeterMust is a wrapper for Meter interfaces that panics when any
// instrument constructor encounters an error.
type MeterMust struct {
	meter Meter
}

// Must constructs a MeterMust implementation from a Meter, allowing
// the application to panic when any instrument constructor yields an
// error.
func Must(meter Meter) MeterMust {
	return MeterMust{meter: meter}
}

// NewInt64Counter calls `Meter.NewInt64Counter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64Counter(name string, cos ...Option) Int64Counter {
	if inst, err := mm.meter.NewInt64Counter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64Counter calls `Meter.NewFloat64Counter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64Counter(name string, cos ...Option) Float64Counter {
	if inst, err := mm.meter.NewFloat64Counter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64Measure calls `Meter.NewInt64Measure` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64Measure(name string, mos ...Option) Int64Measure {
	if inst, err := mm.meter.NewInt64Measure(name, mos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64Measure calls `Meter.NewFloat64Measure` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64Measure(name string, mos ...Option) Float64Measure {
	if inst, err := mm.meter.NewFloat64Measure(name, mos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// RegisterInt64Observer calls `Meter.RegisterInt64Observer` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) RegisterInt64Observer(name string, callback Int64ObserverCallback, oos ...Option) Int64Observer {
	if inst, err := mm.meter.RegisterInt64Observer(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// RegisterFloat64Observer calls `Meter.RegisterFloat64Observer` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) RegisterFloat64Observer(name string, callback Float64ObserverCallback, oos ...Option) Float64Observer {
	if inst, err := mm.meter.RegisterFloat64Observer(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}
