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

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"go.opentelemetry.io/otel/unit"
)

// InstrumentConfig contains options for metric instrument descriptors.
type InstrumentConfig struct {
	// Description describes the instrument in human-readable terms.
	Description string
	// Unit describes the measurement unit for a instrument.
	Unit unit.Unit
	// InstrumentationName is the name of the library providing
	// instrumentation.
	InstrumentationName string
	// InstrumentationVersion is the version of the library providing
	// instrumentation.
	InstrumentationVersion string
}

// InstrumentOption is an interface for applying metric instrument options.
type InstrumentOption interface {
	// ApplyMeter is used to set a InstrumentOption value of a
	// InstrumentConfig.
	ApplyInstrument(*InstrumentConfig)
}

// NewInstrumentConfig creates a new InstrumentConfig
// and applies all the given options.
func NewInstrumentConfig(opts ...InstrumentOption) InstrumentConfig {
	var config InstrumentConfig
	for _, o := range opts {
		o.ApplyInstrument(&config)
	}
	return config
}

// WithDescription applies provided description.
func WithDescription(desc string) InstrumentOption {
	return descriptionOption(desc)
}

type descriptionOption string

func (d descriptionOption) ApplyInstrument(config *InstrumentConfig) {
	config.Description = string(d)
}

// WithUnit applies provided unit.
func WithUnit(unit unit.Unit) InstrumentOption {
	return unitOption(unit)
}

type unitOption unit.Unit

func (u unitOption) ApplyInstrument(config *InstrumentConfig) {
	config.Unit = unit.Unit(u)
}

// WithInstrumentationName sets the instrumentation name.
func WithInstrumentationName(name string) InstrumentOption {
	return instrumentationNameOption(name)
}

type instrumentationNameOption string

func (i instrumentationNameOption) ApplyInstrument(config *InstrumentConfig) {
	config.InstrumentationName = string(i)
}

// MeterConfig contains options for Meters.
type MeterConfig struct {
	// InstrumentationVersion is the version of the library providing
	// instrumentation.
	InstrumentationVersion string
}

// MeterOption is an interface for applying Meter options.
type MeterOption interface {
	// ApplyMeter is used to set a MeterOption value of a MeterConfig.
	ApplyMeter(*MeterConfig)
}

// NewMeterConfig creates a new MeterConfig and applies
// all the given options.
func NewMeterConfig(opts ...MeterOption) MeterConfig {
	var config MeterConfig
	for _, o := range opts {
		o.ApplyMeter(&config)
	}
	return config
}

// InstrumentationOption is an interface for applying instrumentation specific
// options.
type InstrumentationOption interface {
	InstrumentOption
	MeterOption
}

// WithInstrumentationVersion sets the instrumentation version.
func WithInstrumentationVersion(version string) InstrumentationOption {
	return instrumentationVersionOption(version)
}

type instrumentationVersionOption string

func (i instrumentationVersionOption) ApplyMeter(config *MeterConfig) {
	config.InstrumentationVersion = string(i)
}

func (i instrumentationVersionOption) ApplyInstrument(config *InstrumentConfig) {
	config.InstrumentationVersion = string(i)
}
