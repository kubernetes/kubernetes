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

package instrument // import "go.opentelemetry.io/otel/metric/instrument"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
)

// Float64Observable describes a set of instruments used asynchronously to
// record float64 measurements once per collection cycle. Observations of
// these instruments are only made within a callback.
//
// Warning: methods may be added to this interface in minor releases.
type Float64Observable interface {
	Asynchronous

	float64Observable()
}

// Float64ObservableCounter is an instrument used to asynchronously record
// increasing float64 measurements once per collection cycle. Observations are
// only made within a callback for this instrument. The value observed is
// assumed the to be the cumulative sum of the count.
//
// Warning: methods may be added to this interface in minor releases.
type Float64ObservableCounter interface{ Float64Observable }

// Float64ObservableUpDownCounter is an instrument used to asynchronously
// record float64 measurements once per collection cycle. Observations are only
// made within a callback for this instrument. The value observed is assumed
// the to be the cumulative sum of the count.
//
// Warning: methods may be added to this interface in minor releases.
type Float64ObservableUpDownCounter interface{ Float64Observable }

// Float64ObservableGauge is an instrument used to asynchronously record
// instantaneous float64 measurements once per collection cycle. Observations
// are only made within a callback for this instrument.
//
// Warning: methods may be added to this interface in minor releases.
type Float64ObservableGauge interface{ Float64Observable }

// Float64Observer is a recorder of float64 measurements.
//
// Warning: methods may be added to this interface in minor releases.
type Float64Observer interface {
	Observe(value float64, attributes ...attribute.KeyValue)
}

// Float64Callback is a function registered with a Meter that makes
// observations for a Float64Observerable instrument it is registered with.
// Calls to the Float64Observer record measurement values for the
// Float64Observable.
//
// The function needs to complete in a finite amount of time and the deadline
// of the passed context is expected to be honored.
//
// The function needs to make unique observations across all registered
// Float64Callbacks. Meaning, it should not report measurements with the same
// attributes as another Float64Callbacks also registered for the same
// instrument.
//
// The function needs to be concurrent safe.
type Float64Callback func(context.Context, Float64Observer) error

// Float64ObserverConfig contains options for Asynchronous instruments that
// observe float64 values.
type Float64ObserverConfig struct {
	description string
	unit        string
	callbacks   []Float64Callback
}

// NewFloat64ObserverConfig returns a new Float64ObserverConfig with all opts
// applied.
func NewFloat64ObserverConfig(opts ...Float64ObserverOption) Float64ObserverConfig {
	var config Float64ObserverConfig
	for _, o := range opts {
		config = o.applyFloat64Observer(config)
	}
	return config
}

// Description returns the Config description.
func (c Float64ObserverConfig) Description() string {
	return c.description
}

// Unit returns the Config unit.
func (c Float64ObserverConfig) Unit() string {
	return c.unit
}

// Callbacks returns the Config callbacks.
func (c Float64ObserverConfig) Callbacks() []Float64Callback {
	return c.callbacks
}

// Float64ObserverOption applies options to float64 Observer instruments.
type Float64ObserverOption interface {
	applyFloat64Observer(Float64ObserverConfig) Float64ObserverConfig
}

type float64ObserverOptionFunc func(Float64ObserverConfig) Float64ObserverConfig

func (fn float64ObserverOptionFunc) applyFloat64Observer(cfg Float64ObserverConfig) Float64ObserverConfig {
	return fn(cfg)
}

// WithFloat64Callback adds callback to be called for an instrument.
func WithFloat64Callback(callback Float64Callback) Float64ObserverOption {
	return float64ObserverOptionFunc(func(cfg Float64ObserverConfig) Float64ObserverConfig {
		cfg.callbacks = append(cfg.callbacks, callback)
		return cfg
	})
}
