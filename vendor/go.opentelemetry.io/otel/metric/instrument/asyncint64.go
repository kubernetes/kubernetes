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

// Int64Observable describes a set of instruments used asynchronously to record
// int64 measurements once per collection cycle. Observations of these
// instruments are only made within a callback.
//
// Warning: methods may be added to this interface in minor releases.
type Int64Observable interface {
	Asynchronous

	int64Observable()
}

// Int64ObservableCounter is an instrument used to asynchronously record
// increasing int64 measurements once per collection cycle. Observations are
// only made within a callback for this instrument. The value observed is
// assumed the to be the cumulative sum of the count.
//
// Warning: methods may be added to this interface in minor releases.
type Int64ObservableCounter interface{ Int64Observable }

// Int64ObservableUpDownCounter is an instrument used to asynchronously record
// int64 measurements once per collection cycle. Observations are only made
// within a callback for this instrument. The value observed is assumed the to
// be the cumulative sum of the count.
//
// Warning: methods may be added to this interface in minor releases.
type Int64ObservableUpDownCounter interface{ Int64Observable }

// Int64ObservableGauge is an instrument used to asynchronously record
// instantaneous int64 measurements once per collection cycle. Observations are
// only made within a callback for this instrument.
//
// Warning: methods may be added to this interface in minor releases.
type Int64ObservableGauge interface{ Int64Observable }

// Int64Observer is a recorder of int64 measurements.
//
// Warning: methods may be added to this interface in minor releases.
type Int64Observer interface {
	Observe(value int64, attributes ...attribute.KeyValue)
}

// Int64Callback is a function registered with a Meter that makes
// observations for a Int64Observerable instrument it is registered with.
// Calls to the Int64Observer record measurement values for the
// Int64Observable.
//
// The function needs to complete in a finite amount of time and the deadline
// of the passed context is expected to be honored.
//
// The function needs to make unique observations across all registered
// Int64Callback. Meaning, it should not report measurements with the same
// attributes as another Int64Callbacks also registered for the same
// instrument.
//
// The function needs to be concurrent safe.
type Int64Callback func(context.Context, Int64Observer) error

// Int64ObserverConfig contains options for Asynchronous instruments that
// observe int64 values.
type Int64ObserverConfig struct {
	description string
	unit        string
	callbacks   []Int64Callback
}

// NewInt64ObserverConfig returns a new Int64ObserverConfig with all opts
// applied.
func NewInt64ObserverConfig(opts ...Int64ObserverOption) Int64ObserverConfig {
	var config Int64ObserverConfig
	for _, o := range opts {
		config = o.applyInt64Observer(config)
	}
	return config
}

// Description returns the Config description.
func (c Int64ObserverConfig) Description() string {
	return c.description
}

// Unit returns the Config unit.
func (c Int64ObserverConfig) Unit() string {
	return c.unit
}

// Callbacks returns the Config callbacks.
func (c Int64ObserverConfig) Callbacks() []Int64Callback {
	return c.callbacks
}

// Int64ObserverOption applies options to int64 Observer instruments.
type Int64ObserverOption interface {
	applyInt64Observer(Int64ObserverConfig) Int64ObserverConfig
}

type int64ObserverOptionFunc func(Int64ObserverConfig) Int64ObserverConfig

func (fn int64ObserverOptionFunc) applyInt64Observer(cfg Int64ObserverConfig) Int64ObserverConfig {
	return fn(cfg)
}

// WithInt64Callback adds callback to be called for an instrument.
func WithInt64Callback(callback Int64Callback) Int64ObserverOption {
	return int64ObserverOptionFunc(func(cfg Int64ObserverConfig) Int64ObserverConfig {
		cfg.callbacks = append(cfg.callbacks, callback)
		return cfg
	})
}
