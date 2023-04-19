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
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric/instrument"
)

// NewNoopMeterProvider creates a MeterProvider that does not record any metrics.
func NewNoopMeterProvider() MeterProvider {
	return noopMeterProvider{}
}

type noopMeterProvider struct{}

func (noopMeterProvider) Meter(string, ...MeterOption) Meter {
	return noopMeter{}
}

// NewNoopMeter creates a Meter that does not record any metrics.
func NewNoopMeter() Meter {
	return noopMeter{}
}

type noopMeter struct{}

func (noopMeter) Int64Counter(string, ...instrument.Int64Option) (instrument.Int64Counter, error) {
	return nonrecordingSyncInt64Instrument{}, nil
}

func (noopMeter) Int64UpDownCounter(string, ...instrument.Int64Option) (instrument.Int64UpDownCounter, error) {
	return nonrecordingSyncInt64Instrument{}, nil
}

func (noopMeter) Int64Histogram(string, ...instrument.Int64Option) (instrument.Int64Histogram, error) {
	return nonrecordingSyncInt64Instrument{}, nil
}

func (noopMeter) Int64ObservableCounter(string, ...instrument.Int64ObserverOption) (instrument.Int64ObservableCounter, error) {
	return nonrecordingAsyncInt64Instrument{}, nil
}

func (noopMeter) Int64ObservableUpDownCounter(string, ...instrument.Int64ObserverOption) (instrument.Int64ObservableUpDownCounter, error) {
	return nonrecordingAsyncInt64Instrument{}, nil
}

func (noopMeter) Int64ObservableGauge(string, ...instrument.Int64ObserverOption) (instrument.Int64ObservableGauge, error) {
	return nonrecordingAsyncInt64Instrument{}, nil
}

func (noopMeter) Float64Counter(string, ...instrument.Float64Option) (instrument.Float64Counter, error) {
	return nonrecordingSyncFloat64Instrument{}, nil
}

func (noopMeter) Float64UpDownCounter(string, ...instrument.Float64Option) (instrument.Float64UpDownCounter, error) {
	return nonrecordingSyncFloat64Instrument{}, nil
}

func (noopMeter) Float64Histogram(string, ...instrument.Float64Option) (instrument.Float64Histogram, error) {
	return nonrecordingSyncFloat64Instrument{}, nil
}

func (noopMeter) Float64ObservableCounter(string, ...instrument.Float64ObserverOption) (instrument.Float64ObservableCounter, error) {
	return nonrecordingAsyncFloat64Instrument{}, nil
}

func (noopMeter) Float64ObservableUpDownCounter(string, ...instrument.Float64ObserverOption) (instrument.Float64ObservableUpDownCounter, error) {
	return nonrecordingAsyncFloat64Instrument{}, nil
}

func (noopMeter) Float64ObservableGauge(string, ...instrument.Float64ObserverOption) (instrument.Float64ObservableGauge, error) {
	return nonrecordingAsyncFloat64Instrument{}, nil
}

// RegisterCallback creates a register callback that does not record any metrics.
func (noopMeter) RegisterCallback(Callback, ...instrument.Asynchronous) (Registration, error) {
	return noopReg{}, nil
}

type noopReg struct{}

func (noopReg) Unregister() error { return nil }

type nonrecordingAsyncFloat64Instrument struct {
	instrument.Float64Observable
}

var (
	_ instrument.Float64ObservableCounter       = nonrecordingAsyncFloat64Instrument{}
	_ instrument.Float64ObservableUpDownCounter = nonrecordingAsyncFloat64Instrument{}
	_ instrument.Float64ObservableGauge         = nonrecordingAsyncFloat64Instrument{}
)

type nonrecordingAsyncInt64Instrument struct {
	instrument.Int64Observable
}

var (
	_ instrument.Int64ObservableCounter       = nonrecordingAsyncInt64Instrument{}
	_ instrument.Int64ObservableUpDownCounter = nonrecordingAsyncInt64Instrument{}
	_ instrument.Int64ObservableGauge         = nonrecordingAsyncInt64Instrument{}
)

type nonrecordingSyncFloat64Instrument struct {
	instrument.Synchronous
}

var (
	_ instrument.Float64Counter       = nonrecordingSyncFloat64Instrument{}
	_ instrument.Float64UpDownCounter = nonrecordingSyncFloat64Instrument{}
	_ instrument.Float64Histogram     = nonrecordingSyncFloat64Instrument{}
)

func (nonrecordingSyncFloat64Instrument) Add(context.Context, float64, ...attribute.KeyValue)    {}
func (nonrecordingSyncFloat64Instrument) Record(context.Context, float64, ...attribute.KeyValue) {}

type nonrecordingSyncInt64Instrument struct {
	instrument.Synchronous
}

var (
	_ instrument.Int64Counter       = nonrecordingSyncInt64Instrument{}
	_ instrument.Int64UpDownCounter = nonrecordingSyncInt64Instrument{}
	_ instrument.Int64Histogram     = nonrecordingSyncInt64Instrument{}
)

func (nonrecordingSyncInt64Instrument) Add(context.Context, int64, ...attribute.KeyValue)    {}
func (nonrecordingSyncInt64Instrument) Record(context.Context, int64, ...attribute.KeyValue) {}
