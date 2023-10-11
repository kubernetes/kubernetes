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

package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"errors"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

type (
	tracerProviderHolder struct {
		tp trace.TracerProvider
	}

	propagatorsHolder struct {
		tm propagation.TextMapPropagator
	}

	meterProviderHolder struct {
		mp metric.MeterProvider
	}
)

var (
	globalTracer        = defaultTracerValue()
	globalPropagators   = defaultPropagatorsValue()
	globalMeterProvider = defaultMeterProvider()

	delegateTraceOnce             sync.Once
	delegateTextMapPropagatorOnce sync.Once
	delegateMeterOnce             sync.Once
)

// TracerProvider is the internal implementation for global.TracerProvider.
func TracerProvider() trace.TracerProvider {
	return globalTracer.Load().(tracerProviderHolder).tp
}

// SetTracerProvider is the internal implementation for global.SetTracerProvider.
func SetTracerProvider(tp trace.TracerProvider) {
	current := TracerProvider()

	if _, cOk := current.(*tracerProvider); cOk {
		if _, tpOk := tp.(*tracerProvider); tpOk && current == tp {
			// Do not assign the default delegating TracerProvider to delegate
			// to itself.
			Error(
				errors.New("no delegate configured in tracer provider"),
				"Setting tracer provider to it's current value. No delegate will be configured",
			)
			return
		}
	}

	delegateTraceOnce.Do(func() {
		if def, ok := current.(*tracerProvider); ok {
			def.setDelegate(tp)
		}
	})
	globalTracer.Store(tracerProviderHolder{tp: tp})
}

// TextMapPropagator is the internal implementation for global.TextMapPropagator.
func TextMapPropagator() propagation.TextMapPropagator {
	return globalPropagators.Load().(propagatorsHolder).tm
}

// SetTextMapPropagator is the internal implementation for global.SetTextMapPropagator.
func SetTextMapPropagator(p propagation.TextMapPropagator) {
	current := TextMapPropagator()

	if _, cOk := current.(*textMapPropagator); cOk {
		if _, pOk := p.(*textMapPropagator); pOk && current == p {
			// Do not assign the default delegating TextMapPropagator to
			// delegate to itself.
			Error(
				errors.New("no delegate configured in text map propagator"),
				"Setting text map propagator to it's current value. No delegate will be configured",
			)
			return
		}
	}

	// For the textMapPropagator already returned by TextMapPropagator
	// delegate to p.
	delegateTextMapPropagatorOnce.Do(func() {
		if def, ok := current.(*textMapPropagator); ok {
			def.SetDelegate(p)
		}
	})
	// Return p when subsequent calls to TextMapPropagator are made.
	globalPropagators.Store(propagatorsHolder{tm: p})
}

// MeterProvider is the internal implementation for global.MeterProvider.
func MeterProvider() metric.MeterProvider {
	return globalMeterProvider.Load().(meterProviderHolder).mp
}

// SetMeterProvider is the internal implementation for global.SetMeterProvider.
func SetMeterProvider(mp metric.MeterProvider) {
	current := MeterProvider()
	if _, cOk := current.(*meterProvider); cOk {
		if _, mpOk := mp.(*meterProvider); mpOk && current == mp {
			// Do not assign the default delegating MeterProvider to delegate
			// to itself.
			Error(
				errors.New("no delegate configured in meter provider"),
				"Setting meter provider to it's current value. No delegate will be configured",
			)
			return
		}
	}

	delegateMeterOnce.Do(func() {
		if def, ok := current.(*meterProvider); ok {
			def.setDelegate(mp)
		}
	})
	globalMeterProvider.Store(meterProviderHolder{mp: mp})
}

func defaultTracerValue() *atomic.Value {
	v := &atomic.Value{}
	v.Store(tracerProviderHolder{tp: &tracerProvider{}})
	return v
}

func defaultPropagatorsValue() *atomic.Value {
	v := &atomic.Value{}
	v.Store(propagatorsHolder{tm: newTextMapPropagator()})
	return v
}

func defaultMeterProvider() *atomic.Value {
	v := &atomic.Value{}
	v.Store(meterProviderHolder{mp: &meterProvider{}})
	return v
}
