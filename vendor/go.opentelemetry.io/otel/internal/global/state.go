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

package global

import (
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

	meterProviderHolder struct {
		mp metric.MeterProvider
	}

	propagatorsHolder struct {
		tm propagation.TextMapPropagator
	}
)

var (
	globalTracer      = defaultTracerValue()
	globalMeter       = defaultMeterValue()
	globalPropagators = defaultPropagatorsValue()

	delegateMeterOnce             sync.Once
	delegateTraceOnce             sync.Once
	delegateTextMapPropagatorOnce sync.Once
)

// TracerProvider is the internal implementation for global.TracerProvider.
func TracerProvider() trace.TracerProvider {
	return globalTracer.Load().(tracerProviderHolder).tp
}

// SetTracerProvider is the internal implementation for global.SetTracerProvider.
func SetTracerProvider(tp trace.TracerProvider) {
	delegateTraceOnce.Do(func() {
		current := TracerProvider()
		if current == tp {
			// Setting the provider to the prior default is nonsense, panic.
			// Panic is acceptable because we are likely still early in the
			// process lifetime.
			panic("invalid TracerProvider, the global instance cannot be reinstalled")
		} else if def, ok := current.(*tracerProvider); ok {
			def.setDelegate(tp)
		}

	})
	globalTracer.Store(tracerProviderHolder{tp: tp})
}

// MeterProvider is the internal implementation for global.MeterProvider.
func MeterProvider() metric.MeterProvider {
	return globalMeter.Load().(meterProviderHolder).mp
}

// SetMeterProvider is the internal implementation for global.SetMeterProvider.
func SetMeterProvider(mp metric.MeterProvider) {
	delegateMeterOnce.Do(func() {
		current := MeterProvider()

		if current == mp {
			// Setting the provider to the prior default is nonsense, panic.
			// Panic is acceptable because we are likely still early in the
			// process lifetime.
			panic("invalid MeterProvider, the global instance cannot be reinstalled")
		} else if def, ok := current.(*meterProvider); ok {
			def.setDelegate(mp)
		}
	})
	globalMeter.Store(meterProviderHolder{mp: mp})
}

// TextMapPropagator is the internal implementation for global.TextMapPropagator.
func TextMapPropagator() propagation.TextMapPropagator {
	return globalPropagators.Load().(propagatorsHolder).tm
}

// SetTextMapPropagator is the internal implementation for global.SetTextMapPropagator.
func SetTextMapPropagator(p propagation.TextMapPropagator) {
	// For the textMapPropagator already returned by TextMapPropagator
	// delegate to p.
	delegateTextMapPropagatorOnce.Do(func() {
		if current := TextMapPropagator(); current == p {
			// Setting the provider to the prior default is nonsense, panic.
			// Panic is acceptable because we are likely still early in the
			// process lifetime.
			panic("invalid TextMapPropagator, the global instance cannot be reinstalled")
		} else if def, ok := current.(*textMapPropagator); ok {
			def.SetDelegate(p)
		}
	})
	// Return p when subsequent calls to TextMapPropagator are made.
	globalPropagators.Store(propagatorsHolder{tm: p})
}

func defaultTracerValue() *atomic.Value {
	v := &atomic.Value{}
	v.Store(tracerProviderHolder{tp: &tracerProvider{}})
	return v
}

func defaultMeterValue() *atomic.Value {
	v := &atomic.Value{}
	v.Store(meterProviderHolder{mp: newMeterProvider()})
	return v
}

func defaultPropagatorsValue() *atomic.Value {
	v := &atomic.Value{}
	v.Store(propagatorsHolder{tm: newTextMapPropagator()})
	return v
}

// ResetForTest restores the initial global state, for testing purposes.
func ResetForTest() {
	globalTracer = defaultTracerValue()
	globalMeter = defaultMeterValue()
	globalPropagators = defaultPropagatorsValue()
	delegateMeterOnce = sync.Once{}
	delegateTraceOnce = sync.Once{}
	delegateTextMapPropagatorOnce = sync.Once{}
}
