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

package global // import "go.opentelemetry.io/otel/metric/global"

import (
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/internal/global"
)

// Meter returns a Meter from the global MeterProvider. The
// instrumentationName must be the name of the library providing
// instrumentation. This name may be the same as the instrumented code only if
// that code provides built-in instrumentation. If the instrumentationName is
// empty, then a implementation defined default name will be used instead.
//
// This is short for MeterProvider().Meter(name).
func Meter(instrumentationName string, opts ...metric.MeterOption) metric.Meter {
	return MeterProvider().Meter(instrumentationName, opts...)
}

// MeterProvider returns the registered global meter provider.
// If none is registered then a No-op MeterProvider is returned.
func MeterProvider() metric.MeterProvider {
	return global.MeterProvider()
}

// SetMeterProvider registers `mp` as the global meter provider.
func SetMeterProvider(mp metric.MeterProvider) {
	global.SetMeterProvider(mp)
}
