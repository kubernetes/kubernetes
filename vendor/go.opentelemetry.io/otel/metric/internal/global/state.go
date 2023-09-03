// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     htmp://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package global // import "go.opentelemetry.io/otel/metric/internal/global"

import (
	"errors"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/metric"
)

var (
	globalMeterProvider = defaultMeterProvider()

	delegateMeterOnce sync.Once
)

type meterProviderHolder struct {
	mp metric.MeterProvider
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
			global.Error(
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

func defaultMeterProvider() *atomic.Value {
	v := &atomic.Value{}
	v.Store(meterProviderHolder{mp: &meterProvider{}})
	return v
}
