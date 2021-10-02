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
	"go.opentelemetry.io/otel/metric/number"
)

type NoopMeterProvider struct{}

type noopInstrument struct{}
type noopBoundInstrument struct{}
type NoopSync struct{ noopInstrument }
type NoopAsync struct{ noopInstrument }

var _ MeterProvider = NoopMeterProvider{}
var _ SyncImpl = NoopSync{}
var _ BoundSyncImpl = noopBoundInstrument{}
var _ AsyncImpl = NoopAsync{}

func (NoopMeterProvider) Meter(_ string, _ ...MeterOption) Meter {
	return Meter{}
}

func (noopInstrument) Implementation() interface{} {
	return nil
}

func (noopInstrument) Descriptor() Descriptor {
	return Descriptor{}
}

func (noopBoundInstrument) RecordOne(context.Context, number.Number) {
}

func (noopBoundInstrument) Unbind() {
}

func (NoopSync) Bind([]attribute.KeyValue) BoundSyncImpl {
	return noopBoundInstrument{}
}

func (NoopSync) RecordOne(context.Context, number.Number, []attribute.KeyValue) {
}
