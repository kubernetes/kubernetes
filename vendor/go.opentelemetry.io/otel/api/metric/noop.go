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

import (
	"context"

	"go.opentelemetry.io/otel/api/core"
)

type NoopProvider struct{}
type NoopMeter struct{}

type noopInstrument struct{}
type noopBoundInstrument struct{}
type NoopSync struct{ noopInstrument }
type NoopAsync struct{ noopInstrument }

var _ Provider = NoopProvider{}
var _ Meter = NoopMeter{}
var _ SyncImpl = NoopSync{}
var _ BoundSyncImpl = noopBoundInstrument{}
var _ AsyncImpl = NoopAsync{}

func (NoopProvider) Meter(name string) Meter {
	return NoopMeter{}
}

func (noopInstrument) Implementation() interface{} {
	return nil
}

func (noopInstrument) Descriptor() Descriptor {
	return Descriptor{}
}

func (noopBoundInstrument) RecordOne(context.Context, core.Number) {
}

func (noopBoundInstrument) Unbind() {
}

func (NoopSync) Bind([]core.KeyValue) BoundSyncImpl {
	return noopBoundInstrument{}
}

func (NoopSync) RecordOne(context.Context, core.Number, []core.KeyValue) {
}

func (NoopMeter) RecordBatch(context.Context, []core.KeyValue, ...Measurement) {
}

func (NoopMeter) NewInt64Counter(string, ...Option) (Int64Counter, error) {
	return Int64Counter{syncInstrument{NoopSync{}}}, nil
}

func (NoopMeter) NewFloat64Counter(string, ...Option) (Float64Counter, error) {
	return Float64Counter{syncInstrument{NoopSync{}}}, nil
}

func (NoopMeter) NewInt64Measure(string, ...Option) (Int64Measure, error) {
	return Int64Measure{syncInstrument{NoopSync{}}}, nil
}

func (NoopMeter) NewFloat64Measure(string, ...Option) (Float64Measure, error) {
	return Float64Measure{syncInstrument{NoopSync{}}}, nil
}

func (NoopMeter) RegisterInt64Observer(string, Int64ObserverCallback, ...Option) (Int64Observer, error) {
	return Int64Observer{asyncInstrument{NoopAsync{}}}, nil
}

func (NoopMeter) RegisterFloat64Observer(string, Float64ObserverCallback, ...Option) (Float64Observer, error) {
	return Float64Observer{asyncInstrument{NoopAsync{}}}, nil
}
