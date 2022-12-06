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

package global // import "go.opentelemetry.io/otel/metric/internal/global"

import (
	"context"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/instrument"
	"go.opentelemetry.io/otel/metric/instrument/asyncfloat64"
	"go.opentelemetry.io/otel/metric/instrument/asyncint64"
	"go.opentelemetry.io/otel/metric/instrument/syncfloat64"
	"go.opentelemetry.io/otel/metric/instrument/syncint64"
)

type afCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncfloat64.Counter

	instrument.Asynchronous
}

func (i *afCounter) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncFloat64().Counter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *afCounter) Observe(ctx context.Context, x float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncfloat64.Counter).Observe(ctx, x, attrs...)
	}
}

func (i *afCounter) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncfloat64.Counter)
	}
	return nil
}

type afUpDownCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncfloat64.UpDownCounter

	instrument.Asynchronous
}

func (i *afUpDownCounter) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncFloat64().UpDownCounter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *afUpDownCounter) Observe(ctx context.Context, x float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncfloat64.UpDownCounter).Observe(ctx, x, attrs...)
	}
}

func (i *afUpDownCounter) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncfloat64.UpDownCounter)
	}
	return nil
}

type afGauge struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncfloat64.Gauge

	instrument.Asynchronous
}

func (i *afGauge) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncFloat64().Gauge(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *afGauge) Observe(ctx context.Context, x float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncfloat64.Gauge).Observe(ctx, x, attrs...)
	}
}

func (i *afGauge) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncfloat64.Gauge)
	}
	return nil
}

type aiCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncint64.Counter

	instrument.Asynchronous
}

func (i *aiCounter) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncInt64().Counter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *aiCounter) Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncint64.Counter).Observe(ctx, x, attrs...)
	}
}

func (i *aiCounter) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncint64.Counter)
	}
	return nil
}

type aiUpDownCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncint64.UpDownCounter

	instrument.Asynchronous
}

func (i *aiUpDownCounter) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncInt64().UpDownCounter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *aiUpDownCounter) Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncint64.UpDownCounter).Observe(ctx, x, attrs...)
	}
}

func (i *aiUpDownCounter) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncint64.UpDownCounter)
	}
	return nil
}

type aiGauge struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //asyncint64.Gauge

	instrument.Asynchronous
}

func (i *aiGauge) setDelegate(m metric.Meter) {
	ctr, err := m.AsyncInt64().Gauge(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *aiGauge) Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(asyncint64.Gauge).Observe(ctx, x, attrs...)
	}
}

func (i *aiGauge) unwrap() instrument.Asynchronous {
	if ctr := i.delegate.Load(); ctr != nil {
		return ctr.(asyncint64.Gauge)
	}
	return nil
}

//Sync Instruments.
type sfCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncfloat64.Counter

	instrument.Synchronous
}

func (i *sfCounter) setDelegate(m metric.Meter) {
	ctr, err := m.SyncFloat64().Counter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *sfCounter) Add(ctx context.Context, incr float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncfloat64.Counter).Add(ctx, incr, attrs...)
	}
}

type sfUpDownCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncfloat64.UpDownCounter

	instrument.Synchronous
}

func (i *sfUpDownCounter) setDelegate(m metric.Meter) {
	ctr, err := m.SyncFloat64().UpDownCounter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *sfUpDownCounter) Add(ctx context.Context, incr float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncfloat64.UpDownCounter).Add(ctx, incr, attrs...)
	}
}

type sfHistogram struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncfloat64.Histogram

	instrument.Synchronous
}

func (i *sfHistogram) setDelegate(m metric.Meter) {
	ctr, err := m.SyncFloat64().Histogram(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *sfHistogram) Record(ctx context.Context, x float64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncfloat64.Histogram).Record(ctx, x, attrs...)
	}
}

type siCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncint64.Counter

	instrument.Synchronous
}

func (i *siCounter) setDelegate(m metric.Meter) {
	ctr, err := m.SyncInt64().Counter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *siCounter) Add(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncint64.Counter).Add(ctx, x, attrs...)
	}
}

type siUpDownCounter struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncint64.UpDownCounter

	instrument.Synchronous
}

func (i *siUpDownCounter) setDelegate(m metric.Meter) {
	ctr, err := m.SyncInt64().UpDownCounter(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *siUpDownCounter) Add(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncint64.UpDownCounter).Add(ctx, x, attrs...)
	}
}

type siHistogram struct {
	name string
	opts []instrument.Option

	delegate atomic.Value //syncint64.Histogram

	instrument.Synchronous
}

func (i *siHistogram) setDelegate(m metric.Meter) {
	ctr, err := m.SyncInt64().Histogram(i.name, i.opts...)
	if err != nil {
		otel.Handle(err)
		return
	}
	i.delegate.Store(ctr)
}

func (i *siHistogram) Record(ctx context.Context, x int64, attrs ...attribute.KeyValue) {
	if ctr := i.delegate.Load(); ctr != nil {
		ctr.(syncint64.Histogram).Record(ctx, x, attrs...)
	}
}
