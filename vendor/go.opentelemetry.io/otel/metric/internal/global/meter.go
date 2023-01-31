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
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/instrument"
	"go.opentelemetry.io/otel/metric/instrument/asyncfloat64"
	"go.opentelemetry.io/otel/metric/instrument/asyncint64"
	"go.opentelemetry.io/otel/metric/instrument/syncfloat64"
	"go.opentelemetry.io/otel/metric/instrument/syncint64"
)

// meterProvider is a placeholder for a configured SDK MeterProvider.
//
// All MeterProvider functionality is forwarded to a delegate once
// configured.
type meterProvider struct {
	mtx    sync.Mutex
	meters map[il]*meter

	delegate metric.MeterProvider
}

type il struct {
	name    string
	version string
}

// setDelegate configures p to delegate all MeterProvider functionality to
// provider.
//
// All Meters provided prior to this function call are switched out to be
// Meters provided by provider. All instruments and callbacks are recreated and
// delegated.
//
// It is guaranteed by the caller that this happens only once.
func (p *meterProvider) setDelegate(provider metric.MeterProvider) {
	p.mtx.Lock()
	defer p.mtx.Unlock()

	p.delegate = provider

	if len(p.meters) == 0 {
		return
	}

	for _, meter := range p.meters {
		meter.setDelegate(provider)
	}

	p.meters = nil
}

// Meter implements MeterProvider.
func (p *meterProvider) Meter(name string, opts ...metric.MeterOption) metric.Meter {
	p.mtx.Lock()
	defer p.mtx.Unlock()

	if p.delegate != nil {
		return p.delegate.Meter(name, opts...)
	}

	// At this moment it is guaranteed that no sdk is installed, save the meter in the meters map.

	c := metric.NewMeterConfig(opts...)
	key := il{
		name:    name,
		version: c.InstrumentationVersion(),
	}

	if p.meters == nil {
		p.meters = make(map[il]*meter)
	}

	if val, ok := p.meters[key]; ok {
		return val
	}

	t := &meter{name: name, opts: opts}
	p.meters[key] = t
	return t
}

// meter is a placeholder for a metric.Meter.
//
// All Meter functionality is forwarded to a delegate once configured.
// Otherwise, all functionality is forwarded to a NoopMeter.
type meter struct {
	name string
	opts []metric.MeterOption

	mtx         sync.Mutex
	instruments []delegatedInstrument
	callbacks   []delegatedCallback

	delegate atomic.Value // metric.Meter
}

type delegatedInstrument interface {
	setDelegate(metric.Meter)
}

// setDelegate configures m to delegate all Meter functionality to Meters
// created by provider.
//
// All subsequent calls to the Meter methods will be passed to the delegate.
//
// It is guaranteed by the caller that this happens only once.
func (m *meter) setDelegate(provider metric.MeterProvider) {
	meter := provider.Meter(m.name, m.opts...)
	m.delegate.Store(meter)

	m.mtx.Lock()
	defer m.mtx.Unlock()

	for _, inst := range m.instruments {
		inst.setDelegate(meter)
	}

	for _, callback := range m.callbacks {
		callback.setDelegate(meter)
	}

	m.instruments = nil
	m.callbacks = nil
}

// AsyncInt64 is the namespace for the Asynchronous Integer instruments.
//
// To Observe data with instruments it must be registered in a callback.
func (m *meter) AsyncInt64() asyncint64.InstrumentProvider {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.AsyncInt64()
	}
	return (*aiInstProvider)(m)
}

// AsyncFloat64 is the namespace for the Asynchronous Float instruments.
//
// To Observe data with instruments it must be registered in a callback.
func (m *meter) AsyncFloat64() asyncfloat64.InstrumentProvider {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.AsyncFloat64()
	}
	return (*afInstProvider)(m)
}

// RegisterCallback captures the function that will be called during Collect.
//
// It is only valid to call Observe within the scope of the passed function,
// and only on the instruments that were registered with this call.
func (m *meter) RegisterCallback(insts []instrument.Asynchronous, function func(context.Context)) error {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		insts = unwrapInstruments(insts)
		return del.RegisterCallback(insts, function)
	}

	m.mtx.Lock()
	defer m.mtx.Unlock()
	m.callbacks = append(m.callbacks, delegatedCallback{
		instruments: insts,
		function:    function,
	})

	return nil
}

type wrapped interface {
	unwrap() instrument.Asynchronous
}

func unwrapInstruments(instruments []instrument.Asynchronous) []instrument.Asynchronous {
	out := make([]instrument.Asynchronous, 0, len(instruments))

	for _, inst := range instruments {
		if in, ok := inst.(wrapped); ok {
			out = append(out, in.unwrap())
		} else {
			out = append(out, inst)
		}
	}

	return out
}

// SyncInt64 is the namespace for the Synchronous Integer instruments.
func (m *meter) SyncInt64() syncint64.InstrumentProvider {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.SyncInt64()
	}
	return (*siInstProvider)(m)
}

// SyncFloat64 is the namespace for the Synchronous Float instruments.
func (m *meter) SyncFloat64() syncfloat64.InstrumentProvider {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.SyncFloat64()
	}
	return (*sfInstProvider)(m)
}

type delegatedCallback struct {
	instruments []instrument.Asynchronous
	function    func(context.Context)
}

func (c *delegatedCallback) setDelegate(m metric.Meter) {
	insts := unwrapInstruments(c.instruments)
	err := m.RegisterCallback(insts, c.function)
	if err != nil {
		otel.Handle(err)
	}
}

type afInstProvider meter

// Counter creates an instrument for recording increasing values.
func (ip *afInstProvider) Counter(name string, opts ...instrument.Option) (asyncfloat64.Counter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &afCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// UpDownCounter creates an instrument for recording changes of a value.
func (ip *afInstProvider) UpDownCounter(name string, opts ...instrument.Option) (asyncfloat64.UpDownCounter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &afUpDownCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// Gauge creates an instrument for recording the current value.
func (ip *afInstProvider) Gauge(name string, opts ...instrument.Option) (asyncfloat64.Gauge, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &afGauge{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

type aiInstProvider meter

// Counter creates an instrument for recording increasing values.
func (ip *aiInstProvider) Counter(name string, opts ...instrument.Option) (asyncint64.Counter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &aiCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// UpDownCounter creates an instrument for recording changes of a value.
func (ip *aiInstProvider) UpDownCounter(name string, opts ...instrument.Option) (asyncint64.UpDownCounter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &aiUpDownCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// Gauge creates an instrument for recording the current value.
func (ip *aiInstProvider) Gauge(name string, opts ...instrument.Option) (asyncint64.Gauge, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &aiGauge{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

type sfInstProvider meter

// Counter creates an instrument for recording increasing values.
func (ip *sfInstProvider) Counter(name string, opts ...instrument.Option) (syncfloat64.Counter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &sfCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// UpDownCounter creates an instrument for recording changes of a value.
func (ip *sfInstProvider) UpDownCounter(name string, opts ...instrument.Option) (syncfloat64.UpDownCounter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &sfUpDownCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// Histogram creates an instrument for recording a distribution of values.
func (ip *sfInstProvider) Histogram(name string, opts ...instrument.Option) (syncfloat64.Histogram, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &sfHistogram{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

type siInstProvider meter

// Counter creates an instrument for recording increasing values.
func (ip *siInstProvider) Counter(name string, opts ...instrument.Option) (syncint64.Counter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &siCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// UpDownCounter creates an instrument for recording changes of a value.
func (ip *siInstProvider) UpDownCounter(name string, opts ...instrument.Option) (syncint64.UpDownCounter, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &siUpDownCounter{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}

// Histogram creates an instrument for recording a distribution of values.
func (ip *siInstProvider) Histogram(name string, opts ...instrument.Option) (syncint64.Histogram, error) {
	ip.mtx.Lock()
	defer ip.mtx.Unlock()
	ctr := &siHistogram{name: name, opts: opts}
	ip.instruments = append(ip.instruments, ctr)
	return ctr, nil
}
