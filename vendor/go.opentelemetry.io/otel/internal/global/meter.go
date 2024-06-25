// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package global // import "go.opentelemetry.io/otel/internal/global"

import (
	"container/list"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/embedded"
)

// meterProvider is a placeholder for a configured SDK MeterProvider.
//
// All MeterProvider functionality is forwarded to a delegate once
// configured.
type meterProvider struct {
	embedded.MeterProvider

	mtx    sync.Mutex
	meters map[il]*meter

	delegate metric.MeterProvider
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
	embedded.Meter

	name string
	opts []metric.MeterOption

	mtx         sync.Mutex
	instruments []delegatedInstrument

	registry list.List

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

	var n *list.Element
	for e := m.registry.Front(); e != nil; e = n {
		r := e.Value.(*registration)
		r.setDelegate(meter)
		n = e.Next()
		m.registry.Remove(e)
	}

	m.instruments = nil
	m.registry.Init()
}

func (m *meter) Int64Counter(name string, options ...metric.Int64CounterOption) (metric.Int64Counter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64Counter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &siCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64UpDownCounter(name string, options ...metric.Int64UpDownCounterOption) (metric.Int64UpDownCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64UpDownCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &siUpDownCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64Histogram(name string, options ...metric.Int64HistogramOption) (metric.Int64Histogram, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64Histogram(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &siHistogram{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64Gauge(name string, options ...metric.Int64GaugeOption) (metric.Int64Gauge, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64Gauge(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &siGauge{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64ObservableCounter(name string, options ...metric.Int64ObservableCounterOption) (metric.Int64ObservableCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64ObservableCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &aiCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64ObservableUpDownCounter(name string, options ...metric.Int64ObservableUpDownCounterOption) (metric.Int64ObservableUpDownCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64ObservableUpDownCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &aiUpDownCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Int64ObservableGauge(name string, options ...metric.Int64ObservableGaugeOption) (metric.Int64ObservableGauge, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Int64ObservableGauge(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &aiGauge{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64Counter(name string, options ...metric.Float64CounterOption) (metric.Float64Counter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64Counter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &sfCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64UpDownCounter(name string, options ...metric.Float64UpDownCounterOption) (metric.Float64UpDownCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64UpDownCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &sfUpDownCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64Histogram(name string, options ...metric.Float64HistogramOption) (metric.Float64Histogram, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64Histogram(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &sfHistogram{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64Gauge(name string, options ...metric.Float64GaugeOption) (metric.Float64Gauge, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64Gauge(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &sfGauge{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64ObservableCounter(name string, options ...metric.Float64ObservableCounterOption) (metric.Float64ObservableCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64ObservableCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &afCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64ObservableUpDownCounter(name string, options ...metric.Float64ObservableUpDownCounterOption) (metric.Float64ObservableUpDownCounter, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64ObservableUpDownCounter(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &afUpDownCounter{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

func (m *meter) Float64ObservableGauge(name string, options ...metric.Float64ObservableGaugeOption) (metric.Float64ObservableGauge, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		return del.Float64ObservableGauge(name, options...)
	}
	m.mtx.Lock()
	defer m.mtx.Unlock()
	i := &afGauge{name: name, opts: options}
	m.instruments = append(m.instruments, i)
	return i, nil
}

// RegisterCallback captures the function that will be called during Collect.
func (m *meter) RegisterCallback(f metric.Callback, insts ...metric.Observable) (metric.Registration, error) {
	if del, ok := m.delegate.Load().(metric.Meter); ok {
		insts = unwrapInstruments(insts)
		return del.RegisterCallback(f, insts...)
	}

	m.mtx.Lock()
	defer m.mtx.Unlock()

	reg := &registration{instruments: insts, function: f}
	e := m.registry.PushBack(reg)
	reg.unreg = func() error {
		m.mtx.Lock()
		_ = m.registry.Remove(e)
		m.mtx.Unlock()
		return nil
	}
	return reg, nil
}

type wrapped interface {
	unwrap() metric.Observable
}

func unwrapInstruments(instruments []metric.Observable) []metric.Observable {
	out := make([]metric.Observable, 0, len(instruments))

	for _, inst := range instruments {
		if in, ok := inst.(wrapped); ok {
			out = append(out, in.unwrap())
		} else {
			out = append(out, inst)
		}
	}

	return out
}

type registration struct {
	embedded.Registration

	instruments []metric.Observable
	function    metric.Callback

	unreg   func() error
	unregMu sync.Mutex
}

func (c *registration) setDelegate(m metric.Meter) {
	insts := unwrapInstruments(c.instruments)

	c.unregMu.Lock()
	defer c.unregMu.Unlock()

	if c.unreg == nil {
		// Unregister already called.
		return
	}

	reg, err := m.RegisterCallback(c.function, insts...)
	if err != nil {
		GetErrorHandler().Handle(err)
	}

	c.unreg = reg.Unregister
}

func (c *registration) Unregister() error {
	c.unregMu.Lock()
	defer c.unregMu.Unlock()
	if c.unreg == nil {
		// Unregister already called.
		return nil
	}

	var err error
	err, c.unreg = c.unreg(), nil
	return err
}
