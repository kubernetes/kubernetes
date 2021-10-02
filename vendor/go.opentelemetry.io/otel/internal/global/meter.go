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
	"context"
	"sync"
	"sync/atomic"
	"unsafe"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	"go.opentelemetry.io/otel/metric/registry"
)

// This file contains the forwarding implementation of MeterProvider used as
// the default global instance.  Metric events using instruments provided by
// this implementation are no-ops until the first Meter implementation is set
// as the global provider.
//
// The implementation here uses Mutexes to maintain a list of active Meters in
// the MeterProvider and Instruments in each Meter, under the assumption that
// these interfaces are not performance-critical.
//
// We have the invariant that setDelegate() will be called before a new
// MeterProvider implementation is registered as the global provider.  Mutexes
// in the MeterProvider and Meters ensure that each instrument has a delegate
// before the global provider is set.
//
// Bound instrument operations are implemented by delegating to the
// instrument after it is registered, with a sync.Once initializer to
// protect against races with Release().
//
// Metric uniqueness checking is implemented by calling the exported
// methods of the api/metric/registry package.

type meterKey struct {
	Name, Version string
}

type meterProvider struct {
	delegate metric.MeterProvider

	// lock protects `delegate` and `meters`.
	lock sync.Mutex

	// meters maintains a unique entry for every named Meter
	// that has been registered through the global instance.
	meters map[meterKey]*meterEntry
}

type meterImpl struct {
	delegate unsafe.Pointer // (*metric.MeterImpl)

	lock       sync.Mutex
	syncInsts  []*syncImpl
	asyncInsts []*asyncImpl
}

type meterEntry struct {
	unique metric.MeterImpl
	impl   meterImpl
}

type instrument struct {
	descriptor metric.Descriptor
}

type syncImpl struct {
	delegate unsafe.Pointer // (*metric.SyncImpl)

	instrument
}

type asyncImpl struct {
	delegate unsafe.Pointer // (*metric.AsyncImpl)

	instrument

	runner metric.AsyncRunner
}

// SyncImpler is implemented by all of the sync metric
// instruments.
type SyncImpler interface {
	SyncImpl() metric.SyncImpl
}

// AsyncImpler is implemented by all of the async
// metric instruments.
type AsyncImpler interface {
	AsyncImpl() metric.AsyncImpl
}

type syncHandle struct {
	delegate unsafe.Pointer // (*metric.BoundInstrumentImpl)

	inst   *syncImpl
	labels []attribute.KeyValue

	initialize sync.Once
}

var _ metric.MeterProvider = &meterProvider{}
var _ metric.MeterImpl = &meterImpl{}
var _ metric.InstrumentImpl = &syncImpl{}
var _ metric.BoundSyncImpl = &syncHandle{}
var _ metric.AsyncImpl = &asyncImpl{}

func (inst *instrument) Descriptor() metric.Descriptor {
	return inst.descriptor
}

// MeterProvider interface and delegation

func newMeterProvider() *meterProvider {
	return &meterProvider{
		meters: map[meterKey]*meterEntry{},
	}
}

func (p *meterProvider) setDelegate(provider metric.MeterProvider) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.delegate = provider
	for key, entry := range p.meters {
		entry.impl.setDelegate(key.Name, key.Version, provider)
	}
	p.meters = nil
}

func (p *meterProvider) Meter(instrumentationName string, opts ...metric.MeterOption) metric.Meter {
	p.lock.Lock()
	defer p.lock.Unlock()

	if p.delegate != nil {
		return p.delegate.Meter(instrumentationName, opts...)
	}

	key := meterKey{
		Name:    instrumentationName,
		Version: metric.NewMeterConfig(opts...).InstrumentationVersion,
	}
	entry, ok := p.meters[key]
	if !ok {
		entry = &meterEntry{}
		entry.unique = registry.NewUniqueInstrumentMeterImpl(&entry.impl)
		p.meters[key] = entry

	}
	return metric.WrapMeterImpl(entry.unique, key.Name, metric.WithInstrumentationVersion(key.Version))
}

// Meter interface and delegation

func (m *meterImpl) setDelegate(name, version string, provider metric.MeterProvider) {
	m.lock.Lock()
	defer m.lock.Unlock()

	d := new(metric.MeterImpl)
	*d = provider.Meter(name, metric.WithInstrumentationVersion(version)).MeterImpl()
	m.delegate = unsafe.Pointer(d)

	for _, inst := range m.syncInsts {
		inst.setDelegate(*d)
	}
	m.syncInsts = nil
	for _, obs := range m.asyncInsts {
		obs.setDelegate(*d)
	}
	m.asyncInsts = nil
}

func (m *meterImpl) NewSyncInstrument(desc metric.Descriptor) (metric.SyncImpl, error) {
	m.lock.Lock()
	defer m.lock.Unlock()

	if meterPtr := (*metric.MeterImpl)(atomic.LoadPointer(&m.delegate)); meterPtr != nil {
		return (*meterPtr).NewSyncInstrument(desc)
	}

	inst := &syncImpl{
		instrument: instrument{
			descriptor: desc,
		},
	}
	m.syncInsts = append(m.syncInsts, inst)
	return inst, nil
}

// Synchronous delegation

func (inst *syncImpl) setDelegate(d metric.MeterImpl) {
	implPtr := new(metric.SyncImpl)

	var err error
	*implPtr, err = d.NewSyncInstrument(inst.descriptor)

	if err != nil {
		// TODO: There is no standard way to deliver this error to the user.
		// See https://github.com/open-telemetry/opentelemetry-go/issues/514
		// Note that the default SDK will not generate any errors yet, this is
		// only for added safety.
		panic(err)
	}

	atomic.StorePointer(&inst.delegate, unsafe.Pointer(implPtr))
}

func (inst *syncImpl) Implementation() interface{} {
	if implPtr := (*metric.SyncImpl)(atomic.LoadPointer(&inst.delegate)); implPtr != nil {
		return (*implPtr).Implementation()
	}
	return inst
}

func (inst *syncImpl) Bind(labels []attribute.KeyValue) metric.BoundSyncImpl {
	if implPtr := (*metric.SyncImpl)(atomic.LoadPointer(&inst.delegate)); implPtr != nil {
		return (*implPtr).Bind(labels)
	}
	return &syncHandle{
		inst:   inst,
		labels: labels,
	}
}

func (bound *syncHandle) Unbind() {
	bound.initialize.Do(func() {})

	implPtr := (*metric.BoundSyncImpl)(atomic.LoadPointer(&bound.delegate))

	if implPtr == nil {
		return
	}

	(*implPtr).Unbind()
}

// Async delegation

func (m *meterImpl) NewAsyncInstrument(
	desc metric.Descriptor,
	runner metric.AsyncRunner,
) (metric.AsyncImpl, error) {

	m.lock.Lock()
	defer m.lock.Unlock()

	if meterPtr := (*metric.MeterImpl)(atomic.LoadPointer(&m.delegate)); meterPtr != nil {
		return (*meterPtr).NewAsyncInstrument(desc, runner)
	}

	inst := &asyncImpl{
		instrument: instrument{
			descriptor: desc,
		},
		runner: runner,
	}
	m.asyncInsts = append(m.asyncInsts, inst)
	return inst, nil
}

func (obs *asyncImpl) Implementation() interface{} {
	if implPtr := (*metric.AsyncImpl)(atomic.LoadPointer(&obs.delegate)); implPtr != nil {
		return (*implPtr).Implementation()
	}
	return obs
}

func (obs *asyncImpl) setDelegate(d metric.MeterImpl) {
	implPtr := new(metric.AsyncImpl)

	var err error
	*implPtr, err = d.NewAsyncInstrument(obs.descriptor, obs.runner)

	if err != nil {
		// TODO: There is no standard way to deliver this error to the user.
		// See https://github.com/open-telemetry/opentelemetry-go/issues/514
		// Note that the default SDK will not generate any errors yet, this is
		// only for added safety.
		panic(err)
	}

	atomic.StorePointer(&obs.delegate, unsafe.Pointer(implPtr))
}

// Metric updates

func (m *meterImpl) RecordBatch(ctx context.Context, labels []attribute.KeyValue, measurements ...metric.Measurement) {
	if delegatePtr := (*metric.MeterImpl)(atomic.LoadPointer(&m.delegate)); delegatePtr != nil {
		(*delegatePtr).RecordBatch(ctx, labels, measurements...)
	}
}

func (inst *syncImpl) RecordOne(ctx context.Context, number number.Number, labels []attribute.KeyValue) {
	if instPtr := (*metric.SyncImpl)(atomic.LoadPointer(&inst.delegate)); instPtr != nil {
		(*instPtr).RecordOne(ctx, number, labels)
	}
}

// Bound instrument initialization

func (bound *syncHandle) RecordOne(ctx context.Context, number number.Number) {
	instPtr := (*metric.SyncImpl)(atomic.LoadPointer(&bound.inst.delegate))
	if instPtr == nil {
		return
	}
	var implPtr *metric.BoundSyncImpl
	bound.initialize.Do(func() {
		implPtr = new(metric.BoundSyncImpl)
		*implPtr = (*instPtr).Bind(bound.labels)
		atomic.StorePointer(&bound.delegate, unsafe.Pointer(implPtr))
	})
	if implPtr == nil {
		implPtr = (*metric.BoundSyncImpl)(atomic.LoadPointer(&bound.delegate))
	}
	// This may still be nil if instrument was created and bound
	// without a delegate, then the instrument was set to have a
	// delegate and unbound.
	if implPtr == nil {
		return
	}
	(*implPtr).RecordOne(ctx, number)
}

func AtomicFieldOffsets() map[string]uintptr {
	return map[string]uintptr{
		"meterProvider.delegate": unsafe.Offsetof(meterProvider{}.delegate),
		"meterImpl.delegate":     unsafe.Offsetof(meterImpl{}.delegate),
		"syncImpl.delegate":      unsafe.Offsetof(syncImpl{}.delegate),
		"asyncImpl.delegate":     unsafe.Offsetof(asyncImpl{}.delegate),
		"syncHandle.delegate":    unsafe.Offsetof(syncHandle{}.delegate),
	}
}
