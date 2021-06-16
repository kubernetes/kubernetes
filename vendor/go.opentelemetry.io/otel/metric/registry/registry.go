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

package registry // import "go.opentelemetry.io/otel/metric/registry"

import (
	"context"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// MeterProvider is a standard MeterProvider for wrapping `MeterImpl`
type MeterProvider struct {
	impl metric.MeterImpl
}

var _ metric.MeterProvider = (*MeterProvider)(nil)

// uniqueInstrumentMeterImpl implements the metric.MeterImpl interface, adding
// uniqueness checking for instrument descriptors.  Use NewUniqueInstrumentMeter
// to wrap an implementation with uniqueness checking.
type uniqueInstrumentMeterImpl struct {
	lock  sync.Mutex
	impl  metric.MeterImpl
	state map[key]metric.InstrumentImpl
}

var _ metric.MeterImpl = (*uniqueInstrumentMeterImpl)(nil)

type key struct {
	instrumentName         string
	instrumentationName    string
	InstrumentationVersion string
}

// NewMeterProvider returns a new provider that implements instrument
// name-uniqueness checking.
func NewMeterProvider(impl metric.MeterImpl) *MeterProvider {
	return &MeterProvider{
		impl: NewUniqueInstrumentMeterImpl(impl),
	}
}

// Meter implements MeterProvider.
func (p *MeterProvider) Meter(instrumentationName string, opts ...metric.MeterOption) metric.Meter {
	return metric.WrapMeterImpl(p.impl, instrumentationName, opts...)
}

// ErrMetricKindMismatch is the standard error for mismatched metric
// instrument definitions.
var ErrMetricKindMismatch = fmt.Errorf(
	"a metric was already registered by this name with another kind or number type")

// NewUniqueInstrumentMeterImpl returns a wrapped metric.MeterImpl with
// the addition of uniqueness checking.
func NewUniqueInstrumentMeterImpl(impl metric.MeterImpl) metric.MeterImpl {
	return &uniqueInstrumentMeterImpl{
		impl:  impl,
		state: map[key]metric.InstrumentImpl{},
	}
}

// RecordBatch implements metric.MeterImpl.
func (u *uniqueInstrumentMeterImpl) RecordBatch(ctx context.Context, labels []attribute.KeyValue, ms ...metric.Measurement) {
	u.impl.RecordBatch(ctx, labels, ms...)
}

func keyOf(descriptor metric.Descriptor) key {
	return key{
		descriptor.Name(),
		descriptor.InstrumentationName(),
		descriptor.InstrumentationVersion(),
	}
}

// NewMetricKindMismatchError formats an error that describes a
// mismatched metric instrument definition.
func NewMetricKindMismatchError(desc metric.Descriptor) error {
	return fmt.Errorf("metric was %s (%s %s)registered as a %s %s: %w",
		desc.Name(),
		desc.InstrumentationName(),
		desc.InstrumentationVersion(),
		desc.NumberKind(),
		desc.InstrumentKind(),
		ErrMetricKindMismatch)
}

// Compatible determines whether two metric.Descriptors are considered
// the same for the purpose of uniqueness checking.
func Compatible(candidate, existing metric.Descriptor) bool {
	return candidate.InstrumentKind() == existing.InstrumentKind() &&
		candidate.NumberKind() == existing.NumberKind()
}

// checkUniqueness returns an ErrMetricKindMismatch error if there is
// a conflict between a descriptor that was already registered and the
// `descriptor` argument.  If there is an existing compatible
// registration, this returns the already-registered instrument.  If
// there is no conflict and no prior registration, returns (nil, nil).
func (u *uniqueInstrumentMeterImpl) checkUniqueness(descriptor metric.Descriptor) (metric.InstrumentImpl, error) {
	impl, ok := u.state[keyOf(descriptor)]
	if !ok {
		return nil, nil
	}

	if !Compatible(descriptor, impl.Descriptor()) {
		return nil, NewMetricKindMismatchError(impl.Descriptor())
	}

	return impl, nil
}

// NewSyncInstrument implements metric.MeterImpl.
func (u *uniqueInstrumentMeterImpl) NewSyncInstrument(descriptor metric.Descriptor) (metric.SyncImpl, error) {
	u.lock.Lock()
	defer u.lock.Unlock()

	impl, err := u.checkUniqueness(descriptor)

	if err != nil {
		return nil, err
	} else if impl != nil {
		return impl.(metric.SyncImpl), nil
	}

	syncInst, err := u.impl.NewSyncInstrument(descriptor)
	if err != nil {
		return nil, err
	}
	u.state[keyOf(descriptor)] = syncInst
	return syncInst, nil
}

// NewAsyncInstrument implements metric.MeterImpl.
func (u *uniqueInstrumentMeterImpl) NewAsyncInstrument(
	descriptor metric.Descriptor,
	runner metric.AsyncRunner,
) (metric.AsyncImpl, error) {
	u.lock.Lock()
	defer u.lock.Unlock()

	impl, err := u.checkUniqueness(descriptor)

	if err != nil {
		return nil, err
	} else if impl != nil {
		return impl.(metric.AsyncImpl), nil
	}

	asyncInst, err := u.impl.NewAsyncInstrument(descriptor, runner)
	if err != nil {
		return nil, err
	}
	u.state[keyOf(descriptor)] = asyncInst
	return asyncInst, nil
}
