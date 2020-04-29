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
	"errors"

	"go.opentelemetry.io/otel/api/core"
)

type syncInstrument struct {
	instrument SyncImpl
}

type syncBoundInstrument struct {
	boundInstrument BoundSyncImpl
}

type asyncInstrument struct {
	instrument AsyncImpl
}

var ErrSDKReturnedNilImpl = errors.New("SDK returned a nil implementation")

func (s syncInstrument) bind(labels []core.KeyValue) syncBoundInstrument {
	return newSyncBoundInstrument(s.instrument.Bind(labels))
}

func (s syncInstrument) float64Measurement(value float64) Measurement {
	return newMeasurement(s.instrument, core.NewFloat64Number(value))
}

func (s syncInstrument) int64Measurement(value int64) Measurement {
	return newMeasurement(s.instrument, core.NewInt64Number(value))
}

func (s syncInstrument) directRecord(ctx context.Context, number core.Number, labels []core.KeyValue) {
	s.instrument.RecordOne(ctx, number, labels)
}

func (s syncInstrument) SyncImpl() SyncImpl {
	return s.instrument
}

func (h syncBoundInstrument) directRecord(ctx context.Context, number core.Number) {
	h.boundInstrument.RecordOne(ctx, number)
}

func (h syncBoundInstrument) Unbind() {
	h.boundInstrument.Unbind()
}

func (a asyncInstrument) AsyncImpl() AsyncImpl {
	return a.instrument
}

// checkNewSync receives an SyncImpl and potential
// error, and returns the same types, checking for and ensuring that
// the returned interface is not nil.
func checkNewSync(instrument SyncImpl, err error) (syncInstrument, error) {
	if instrument == nil {
		if err == nil {
			err = ErrSDKReturnedNilImpl
		}
		// Note: an alternate behavior would be to synthesize a new name
		// or group all duplicately-named instruments of a certain type
		// together and use a tag for the original name, e.g.,
		//   name = 'invalid.counter.int64'
		//   label = 'original-name=duplicate-counter-name'
		instrument = NoopSync{}
	}
	return syncInstrument{
		instrument: instrument,
	}, err
}

func newSyncBoundInstrument(boundInstrument BoundSyncImpl) syncBoundInstrument {
	return syncBoundInstrument{
		boundInstrument: boundInstrument,
	}
}

func newMeasurement(instrument SyncImpl, number core.Number) Measurement {
	return Measurement{
		instrument: instrument,
		number:     number,
	}
}

// checkNewAsync receives an AsyncImpl and potential
// error, and returns the same types, checking for and ensuring that
// the returned interface is not nil.
func checkNewAsync(instrument AsyncImpl, err error) (asyncInstrument, error) {
	if instrument == nil {
		if err == nil {
			err = ErrSDKReturnedNilImpl
		}
		instrument = NoopAsync{}
	}
	return asyncInstrument{
		instrument: instrument,
	}, err
}
