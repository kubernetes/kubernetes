// Copyright 2018, OpenCensus Authors
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
//

package stats

import (
	"context"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/stats/internal"
	"go.opencensus.io/tag"
)

func init() {
	internal.SubscriptionReporter = func(measure string) {
		mu.Lock()
		measures[measure].subscribe()
		mu.Unlock()
	}
}

// Recorder provides an interface for exporting measurement information from
// the static Record method by using the WithRecorder option.
type Recorder interface {
	// Record records a set of measurements associated with the given tags and attachments.
	// The second argument is a `[]Measurement`.
	Record(*tag.Map, interface{}, map[string]interface{})
}

type recordOptions struct {
	attachments  metricdata.Attachments
	mutators     []tag.Mutator
	measurements []Measurement
	recorder     Recorder
}

// WithAttachments applies provided exemplar attachments.
func WithAttachments(attachments metricdata.Attachments) Options {
	return func(ro *recordOptions) {
		ro.attachments = attachments
	}
}

// WithTags applies provided tag mutators.
func WithTags(mutators ...tag.Mutator) Options {
	return func(ro *recordOptions) {
		ro.mutators = mutators
	}
}

// WithMeasurements applies provided measurements.
func WithMeasurements(measurements ...Measurement) Options {
	return func(ro *recordOptions) {
		ro.measurements = measurements
	}
}

// WithRecorder records the measurements to the specified `Recorder`, rather
// than to the global metrics recorder.
func WithRecorder(meter Recorder) Options {
	return func(ro *recordOptions) {
		ro.recorder = meter
	}
}

// Options apply changes to recordOptions.
type Options func(*recordOptions)

func createRecordOption(ros ...Options) *recordOptions {
	o := &recordOptions{}
	for _, ro := range ros {
		ro(o)
	}
	return o
}

type measurementRecorder = func(tags *tag.Map, measurement []Measurement, attachments map[string]interface{})

// Record records one or multiple measurements with the same context at once.
// If there are any tags in the context, measurements will be tagged with them.
func Record(ctx context.Context, ms ...Measurement) {
	// Record behaves the same as RecordWithOptions, but because we do not have to handle generic functionality
	// (RecordOptions) we can reduce some allocations to speed up this hot path
	if len(ms) == 0 {
		return
	}
	recorder := internal.MeasurementRecorder.(measurementRecorder)
	record := false
	for _, m := range ms {
		if m.desc.subscribed() {
			record = true
			break
		}
	}
	if !record {
		return
	}
	recorder(tag.FromContext(ctx), ms, nil)
	return
}

// RecordWithTags records one or multiple measurements at once.
//
// Measurements will be tagged with the tags in the context mutated by the mutators.
// RecordWithTags is useful if you want to record with tag mutations but don't want
// to propagate the mutations in the context.
func RecordWithTags(ctx context.Context, mutators []tag.Mutator, ms ...Measurement) error {
	return RecordWithOptions(ctx, WithTags(mutators...), WithMeasurements(ms...))
}

// RecordWithOptions records measurements from the given options (if any) against context
// and tags and attachments in the options (if any).
// If there are any tags in the context, measurements will be tagged with them.
func RecordWithOptions(ctx context.Context, ros ...Options) error {
	o := createRecordOption(ros...)
	if len(o.measurements) == 0 {
		return nil
	}
	recorder := internal.DefaultRecorder
	if o.recorder != nil {
		recorder = o.recorder.Record
	}
	if recorder == nil {
		return nil
	}
	record := false
	for _, m := range o.measurements {
		if m.desc.subscribed() {
			record = true
			break
		}
	}
	if !record {
		return nil
	}
	if len(o.mutators) > 0 {
		var err error
		if ctx, err = tag.New(ctx, o.mutators...); err != nil {
			return err
		}
	}
	recorder(tag.FromContext(ctx), o.measurements, o.attachments)
	return nil
}
