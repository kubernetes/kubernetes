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

type recordOptions struct {
	attachments  metricdata.Attachments
	mutators     []tag.Mutator
	measurements []Measurement
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

// Options apply changes to recordOptions.
type Options func(*recordOptions)

func createRecordOption(ros ...Options) *recordOptions {
	o := &recordOptions{}
	for _, ro := range ros {
		ro(o)
	}
	return o
}

// Record records one or multiple measurements with the same context at once.
// If there are any tags in the context, measurements will be tagged with them.
func Record(ctx context.Context, ms ...Measurement) {
	RecordWithOptions(ctx, WithMeasurements(ms...))
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
