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

// Record records one or multiple measurements with the same context at once.
// If there are any tags in the context, measurements will be tagged with them.
func Record(ctx context.Context, ms ...Measurement) {
	recorder := internal.DefaultRecorder
	if recorder == nil {
		return
	}
	if len(ms) == 0 {
		return
	}
	record := false
	for _, m := range ms {
		if m.m.subscribed() {
			record = true
			break
		}
	}
	if !record {
		return
	}
	recorder(tag.FromContext(ctx), ms)
}

// RecordWithTags records one or multiple measurements at once.
//
// Measurements will be tagged with the tags in the context mutated by the mutators.
// RecordWithTags is useful if you want to record with tag mutations but don't want
// to propagate the mutations in the context.
func RecordWithTags(ctx context.Context, mutators []tag.Mutator, ms ...Measurement) error {
	ctx, err := tag.New(ctx, mutators...)
	if err != nil {
		return err
	}
	Record(ctx, ms...)
	return nil
}
