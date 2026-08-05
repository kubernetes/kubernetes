/*
 *
 * Copyright 2024 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package stats provides internal stats related functionality.
package stats

import (
	"context"
	"maps"
)

// LabelCallback is a function that is executed when telemetry
// label keys are updated.
type LabelCallback func(map[string]string)
type telemetryLabelCallbackKey struct{}

// UpdateLabels executes registered telemetry callbacks with the update labels. Labels
// are copied before being processed by any callbacks to ensure mutations are not
// shared among derived contexts.
//
// It is the responsibility of the registrant to handle conflicts or label resets.
func UpdateLabels(ctx context.Context, update map[string]string) {
	executeTelemetryLabelCallbacks(ctx, update)
}

// RegisterTelemetryLabelCallback registers a callback function that is executed whenever
// telemetry labels are updated.
func RegisterTelemetryLabelCallback(ctx context.Context, callback LabelCallback) context.Context {
	if callback == nil {
		return ctx
	}

	callbacks, ok := ctx.Value(telemetryLabelCallbackKey{}).([]LabelCallback)
	if !ok {
		return context.WithValue(ctx, telemetryLabelCallbackKey{}, []LabelCallback{callback})
	}
	return context.WithValue(ctx, telemetryLabelCallbackKey{}, append(append([]LabelCallback(nil), callbacks...), callback))

}

// executeTelemetryLabelCallback runs the registered callbacks in the order they were
// registered on the context with the provided labels. If no callbacks are registered
// it does nothing.
//
// To ensure callbacks do not mutate the state of the provided label map it is copied
// before execution.
func executeTelemetryLabelCallbacks(ctx context.Context, labels map[string]string) {
	callbacks, ok := ctx.Value(telemetryLabelCallbackKey{}).([]LabelCallback)
	if !ok {
		return
	}

	labelsCopy := map[string]string{}
	maps.Copy(labelsCopy, labels)
	for _, callback := range callbacks {
		callback(labelsCopy)
	}

}
