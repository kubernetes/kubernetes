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

package otel // import "go.opentelemetry.io/otel"

import (
	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/propagation"
)

// GetTextMapPropagator returns the global TextMapPropagator. If none has been
// set, a No-Op TextMapPropagator is returned.
func GetTextMapPropagator() propagation.TextMapPropagator {
	return global.TextMapPropagator()
}

// SetTextMapPropagator sets propagator as the global TextMapPropagator.
func SetTextMapPropagator(propagator propagation.TextMapPropagator) {
	global.SetTextMapPropagator(propagator)
}
