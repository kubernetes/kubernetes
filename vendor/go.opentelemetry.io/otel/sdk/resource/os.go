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

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"strings"

	"go.opentelemetry.io/otel/semconv"
)

type osTypeDetector struct{}

// Detect returns a *Resource that describes the operating system type the
// service is running on.
func (osTypeDetector) Detect(ctx context.Context) (*Resource, error) {
	osType := runtimeOS()

	return NewWithAttributes(
		semconv.OSTypeKey.String(strings.ToLower(osType)),
	), nil
}

// WithOSType adds an attribute with the operating system type to the configured Resource.
func WithOSType() Option {
	return WithDetectors(osTypeDetector{})
}
