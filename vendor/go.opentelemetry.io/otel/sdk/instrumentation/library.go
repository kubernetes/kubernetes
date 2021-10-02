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

/*
Package instrumentation provides an instrumentation library structure to be
passed to both the OpenTelemetry Tracer and Meter components.

This package is currently in a pre-GA phase. Backwards incompatible changes
may be introduced in subsequent minor version releases as we work to track the
evolving OpenTelemetry specification and user feedback.

For more information see
[this](https://github.com/open-telemetry/oteps/blob/main/text/0083-component.md).
*/
package instrumentation // import "go.opentelemetry.io/otel/sdk/instrumentation"

// Library represents the instrumentation library.
type Library struct {
	// Name is the name of the instrumentation library. This should be the
	// Go package name of that library.
	Name string
	// Version is the version of the instrumentation library.
	Version string
}
