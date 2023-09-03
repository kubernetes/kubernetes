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

// Package instrumentation provides types to represent the code libraries that
// provide OpenTelemetry instrumentation. These types are used in the
// OpenTelemetry signal pipelines to identify the source of telemetry.
//
// See
// https://github.com/open-telemetry/oteps/blob/d226b677d73a785523fe9b9701be13225ebc528d/text/0083-component.md
// and
// https://github.com/open-telemetry/oteps/blob/d226b677d73a785523fe9b9701be13225ebc528d/text/0201-scope-attributes.md
// for more information.
package instrumentation // import "go.opentelemetry.io/otel/sdk/instrumentation"
