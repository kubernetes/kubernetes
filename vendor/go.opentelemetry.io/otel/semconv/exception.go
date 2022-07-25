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
package semconv

import "go.opentelemetry.io/otel/attribute"

// Semantic conventions for exception attribute keys.
const (
	// The Go type containing the error or exception.
	ExceptionTypeKey = attribute.Key("exception.type")

	// The exception message.
	ExceptionMessageKey = attribute.Key("exception.message")

	// A stacktrace as a string. This most commonly will come from
	// "runtime/debug".Stack.
	ExceptionStacktraceKey = attribute.Key("exception.stacktrace")

	// If the exception event is recorded at a point where it is known
	// that the exception is escaping the scope of the span this
	// attribute is set to true.
	ExceptionEscapedKey = attribute.Key("exception.escaped")
)

const (
	// ExceptionEventName is the name of the Span event representing an exception.
	ExceptionEventName = "exception"
)
