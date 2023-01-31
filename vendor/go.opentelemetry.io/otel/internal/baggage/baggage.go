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
Package baggage provides base types and functionality to store and retrieve
baggage in Go context. This package exists because the OpenTracing bridge to
OpenTelemetry needs to synchronize state whenever baggage for a context is
modified and that context contains an OpenTracing span. If it were not for
this need this package would not need to exist and the
`go.opentelemetry.io/otel/baggage` package would be the singular place where
W3C baggage is handled.
*/
package baggage // import "go.opentelemetry.io/otel/internal/baggage"

// List is the collection of baggage members. The W3C allows for duplicates,
// but OpenTelemetry does not, therefore, this is represented as a map.
type List map[string]Item

// Item is the value and metadata properties part of a list-member.
type Item struct {
	Value      string
	Properties []Property
}

// Property is a metadata entry for a list-member.
type Property struct {
	Key, Value string

	// HasValue indicates if a zero-value value means the property does not
	// have a value or if it was the zero-value.
	HasValue bool
}
