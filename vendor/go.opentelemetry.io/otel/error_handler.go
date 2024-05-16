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

// ErrorHandler handles irremediable events.
type ErrorHandler interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Handle handles any error deemed irremediable by an OpenTelemetry
	// component.
	Handle(error)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

// ErrorHandlerFunc is a convenience adapter to allow the use of a function
// as an ErrorHandler.
type ErrorHandlerFunc func(error)

var _ ErrorHandler = ErrorHandlerFunc(nil)

// Handle handles the irremediable error by calling the ErrorHandlerFunc itself.
func (f ErrorHandlerFunc) Handle(err error) {
	f(err)
}
