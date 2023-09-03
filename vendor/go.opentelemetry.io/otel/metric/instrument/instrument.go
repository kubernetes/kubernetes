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

package instrument // import "go.opentelemetry.io/otel/metric/instrument"

// Asynchronous instruments are instruments that are updated within a Callback.
// If an instrument is observed outside of it's callback it should be an error.
//
// This interface is used as a grouping mechanism.
type Asynchronous interface {
	asynchronous()
}

// Synchronous instruments are updated in line with application code.
//
// This interface is used as a grouping mechanism.
type Synchronous interface {
	synchronous()
}
