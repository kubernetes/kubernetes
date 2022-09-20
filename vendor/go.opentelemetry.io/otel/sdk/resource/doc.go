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

// Package resource provides detecting and representing resources.
//
// The fundamental struct is a Resource which holds identifying information
// about the entities for which telemetry is exported.
//
// To automatically construct Resources from an environment a Detector
// interface is defined. Implementations of this interface can be passed to
// the Detect function to generate a Resource from the merged information.
//
// To load a user defined Resource from the environment variable
// OTEL_RESOURCE_ATTRIBUTES the FromEnv Detector can be used. It will interpret
// the value as a list of comma delimited key/value pairs
// (e.g. `<key1>=<value1>,<key2>=<value2>,...`).
package resource // import "go.opentelemetry.io/otel/sdk/resource"
